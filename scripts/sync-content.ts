import { copyFile, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkParse from 'remark-parse';
import remarkStringify from 'remark-stringify';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import {
  assertInside,
  extractOfficialUrl,
  publicText,
  readJson,
  sha256,
  toPosix,
} from './publish-utils';

type Allowlist = {
  version: number;
  sources: string[];
  assets: string[];
  publicDerivations: string[];
  publicCode: string[];
};

type Chapter = {
  order: number;
  source: string;
  slug: string;
  title: string;
  description: string;
  scope: string;
  tags: string[];
  featured: boolean;
  mermaid: boolean;
};

type RouteManifest = {
  version: number;
  series: 'schrodinger-bridge';
  release: {
    publishedAt: string | null;
    updatedAt: string;
    draft: boolean;
    license: string;
    authors: string[];
  };
  chapters: Chapter[];
};

type GeneratedEntry = {
  source: string;
  sourceSha256: string;
  output: string;
  outputSha256: string;
};

type GeneratedManifest = {
  entries: GeneratedEntry[];
};

const siteRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const researchRoot = path.dirname(siteRoot);
const contentRoot = path.join(siteRoot, 'src/content/series/schrodinger-bridge');
const assetRoot = path.join(siteRoot, 'public/images/bridge');
const writeMode = process.argv.includes('--write');

const allowlist = await readJson<Allowlist>(path.join(siteRoot, 'content-allowlist.json'));
const routes = await readJson<RouteManifest>(path.join(siteRoot, 'route-manifest.json'));
const routeBySource = new Map(routes.chapters.map((chapter) => [chapter.source, chapter]));
const routeByBasename = new Map(
  routes.chapters.map((chapter) => [path.basename(chapter.source), chapter]),
);

if (routes.chapters.length !== 5 || allowlist.sources.length !== 5) {
  throw new Error('Bridge publication requires exactly 5 declared chapters.');
}
if (allowlist.assets.length !== 11) {
  throw new Error(
    `Bridge publication requires exactly 11 assets, received ${allowlist.assets.length}.`,
  );
}

for (const source of allowlist.sources) {
  if (!routeBySource.has(source)) throw new Error(`Allowlisted source has no route: ${source}`);
}
for (const chapter of routes.chapters) {
  if (!allowlist.sources.includes(chapter.source)) {
    throw new Error(`Route is not allowlisted: ${chapter.source}`);
  }
}
for (const asset of allowlist.assets) {
  const assetPath = path.join(researchRoot, asset);
  assertInside(path.join(researchRoot, 'bridge/figures'), assetPath);
  await readFile(assetPath);
}

function stripResearchHeader(source: string): string {
  const lines = source.replace(/^\uFEFF/, '').split(/\r?\n/);
  if (lines[0]?.startsWith('# ')) lines.shift();
  while (lines[0]?.trim() === '') lines.shift();
  while (lines[0]?.startsWith('>')) lines.shift();
  while (lines[0]?.trim() === '') lines.shift();
  return lines.join('\n').trimEnd() + '\n';
}

function replaceInlineMathDelimiters(line: string): string {
  return line
    .split(/(`+[^`]*`+)/g)
    .map((segment) =>
      segment.startsWith('`') ? segment : segment.replace(/\\\(/g, '$').replace(/\\\)/g, '$'),
    )
    .join('');
}

function normalizeMathDelimiters(source: string): string {
  let fence: '`' | '~' | undefined;
  return source
    .split(/\r?\n/)
    .map((line) => {
      const fenceMatch = line.match(/^\s*(```+|~~~+)/);
      if (fenceMatch) {
        const marker = fenceMatch[1][0] as '`' | '~';
        fence = fence === marker ? undefined : (fence ?? marker);
        return line;
      }
      if (fence) return line;
      if (line.trim() === '\\[' || line.trim() === '\\]') {
        return `${line.match(/^\s*/)?.[0] ?? ''}$$`;
      }
      return replaceInlineMathDelimiters(line);
    })
    .join('\n');
}

function replaceLinkWithText(
  node: { children: unknown[] },
  index: number,
  parent: { children: unknown[] },
): void {
  parent.children[index] = {
    type: 'text',
    value: `${publicText(node.children)}（补充材料暂未公开）`,
  };
}

async function transformMarkdown(chapter: Chapter, source: string): Promise<string> {
  const normalized = normalizeMathDelimiters(stripResearchHeader(source));
  const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(normalized);

  visit(tree, 'image', (node) => {
    if (/^(?:\.\/)?figures\//.test(node.url) || /^(?:\.\.\/)+bridge\/figures\//.test(node.url)) {
      node.url = `/images/bridge/${path.basename(node.url)}`;
      return;
    }
    if (!/^(https?:|\/)/.test(node.url)) {
      throw new Error(`${chapter.source}: unrecognized image URL ${node.url}`);
    }
  });

  const pending: Promise<void>[] = [];
  visit(tree, 'link', (node, index, parent) => {
    if (/^(https?:|mailto:|#)/.test(node.url) || !parent || typeof index !== 'number') return;

    const cleanUrl = node.url.split('#')[0];
    const basename = path.basename(cleanUrl);
    const chapterTarget = routeByBasename.get(basename);
    if (chapterTarget) {
      const anchor = node.url.includes('#') ? `#${node.url.split('#')[1]}` : '';
      node.url = `/blog/schrodinger-bridge/${chapterTarget.slug}/${anchor}`;
      return;
    }

    if (/^(?:\.\.\/)+references\/notes\/(?:sources|bridge)\//.test(cleanUrl)) {
      pending.push(
        (async () => {
          const notePath = path.resolve(
            path.dirname(path.join(researchRoot, chapter.source)),
            cleanUrl,
          );
          assertInside(path.join(researchRoot, 'references/notes'), notePath);
          const note = await readFile(notePath, 'utf8');
          const officialUrl = extractOfficialUrl(note);
          if (officialUrl) {
            node.url = officialUrl;
            node.title = '官方论文页面';
          } else {
            replaceLinkWithText(node, index, parent);
          }
        })(),
      );
      return;
    }

    if (/^(?:\.\.\/)+references\//.test(cleanUrl)) {
      replaceLinkWithText(node, index, parent);
      return;
    }

    throw new Error(`${chapter.source}: unrecognized relative link ${node.url}`);
  });

  await Promise.all(pending);
  const body = unified()
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkStringify, { bullet: '-', fences: true, listItemIndent: 'one' })
    .stringify(tree);

  return matter.stringify(body, {
    title: chapter.title,
    description: chapter.description,
    publishedAt: routes.release.publishedAt,
    updatedAt: routes.release.updatedAt,
    draft: routes.release.draft,
    type: 'series-chapter',
    series: routes.series,
    order: chapter.order,
    slug: chapter.slug,
    tags: chapter.tags,
    authors: routes.release.authors,
    license: routes.release.license,
    math: true,
    mermaid: chapter.mermaid,
    toc: true,
    featured: chapter.featured,
    includeInFeed: false,
    indexable: true,
    scope: chapter.scope,
  });
}

const generated: GeneratedEntry[] = [];
const referencedAssets = new Set<string>();
for (const chapter of [...routes.chapters].sort((a, b) => a.order - b.order)) {
  const sourcePath = path.join(researchRoot, chapter.source);
  const outputPath = path.join(contentRoot, `${chapter.slug}.md`);
  assertInside(contentRoot, outputPath);
  const source = await readFile(sourcePath, 'utf8');

  for (const match of source.matchAll(
    /!\[[^\]]*\]\((?:(?:\.\/)?figures\/|(?:\.\.\/)+bridge\/figures\/)([^)\s]+)[^)]*\)/g,
  )) {
    const asset = `bridge/figures/${match[1]}`;
    if (!allowlist.assets.includes(asset)) {
      throw new Error(`${chapter.source}: image is not allowlisted: ${asset}`);
    }
    referencedAssets.add(asset);
  }

  const output = await transformMarkdown(chapter, source);
  generated.push({
    source: chapter.source,
    sourceSha256: sha256(source),
    output: toPosix(path.relative(siteRoot, outputPath)),
    outputSha256: sha256(output),
  });

  if (writeMode) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, output, 'utf8');
  }
}

if (referencedAssets.size !== allowlist.assets.length) {
  const unreferenced = allowlist.assets.filter((asset) => !referencedAssets.has(asset));
  throw new Error(`Bridge allowlist contains unreferenced assets: ${unreferenced.join(', ')}`);
}

for (const asset of allowlist.assets) {
  const sourcePath = path.join(researchRoot, asset);
  const outputPath = path.join(assetRoot, path.basename(asset));
  assertInside(assetRoot, outputPath);
  const source = await readFile(sourcePath);
  generated.push({
    source: asset,
    sourceSha256: sha256(source),
    output: toPosix(path.relative(siteRoot, outputPath)),
    outputSha256: sha256(source),
  });
  if (writeMode) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await copyFile(sourcePath, outputPath);
  }
}

if (writeMode) {
  const expectedContentFiles = new Set(routes.chapters.map((chapter) => `${chapter.slug}.md`));
  for (const name of await readdir(contentRoot)) {
    if (!name.endsWith('.md') || expectedContentFiles.has(name)) continue;
    const stalePath = path.join(contentRoot, name);
    assertInside(contentRoot, stalePath);
    await rm(stalePath, { force: true });
  }

  const expectedAssetFiles = new Set(allowlist.assets.map((asset) => path.basename(asset)));
  for (const name of await readdir(assetRoot)) {
    if (expectedAssetFiles.has(name)) continue;
    const stalePath = path.join(assetRoot, name);
    assertInside(assetRoot, stalePath);
    await rm(stalePath, { force: true });
  }

  let existingEntries: GeneratedEntry[] = [];
  let staleBridgeEntries: GeneratedEntry[] = [];
  try {
    const existing = await readJson<GeneratedManifest>(
      path.join(siteRoot, 'generated-manifest.json'),
    );
    staleBridgeEntries = existing.entries.filter(
      (entry) =>
        entry.output.startsWith('src/content/series/schrodinger-bridge/') ||
        entry.output.startsWith('public/images/bridge/'),
    );
    existingEntries = existing.entries.filter(
      (entry) =>
        !entry.output.startsWith('src/content/series/schrodinger-bridge/') &&
        !entry.output.startsWith('public/images/bridge/'),
    );
  } catch {
    existingEntries = [];
  }

  const currentOutputs = new Set(generated.map((entry) => entry.output));
  for (const stale of staleBridgeEntries) {
    if (currentOutputs.has(stale.output)) continue;
    const stalePath = path.join(siteRoot, stale.output);
    if (stale.output.startsWith('src/content/series/schrodinger-bridge/')) {
      assertInside(contentRoot, stalePath);
    } else if (stale.output.startsWith('public/images/bridge/')) {
      assertInside(assetRoot, stalePath);
    } else {
      continue;
    }
    await rm(stalePath, { force: true });
  }

  const entries = [...existingEntries, ...generated];
  const generatedManifest = {
    version: 2,
    transformerVersion: '2.0.0',
    allowlistVersion: allowlist.version,
    routeManifestVersion: routes.version,
    entries,
  };
  await writeFile(
    path.join(siteRoot, 'generated-manifest.json'),
    JSON.stringify(generatedManifest, null, 2) + '\n',
    'utf8',
  );
  console.log(
    `Generated ${routes.chapters.length} Bridge chapters and ${allowlist.assets.length} assets; manifest now contains ${entries.length} entries.`,
  );
} else {
  console.log(
    `Dry run: ${routes.chapters.length} Bridge chapters and ${allowlist.assets.length} assets are ready.`,
  );
  console.log('Run npm run sync-content:write to update all publication outputs.');
}
