import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
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

const coreMath: Record<string, Array<{ startsWith: string; latex: string }>> = {
  'b0-schrodinger-problem-timeline': [
    {
      startsWith: 'p(X_t=z|X_0=x,X_T=y)',
      latex: String.raw`\mathbb{P}(X_t=z\mid X_0=x,X_T=y)=\frac{r_{0,t}(x,z)\,r_{t,T}(z,y)}{r_{0,T}(x,y)}. \tag{2.1}`,
    },
    {
      startsWith: 'P_0=mu_0',
      latex: String.raw`P_0=\mu_0,\qquad P_T=\mu_T. \tag{2.2}`,
    },
  ],
  'b2-path-space-schrodinger-problem': [
    {
      startsWith: 'minimize H(P|R)',
      latex: String.raw`\min_{P}\ H(P\mid R)\quad\text{subject to}\quad P_0=\mu_0,\;P_T=\mu_T. \tag{1.1}`,
    },
    {
      startsWith: 'H(P|R)\n = H(gamma|R_0T)',
      latex: String.raw`H(P\mid R)=H(\gamma\mid R_{0T})+\int H(P^{xy}\mid R^{xy})\,\gamma(dx\,dy). \tag{4.1}`,
    },
  ],
  'b3-static-schrodinger-system': [
    {
      startsWith: 'inf_{gamma in Pi(mu_0,mu_T)}',
      latex: String.raw`\inf_{\gamma\in\Pi(\mu_0,\mu_T)} H(\gamma\mid R_{0T}). \tag{1.1}`,
    },
    {
      startsWith: 'gamma*(i,j)=u_i R_ij v_j',
      latex: String.raw`\gamma^*(i,j)=u_i\,R_{ij}\,v_j. \tag{1.2}`,
    },
  ],
  'b5-schrodinger-bridge-entropic-ot': [
    {
      startsWith: 'inf_{gamma in Pi(mu_0,mu_T)} integral c(x,y)',
      latex: String.raw`\inf_{\gamma\in\Pi(\mu_0,\mu_T)}\int c(x,y)\,\gamma(dx\,dy). \tag{1.1}`,
    },
    {
      startsWith: 'inf { H(P|R) : P_0=mu_0',
      latex: String.raw`\inf\left\{H(P\mid R):P_0=\mu_0,\;P_T=\mu_T\right\}. \tag{1.2}`,
    },
  ],
  'b6-stochastic-control-girsanov-follmer': [
    {
      startsWith: 'minimize  H(P|R)',
      latex: String.raw`\min_{P}\ H(P\mid R)\quad\text{subject to}\quad P_0=\mu_0,\;P_T=\mu_T. \tag{1.1}`,
    },
    {
      startsWith: 'minimize  (1/2) E_P integral_0^T',
      latex: String.raw`\min_u\ \frac{1}{2}\,\mathbb{E}_P\!\left[\int_0^T\lVert u_t\rVert^2\,dt\right]. \tag{1.2}`,
    },
  ],
  'b9-bridge-matching-markovian-projection-imf': [
    {
      startsWith: 'Pi_gamma = integral Q^{xy}',
      latex: String.raw`\Pi_\gamma=\int Q^{xy}\,\gamma(dx\,dy). \tag{1.1}`,
    },
    {
      startsWith: 'b_M(t,x)=E[Y_t|X_t=x]',
      latex: String.raw`b_M(t,x)=\mathbb{E}[Y_t\mid X_t=x]. \tag{3.1}`,
    },
  ],
  'b12-diffusion-flow-matching-unification': [
    {
      startsWith: 's_t(x)=grad log p_t(x)',
      latex: String.raw`s_t(x)=\nabla\log p_t(x)=\mathbb{E}\!\left[\nabla\log p_t(x\mid X_0)\mid X_t=x\right]. \tag{2.1}`,
    },
    {
      startsWith: 'v_t(x)=f_t(x)-(g_t^2/2)s_t(x)',
      latex: String.raw`v_t(x)=f_t(x)-\frac{g_t^2}{2}\,s_t(x). \tag{2.3}`,
    },
  ],
};

if (routes.chapters.length !== 15 || allowlist.sources.length !== 15) {
  throw new Error('Bridge publication requires exactly 15 declared chapters.');
}

for (const source of allowlist.sources) {
  if (!routeBySource.has(source)) throw new Error(`Allowlisted source has no route: ${source}`);
}

function stripResearchHeader(source: string): string {
  const lines = source.replace(/^\uFEFF/, '').split(/\r?\n/);
  if (lines[0]?.startsWith('# ')) lines.shift();
  while (lines[0]?.trim() === '') lines.shift();
  while (lines[0]?.startsWith('>')) lines.shift();
  while (lines[0]?.trim() === '') lines.shift();
  return lines.join('\n').trimEnd() + '\n';
}

async function transformMarkdown(chapter: Chapter, source: string): Promise<string> {
  const tree = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .parse(stripResearchHeader(source));

  visit(tree, 'code', (node, index, parent) => {
    if (node.lang !== 'text' || !parent || typeof index !== 'number') return;
    const conversion = coreMath[chapter.slug]?.find(({ startsWith }) =>
      node.value.trim().startsWith(startsWith),
    );
    if (conversion) parent.children[index] = { type: 'math', value: conversion.latex };
  });

  const pending: Promise<void>[] = [];
  visit(tree, 'image', (node) => {
    if (node.url.startsWith('figures/')) {
      node.url = `/images/bridge/${path.basename(node.url)}`;
      return;
    }
    if (!/^(https?:|\/)/.test(node.url)) {
      throw new Error(`${chapter.source}: unrecognized image URL ${node.url}`);
    }
  });

  visit(tree, 'link', (node, index, parent) => {
    if (/^(https?:|mailto:|#)/.test(node.url)) return;

    const basename = path.basename(node.url.split('#')[0]);
    const chapterTarget = routeByBasename.get(basename);
    if (chapterTarget) {
      const anchor = node.url.includes('#') ? `#${node.url.split('#')[1]}` : '';
      node.url = `/blog/schrodinger-bridge/${chapterTarget.slug}/${anchor}`;
      return;
    }

    if (node.url.startsWith('../references/notes/bridge/')) {
      pending.push(
        (async () => {
          const notePath = path.resolve(
            path.dirname(path.join(researchRoot, chapter.source)),
            node.url,
          );
          assertInside(path.join(researchRoot, 'references/notes/bridge'), notePath);
          const note = await readFile(notePath, 'utf8');
          const officialUrl = extractOfficialUrl(note);
          if (officialUrl) {
            node.url = officialUrl;
            node.title = '官方论文页面';
          } else if (parent && typeof index === 'number') {
            parent.children[index] = { type: 'text', value: publicText(node.children) };
          }
        })(),
      );
      return;
    }

    if (
      node.url.startsWith('../references/notes/derivations/') ||
      node.url.startsWith('../references/code/')
    ) {
      if (parent && typeof index === 'number') {
        parent.children[index] = {
          type: 'text',
          value: `${publicText(node.children)}（补充材料暂未公开）`,
        };
      }
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

  const publicBody = body
    .replace(/`?formal_draft_complete_v0\.1`?/g, '公开预览版 v0.1')
    .replace(/`?source_verified_bounded`?/g, '已完成有界范围的来源核验')
    .replace(/`?source_verified`?/g, '已完成来源核验');

  return matter.stringify(publicBody, {
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
for (const chapter of routes.chapters.sort((a, b) => a.order - b.order)) {
  if (!allowlist.sources.includes(chapter.source)) {
    throw new Error(`Route is not allowlisted: ${chapter.source}`);
  }
  const sourcePath = path.join(researchRoot, chapter.source);
  const outputPath = path.join(contentRoot, `${chapter.slug}.md`);
  assertInside(contentRoot, outputPath);
  const source = await readFile(sourcePath, 'utf8');
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

const generatedManifest = {
  version: 1,
  transformerVersion: '1.0.0',
  allowlistVersion: allowlist.version,
  routeManifestVersion: routes.version,
  entries: generated,
};

if (writeMode) {
  await writeFile(
    path.join(siteRoot, 'generated-manifest.json'),
    JSON.stringify(generatedManifest, null, 2) + '\n',
    'utf8',
  );
  console.log(
    `Generated ${routes.chapters.length} chapters and ${allowlist.assets.length} assets.`,
  );
} else {
  console.log(
    `Dry run: ${routes.chapters.length} chapters and ${allowlist.assets.length} assets are ready.`,
  );
  console.log('Run npm run sync-content:write to update publication outputs.');
}
