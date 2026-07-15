import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { load } from 'cheerio';

const siteRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const distRoot = path.join(siteRoot, 'dist');
const errors: string[] = [];
const basePath = normalizeBasePath(process.env.PUBLIC_BASE_PATH);
const mustNoIndex =
  process.env.SITE_ENV !== 'production' || process.env.PUBLIC_ALLOW_INDEXING !== 'true';

function normalizeBasePath(value: string | undefined): string {
  const trimmed = (value ?? '').trim();
  if (!trimmed || trimmed === '/') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

function withBasePath(pathname: string): string {
  return basePath ? (pathname === '/' ? `${basePath}/` : `${basePath}${pathname}`) : pathname;
}

function withoutBasePath(pathname: string): string {
  if (!basePath) return pathname;
  if (pathname === basePath || pathname === `${basePath}/`) return '/';
  return pathname.startsWith(`${basePath}/`) ? pathname.slice(basePath.length) : pathname;
}

async function walk(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries.map((entry) => {
      const target = path.join(directory, entry.name);
      return entry.isDirectory() ? walk(target) : Promise.resolve([target]);
    }),
  );
  return files.flat();
}

function routeFor(file: string): string {
  const relative = path.relative(distRoot, file).split(path.sep).join('/');
  if (relative === 'index.html') return '/';
  if (relative.endsWith('/index.html')) return `/${relative.slice(0, -'index.html'.length)}`;
  return `/${relative}`;
}

function outputFor(pathname: string): string {
  const clean = decodeURIComponent(withoutBasePath(pathname)).replace(/^\//, '');
  if (!clean) return path.join(distRoot, 'index.html');
  if (path.extname(clean)) return path.join(distRoot, clean);
  return path.join(distRoot, clean, 'index.html');
}

async function exists(file: string): Promise<boolean> {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

const htmlFiles = (await walk(distRoot)).filter(
  (file) => file.endsWith('.html') && !file.includes(`${path.sep}pagefind${path.sep}`),
);
let redirectCount = 0;

for (const file of htmlFiles) {
  const route = routeFor(file);
  const html = await readFile(file, 'utf8');
  const $ = load(html);
  const refresh = $('meta[http-equiv="refresh"]').attr('content');
  const refreshMatch = refresh?.match(/^\s*\d+\s*;\s*url=(.+?)\s*$/i);
  const redirectTarget = refreshMatch?.[1]?.replace(/^['"]|['"]$/g, '');
  const isRedirect = refresh !== undefined;

  if (isRedirect) {
    redirectCount += 1;
    const fallbackTarget = $('body a[href]').first().attr('href');
    if (!redirectTarget?.startsWith('/')) {
      errors.push(`${route}: redirect target must be a valid root-relative URL`);
    } else if (
      basePath &&
      redirectTarget !== basePath &&
      !redirectTarget.startsWith(`${basePath}/`)
    ) {
      errors.push(`${route}: redirect target is missing deployment base ${basePath}`);
    } else if (fallbackTarget !== redirectTarget) {
      errors.push(`${route}: redirect fallback link does not match ${redirectTarget}`);
    }
  } else if ($('h1').length !== 1) {
    errors.push(`${route}: expected one H1, found ${$('h1').length}`);
  }
  const robots = $('meta[name="robots"]').attr('content') ?? '';
  if (mustNoIndex && !robots.includes('noindex')) {
    errors.push(`${route}: preview builds must be noindex`);
  }
  if (!mustNoIndex && robots.includes('noindex')) {
    errors.push(`${route}: approved production content must be indexable`);
  }

  $('img').each((_index, image) => {
    const src = $(image).attr('src') ?? '(missing src)';
    if (!$(image).attr('alt')) errors.push(`${route}: image has no alt: ${src}`);
    if (!$(image).attr('width') || !$(image).attr('height')) {
      errors.push(`${route}: image has no stable dimensions: ${src}`);
    }
  });

  if (
    /formal_draft_complete|source_verified|references\/papers|bridge\/staging|D:\\\\/.test(html)
  ) {
    errors.push(`${route}: contains private or research-only text`);
  }

  const targets = [
    ...$('a[href]')
      .map((_index, node) => $(node).attr('href'))
      .get(),
    ...$('link[href]')
      .map((_index, node) => $(node).attr('href'))
      .get(),
    ...$('script[src]')
      .map((_index, node) => $(node).attr('src'))
      .get(),
    ...$('img[src]')
      .map((_index, node) => $(node).attr('src'))
      .get(),
  ].filter((value): value is string => Boolean(value));

  for (const target of targets) {
    if (/^(https?:|mailto:|data:|\/\/)/.test(target)) continue;
    if (
      basePath &&
      target.startsWith('/') &&
      target !== basePath &&
      !target.startsWith(`${basePath}/`)
    ) {
      errors.push(`${route}: internal target is missing deployment base ${basePath}: ${target}`);
      continue;
    }
    const resolved = new URL(target, `https://local.invalid${withBasePath(route)}`);
    const output = outputFor(resolved.pathname);
    if (!(await exists(output))) errors.push(`${route}: broken internal target ${target}`);
  }
}

if (errors.length) {
  throw new Error(`Build audit failed:\n${errors.join('\n')}`);
}

console.log(
  `Audited ${htmlFiles.length - redirectCount} content pages and ${redirectCount} redirects: links, H1, images, robots, and leakage passed.`,
);
