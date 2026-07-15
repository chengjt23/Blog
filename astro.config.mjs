// @ts-check
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import { defineConfig } from 'astro/config';
import { readFileSync } from 'node:fs';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import remarkMath from 'remark-math';
import rehypePublicImages from './scripts/rehype-public-images.mjs';

const siteEnv = process.env.SITE_ENV ?? 'local';
const isProduction = siteEnv === 'production';
const isHosted = siteEnv === 'public-beta' || isProduction;
const site = isHosted
  ? (process.env.PUBLIC_SITE_URL ?? 'https://invalid.example')
  : 'http://localhost:4321';
const basePath = normalizeBasePath(process.env.PUBLIC_BASE_PATH);
/** @type {{ chapters: Array<{ slug: string }> }} */
const routeManifest = JSON.parse(
  readFileSync(new URL('./route-manifest.json', import.meta.url), 'utf8'),
);
const redirects = Object.fromEntries([
  ['/series', withBasePath('/blog')],
  ['/series/schrodinger-bridge', withBasePath('/blog/schrodinger-bridge')],
  ['/about', withBasePath('/')],
  ...routeManifest.chapters.map((chapter) => [
    `/series/schrodinger-bridge/${chapter.slug}`,
    withBasePath(`/blog/schrodinger-bridge/${chapter.slug}`),
  ]),
]);

/** @param {string | undefined} value */
function normalizeBasePath(value) {
  const trimmed = (value ?? '').trim();
  if (!trimmed || trimmed === '/') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

/** @param {string} pathname */
function withBasePath(pathname) {
  return basePath ? (pathname === '/' ? `${basePath}/` : `${basePath}${pathname}`) : pathname;
}

export default defineConfig({
  site,
  base: basePath || '/',
  output: 'static',
  trailingSlash: 'always',
  redirects,
  integrations: isProduction ? [sitemap()] : [],
  markdown: {
    processor: unified({
      gfm: true,
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        rehypeSlug,
        [rehypePublicImages, { basePath }],
        [rehypeAutolinkHeadings, { behavior: 'wrap' }],
      ],
    }),
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      wrap: true,
    },
  },
  vite: {
    build: {
      cssMinify: 'lightningcss',
    },
  },
});
