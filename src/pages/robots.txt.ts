import type { APIRoute } from 'astro';
import { withBasePath } from '../lib/paths';
import { isIndexable, siteConfig } from '../lib/site-config';

export const GET: APIRoute = () => {
  const body = isIndexable
    ? `User-agent: *\nAllow: ${withBasePath('/')}\nSitemap: ${new URL(withBasePath('/sitemap-index.xml'), siteConfig.url)}\n`
    : 'User-agent: *\nDisallow: /\n';
  return new Response(body, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
};
