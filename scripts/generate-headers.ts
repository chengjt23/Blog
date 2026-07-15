import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const siteRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const siteEnv = process.env.SITE_ENV ?? 'local';
const noindex = siteEnv !== 'production' || process.env.PUBLIC_ALLOW_INDEXING !== 'true';

const headers = [
  '/*',
  '  X-Content-Type-Options: nosniff',
  '  Referrer-Policy: strict-origin-when-cross-origin',
  '  X-Frame-Options: DENY',
  '  Permissions-Policy: camera=(), microphone=(), geolocation=()',
  "  Content-Security-Policy: default-src 'self'; base-uri 'self'; frame-ancestors 'none'; img-src 'self' data:; font-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'",
  ...(noindex ? ['  X-Robots-Tag: noindex, nofollow'] : []),
  '',
].join('\n');

await writeFile(path.join(siteRoot, 'public/_headers'), headers, 'utf8');
console.log(`Generated security headers for ${siteEnv}; noindex=${noindex}.`);
