import { SITE_ENVS } from '../src/lib/site-env';

const siteEnv = process.env.SITE_ENV ?? 'local';
if (!SITE_ENVS.includes(siteEnv as (typeof SITE_ENVS)[number])) {
  throw new Error(`Unknown SITE_ENV: ${siteEnv}`);
}

const isHosted = siteEnv === 'public-beta' || siteEnv === 'production';
const basePath = process.env.PUBLIC_BASE_PATH ?? '/';
if (!/^\/(?:[^/?#]+(?:\/[^/?#]+)*)?$/.test(basePath) || basePath.split('/').includes('..')) {
  throw new Error(`PUBLIC_BASE_PATH must be a root-relative path: ${basePath}`);
}

if (isHosted) {
  if (!process.env.PUBLIC_SITE_URL) throw new Error(`${siteEnv} requires PUBLIC_SITE_URL.`);
  const url = new URL(process.env.PUBLIC_SITE_URL);
  if (url.protocol !== 'https:') throw new Error(`${siteEnv} site URL must use HTTPS.`);
  if (url.pathname !== '/') {
    throw new Error(
      'PUBLIC_SITE_URL must contain only the origin; use PUBLIC_BASE_PATH for /Blog.',
    );
  }
}

if (siteEnv === 'public-beta') {
  for (const key of [
    'PUBLIC_SITE_NAME',
    'PUBLIC_AUTHOR_NAME',
    'PUBLIC_SITE_DESCRIPTION',
    'PUBLIC_CONTENT_LICENSE',
  ]) {
    if (!process.env[key]) throw new Error(`Public beta requires ${key}.`);
  }
  if (process.env.PUBLIC_BETA_APPROVED !== 'true') {
    throw new Error('Public beta requires explicit PUBLIC_BETA_APPROVED=true.');
  }
  if (process.env.PUBLIC_AUTHOR_NAME === 'Preview Author') {
    throw new Error('Preview identity cannot enter public beta.');
  }
}

if (siteEnv === 'production') {
  const required = [
    'PUBLIC_SITE_URL',
    'PUBLIC_SITE_NAME',
    'PUBLIC_AUTHOR_NAME',
    'PUBLIC_SITE_DESCRIPTION',
    'PUBLIC_CONTENT_LICENSE',
  ];
  for (const key of required) {
    if (!process.env[key]) throw new Error(`Production requires ${key}.`);
  }
  if (process.env.PUBLIC_RELEASE_APPROVED !== 'true') {
    throw new Error('Production requires explicit PUBLIC_RELEASE_APPROVED=true.');
  }
  if (process.env.PUBLIC_AUTHOR_NAME === 'Preview Author') {
    throw new Error('Preview identity cannot enter production.');
  }
}

if (siteEnv !== 'production' && process.env.PUBLIC_ALLOW_INDEXING === 'true') {
  throw new Error(`${siteEnv} cannot enable search-engine indexing.`);
}

console.log(`Environment ${siteEnv} satisfies publication safety rules.`);
