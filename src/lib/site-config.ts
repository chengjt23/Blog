import { SITE_ENVS, type SiteEnv } from './site-env';
import { withBasePath, withoutBasePath } from './paths';

const rawEnv = import.meta.env.SITE_ENV ?? 'local';
export const siteEnv: SiteEnv = SITE_ENVS.includes(rawEnv as SiteEnv)
  ? (rawEnv as SiteEnv)
  : 'local';

export const isProduction = siteEnv === 'production';
export const isIndexable = isProduction && import.meta.env.PUBLIC_ALLOW_INDEXING === 'true';
const isHosted = siteEnv === 'public-beta' || isProduction;
const siteOrigin = isHosted
  ? import.meta.env.PUBLIC_SITE_URL || 'https://invalid.example'
  : 'http://localhost:4321';

const previewIdentity = {
  name: '研究笔记',
  author: 'Preview Author',
  description: '生成建模、随机过程与最优传输的系统教程与研究笔记。',
};

export const siteConfig = {
  name: isHosted ? import.meta.env.PUBLIC_SITE_NAME || previewIdentity.name : previewIdentity.name,
  author: isHosted
    ? import.meta.env.PUBLIC_AUTHOR_NAME || previewIdentity.author
    : previewIdentity.author,
  description: isHosted
    ? import.meta.env.PUBLIC_SITE_DESCRIPTION || previewIdentity.description
    : previewIdentity.description,
  url: new URL(withBasePath('/'), siteOrigin).toString(),
  language: 'zh-CN',
  email: isHosted ? import.meta.env.PUBLIC_CONTACT_EMAIL || '' : '',
  github: isHosted ? import.meta.env.PUBLIC_GITHUB_URL || '' : '',
  license: isHosted
    ? import.meta.env.PUBLIC_CONTENT_LICENSE || 'all-rights-reserved'
    : 'all-rights-reserved',
  analyticsEnabled: isProduction && import.meta.env.PUBLIC_ANALYTICS_ENABLED === 'true',
  isPreview: !isProduction,
} as const;

export function canonicalUrl(pathname: string): string | undefined {
  if (!isProduction) return undefined;
  return new URL(withBasePath(withoutBasePath(pathname)), siteOrigin).toString();
}
