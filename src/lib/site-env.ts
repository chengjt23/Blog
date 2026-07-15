export const SITE_ENVS = ['local', 'private-preview', 'public-beta', 'production'] as const;
export type SiteEnv = (typeof SITE_ENVS)[number];
