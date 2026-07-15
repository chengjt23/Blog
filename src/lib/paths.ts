export function normalizeBasePath(value: string | undefined): string {
  const trimmed = (value ?? '').trim();
  if (!trimmed || trimmed === '/') return '';

  const normalized = `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
  if (
    normalized.includes('?') ||
    normalized.includes('#') ||
    normalized.split('/').includes('..')
  ) {
    throw new Error(`Invalid base path: ${value}`);
  }
  return normalized;
}

export const basePath = normalizeBasePath(import.meta.env.BASE_URL);

export function withBasePath(pathname: string, base = basePath): string {
  if (!pathname.startsWith('/')) throw new Error(`Expected a root-relative path: ${pathname}`);
  const normalizedBase = normalizeBasePath(base);
  if (!normalizedBase) return pathname;
  if (pathname === normalizedBase || pathname.startsWith(`${normalizedBase}/`)) return pathname;
  return pathname === '/' ? `${normalizedBase}/` : `${normalizedBase}${pathname}`;
}

export function withoutBasePath(pathname: string, base = basePath): string {
  const normalizedBase = normalizeBasePath(base);
  if (!normalizedBase) return pathname;
  if (pathname === normalizedBase || pathname === `${normalizedBase}/`) return '/';
  return pathname.startsWith(`${normalizedBase}/`)
    ? pathname.slice(normalizedBase.length)
    : pathname;
}
