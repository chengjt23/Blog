import { describe, expect, it } from 'vitest';
import { normalizeBasePath, withBasePath, withoutBasePath } from '../src/lib/paths';

describe('base path utilities', () => {
  it('normalizes root and repository paths', () => {
    expect(normalizeBasePath('/')).toBe('');
    expect(normalizeBasePath('/Blog/')).toBe('/Blog');
  });

  it('adds a repository base without duplicating it', () => {
    expect(withBasePath('/blog/', '/Blog')).toBe('/Blog/blog/');
    expect(withBasePath('/Blog/blog/', '/Blog')).toBe('/Blog/blog/');
    expect(withBasePath('/', '/Blog')).toBe('/Blog/');
  });

  it('removes a repository base for route matching', () => {
    expect(withoutBasePath('/Blog/blog/', '/Blog')).toBe('/blog/');
    expect(withoutBasePath('/Blog/', '/Blog')).toBe('/');
  });
});
