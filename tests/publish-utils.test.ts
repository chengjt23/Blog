import { describe, expect, it } from 'vitest';
import { assertInside, extractOfficialUrl, sha256 } from '../scripts/publish-utils';

describe('publication utilities', () => {
  it('extracts DOI links before arXiv links', () => {
    expect(extractOfficialUrl('DOI: `10.3934/dcds.2014.34.1533`; arXiv: `1308.0215v1`.')).toBe(
      'https://doi.org/10.3934/dcds.2014.34.1533',
    );
  });

  it('extracts arXiv links when no DOI is present', () => {
    expect(extractOfficialUrl('arXiv:`2210.02747v2`.')).toBe('https://arxiv.org/abs/2210.02747');
  });

  it('rejects output path traversal', () => {
    expect(() => assertInside('C:/site/content', 'C:/site/private.txt')).toThrow();
  });

  it('creates stable SHA-256 values', () => {
    expect(sha256('bridge')).toHaveLength(64);
  });
});
