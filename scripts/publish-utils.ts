import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

export function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

export function toPosix(value: string): string {
  return value.split(path.sep).join('/');
}

export function assertInside(parent: string, candidate: string): void {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Refusing path outside publication target: ${candidate}`);
  }
}

export async function readJson<T>(file: string): Promise<T> {
  return JSON.parse(await readFile(file, 'utf8')) as T;
}

export function extractOfficialUrl(note: string): string | undefined {
  const doi = note.match(/10\.\d{4,9}\/[A-Z0-9._;()/:+-]+/i)?.[0]?.replace(/[.,;:)`]+$/, '');
  if (doi) return `https://doi.org/${doi}`;

  const arxiv = note.match(/(?:arXiv(?:\s+identifier)?)[\s:`]*(\d{4}\.\d{4,5})(?:v\d+)?/i)?.[1];
  if (arxiv) return `https://arxiv.org/abs/${arxiv}`;

  return undefined;
}

export function publicText(children: unknown[]): string {
  return children
    .map((child) => {
      if (!child || typeof child !== 'object') return '';
      if ('value' in child && typeof child.value === 'string') return child.value;
      if ('children' in child && Array.isArray(child.children)) return publicText(child.children);
      return '';
    })
    .join('')
    .trim();
}
