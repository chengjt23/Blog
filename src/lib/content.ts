import { getCollection, type CollectionEntry } from 'astro:content';
import { withBasePath } from './paths';
import { isProduction } from './site-config';

export type ChapterEntry = CollectionEntry<'series'>;

export async function getChapters(): Promise<ChapterEntry[]> {
  const entries = await getCollection('series', ({ data }) => !isProduction || !data.draft);
  return entries.sort((a, b) => a.data.order - b.data.order);
}

export function chapterUrl(entry: ChapterEntry): string {
  return withBasePath(`/blog/schrodinger-bridge/${entry.data.slug}/`);
}
