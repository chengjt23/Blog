import { getCollection, type CollectionEntry } from 'astro:content';
import { withBasePath } from './paths';
import { isProduction } from './site-config';

export type ChapterEntry = CollectionEntry<'series'>;
export type SeriesId = ChapterEntry['data']['series'];

export async function getChapters(
  series: SeriesId = 'schrodinger-bridge',
): Promise<ChapterEntry[]> {
  const entries = await getCollection(
    'series',
    ({ data }) => data.series === series && (!isProduction || !data.draft),
  );
  return entries.sort((a, b) => a.data.order - b.data.order);
}

export function chapterUrl(entry: ChapterEntry): string {
  return withBasePath(`/blog/${entry.data.series}/${entry.data.slug}/`);
}
