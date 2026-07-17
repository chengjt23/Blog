import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const series = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/series' }),
  schema: z.object({
    title: z.string().min(3),
    description: z.string().min(20).max(220),
    publishedAt: z.coerce.date().nullable(),
    updatedAt: z.coerce.date(),
    draft: z.boolean(),
    type: z.literal('series-chapter'),
    series: z.enum(['schrodinger-bridge', 'diffusion']),
    order: z.number().int().min(0).max(14),
    slug: z.string().regex(/^[bd]\d{1,2}-[a-z0-9-]+$/),
    tags: z.array(z.string()).min(1),
    authors: z.array(z.string()).min(1),
    license: z.string(),
    math: z.boolean(),
    mermaid: z.boolean(),
    toc: z.boolean(),
    featured: z.boolean(),
    includeInFeed: z.boolean(),
    indexable: z.boolean(),
    scope: z.string().min(10),
  }),
});

export const collections = { series };
