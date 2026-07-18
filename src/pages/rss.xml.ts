import rss from '@astrojs/rss';
import type { APIRoute } from 'astro';
import { withBasePath } from '../lib/paths';
import { siteConfig } from '../lib/site-config';

export const GET: APIRoute = () =>
  rss({
    title: `${siteConfig.name} RSS`,
    description: siteConfig.description,
    site: siteConfig.url,
    items: [
      {
        title: 'Diffusion Blog',
        description:
          '5 章系统 Blog：从前向加噪、DDPM、Score 与连续动力学，进入现代系统、Flow 与少步生成。',
        link: withBasePath('/blog/diffusion/'),
        pubDate: new Date('2026-07-17T00:00:00+08:00'),
      },
      {
        title: 'Schrödinger Bridge Blog',
        description: '5 章系统 Blog：从问题定义与精确结构，进入计算、神经近似和方法选择。',
        link: withBasePath('/blog/schrodinger-bridge/'),
        pubDate: new Date('2026-07-19T00:00:00+08:00'),
      },
    ],
    customData: '<language>zh-CN</language>',
  });
