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
          '14 章系统教程：从前向扩散、DDPM、Score 与 SDE，到采样、Guidance、架构、Flow、离散模型、应用和理论安全。',
        link: withBasePath('/blog/diffusion/'),
        pubDate: new Date('2026-07-17T00:00:00+08:00'),
      },
      {
        title: 'Schrödinger Bridge Blog',
        description: '15 章系统教程：从路径空间熵投影与随机控制到现代神经桥匹配和理论前沿。',
        link: withBasePath('/blog/schrodinger-bridge/'),
        pubDate: new Date('2026-07-17T00:00:00+08:00'),
      },
    ],
    customData: '<language>zh-CN</language>',
  });
