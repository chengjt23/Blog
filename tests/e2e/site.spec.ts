import { expect, test } from '@playwright/test';

const configuredBase = process.env.PUBLIC_BASE_PATH ?? '';
const basePath =
  !configuredBase || configuredBase === '/' ? '' : `/${configuredBase.replace(/^\/+|\/+$/g, '')}`;
const pagePath = (pathname: string) => `${basePath}${pathname}`;

test('desktop Home is empty and Blog titles open directly', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(pagePath('/'));
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Home');
  await expect(page.locator('.empty-home')).toBeVisible();
  await expect(page.locator('.site-mark')).toHaveText('R');
  await expect(page.locator('.site-mark-label')).toHaveCount(0);
  await expect(page.locator('.main-nav a')).toHaveCount(2);
  await expect(page.locator('.main-nav')).toContainText('Home');
  await expect(page.locator('.main-nav')).toContainText('Blog');
  await expect(page.locator('#theme-toggle')).toHaveCount(0);
  await expect(page.locator('.site-header a[href="/search/"]')).toHaveCount(0);
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content', 'noindex,nofollow');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1440);
  await page.screenshot({ path: 'artifacts/qa/home-desktop.png', fullPage: true });

  await page.getByRole('link', { name: 'Blog' }).click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Blog');
  const diffusionTitle = page.getByRole('link', { name: 'Diffusion', exact: true });
  await expect(diffusionTitle).toHaveAttribute('href', pagePath('/blog/diffusion/'));
  const blogTitle = page.getByRole('link', { name: 'Schrödinger Bridge', exact: true });
  await expect(blogTitle).toHaveAttribute('href', pagePath('/blog/schrodinger-bridge/'));
  await page.screenshot({ path: 'artifacts/qa/blog-desktop.png', fullPage: true });
  await blogTitle.click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Schrödinger Bridge');
  await expect(page.locator('.chapter-row')).toHaveCount(5);
});

test('Diffusion Blog publishes all chapters with math, images, and direct navigation', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(pagePath('/blog/'));
  await page.getByRole('link', { name: 'Diffusion', exact: true }).click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Diffusion');
  await expect(page.locator('.chapter-row')).toHaveCount(5);
  await page.screenshot({ path: 'artifacts/qa/diffusion-index-desktop.png', fullPage: true });

  await page.getByRole('link', { name: /去噪网络真正学到的是什么/ }).click();
  await expect(page.getByRole('heading', { level: 1 })).toContainText('去噪网络真正学到的是什么');
  await expect(page.locator('.chapter-meta')).toContainText('正式版 v1.0');
  expect(await page.locator('.katex-display').count()).toBeGreaterThanOrEqual(5);
  const images = page.locator('.article-content img');
  for (let index = 0; index < (await images.count()); index += 1) {
    const image = images.nth(index);
    await image.scrollIntoViewIfNeeded();
    await expect
      .poll(() =>
        image.evaluate((node) => {
          const imageNode = node as HTMLImageElement;
          return imageNode.complete && imageNode.naturalWidth > 0;
        }),
      )
      .toBe(true);
  }
  await expect(page.locator('.series-nav-list').first()).toContainText('D5');
  await page.screenshot({ path: 'artifacts/qa/diffusion-chapter-viewport.png' });
  await expect(page.getByRole('link', { name: /下一篇 · D4/ })).toHaveAttribute(
    'href',
    /d4-modern-diffusion-system/,
  );
});

test('Bridge chapter navigation, math, images, and dark theme work', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto(pagePath('/blog/schrodinger-bridge/b2-how-exact-bridge-is-formed/'));
  await expect(page.locator('.series-rail')).toBeVisible();
  await expect(page.locator('.toc-rail')).toBeVisible();
  expect(await page.locator('.katex-display').count()).toBeGreaterThanOrEqual(10);
  await expect(page.locator('.article-content pre')).toHaveCount(0);

  const images = page.locator('.article-content img');
  for (let index = 0; index < (await images.count()); index += 1) {
    const image = images.nth(index);
    await image.scrollIntoViewIfNeeded();
    await expect
      .poll(() =>
        image.evaluate((node) => {
          const imageNode = node as HTMLImageElement;
          return imageNode.complete && imageNode.naturalWidth > 0;
        }),
      )
      .toBe(true);
  }

  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  const articleColor = await page
    .locator('.article-content')
    .evaluate((node) => getComputedStyle(node).color);
  expect(articleColor).not.toBe('rgb(36, 41, 46)');
  await page.screenshot({ path: 'artifacts/qa/chapter-dark-viewport.png' });
  await page.screenshot({ path: 'artifacts/qa/chapter-dark-desktop.png', fullPage: true });

  const next = page.getByRole('link', { name: /下一篇/ });
  await expect(next).toHaveAttribute('href', /b3-how-exact-bridge-is-computed/);
});

test('mobile navigation and article layout do not overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(pagePath('/blog/schrodinger-bridge/b5-when-schrodinger-bridge-is-needed/'));
  await expect(page.locator('.mobile-series-nav')).toBeVisible();
  await expect(page.locator('.series-rail')).toBeHidden();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  await page.getByRole('button', { name: '打开导航' }).click();
  await expect(page.locator('#main-nav')).toHaveAttribute('data-open', 'true');
  const headerBox = await page.locator('.site-header').boundingBox();
  const titleBox = await page.getByRole('heading', { level: 1 }).boundingBox();
  expect(titleBox?.y ?? 0).toBeGreaterThanOrEqual((headerBox?.height ?? 0) - 1);
  await page.screenshot({ path: 'artifacts/qa/chapter-mobile-viewport.png' });
  await page.screenshot({ path: 'artifacts/qa/chapter-mobile.png', fullPage: true });
});

test('Pagefind returns Chinese and English technical results', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 800 });
  await page.goto(pagePath('/search/'));
  const input = page.getByRole('searchbox', { name: '关键词' });
  await input.fill('Markovian projection');
  const firstResult = page.locator('#search-results li').first();
  await expect(firstResult).toBeVisible({ timeout: 10_000 });
  if (basePath) {
    await expect(firstResult.locator('a')).toHaveAttribute('href', new RegExp(`^${basePath}/`));
  }
  await expect(page.locator('#search-status')).toContainText('找到');

  await input.fill('随机控制');
  await expect(page.locator('#search-results li').first()).toBeVisible({ timeout: 10_000 });

  await input.fill('Probability-Flow ODE');
  await expect(page.locator('#search-results li').first()).toBeVisible({ timeout: 10_000 });
  await expect(
    page.locator('#search-results li a[href*="/blog/diffusion/"]').first(),
  ).toBeVisible();
});
