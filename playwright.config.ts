import { defineConfig } from '@playwright/test';

const configuredBase = process.env.PUBLIC_BASE_PATH ?? '';
const basePath =
  !configuredBase || configuredBase === '/' ? '' : `/${configuredBase.replace(/^\/+|\/+$/g, '')}`;
const previewUrl = `http://127.0.0.1:4174${basePath}/`;

export default defineConfig({
  testDir: './tests/e2e',
  outputDir: './artifacts/playwright',
  fullyParallel: false,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: previewUrl,
    channel: 'chrome',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'npm run preview -- --host 127.0.0.1 --port 4174',
    url: previewUrl,
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
