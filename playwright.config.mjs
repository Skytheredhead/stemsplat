import { defineConfig, devices } from '@playwright/test';

const e2eHome = process.env.STEMSPLAT_E2E_HOME;
if (!e2eHome) throw new Error('run Playwright through npm run test:e2e so state is isolated');

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: 'http://127.0.0.1:9876',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'mobile-320', use: { ...devices['iPhone 13 Mini'], browserName: 'chromium', viewport: { width: 320, height: 720 } } },
    { name: 'mobile-375', use: { ...devices['iPhone 13'], browserName: 'chromium', viewport: { width: 375, height: 812 } } },
    { name: 'tablet-768', use: { ...devices['iPad Mini'], browserName: 'chromium', viewport: { width: 768, height: 1024 } } },
  ],
  webServer: {
    command: 'STEMSPLAT_DISABLE_BACKGROUND_THREADS=1 python3 launcher.py --no-browser --host 127.0.0.1 --port 9876',
    env: { ...process.env, STEMSPLAT_HOME: e2eHome },
    url: 'http://127.0.0.1:9876/api/runtime_status',
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
