import { expect, test } from '@playwright/test';

test('desktop and responsive UI start entirely from packaged assets', async ({ page }, testInfo) => {
  const externalRequests = [];
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (!['127.0.0.1', 'localhost'].includes(url.hostname)) externalRequests.push(request.url());
  });
  const target = /mobile|tablet/.test(testInfo.project.name) ? '/mobile-preview' : '/';
  await page.goto(target);
  await expect(page).toHaveTitle(/stemsplat/i);
  await expect(page.locator('body')).toBeVisible();
  await expect(page.locator('#dropzone')).toBeVisible();
  await expect(page.locator('#app-version')).toContainText('v0.4.3');
  expect(externalRequests).toEqual([]);
});

test('corrupt legacy browser state cannot prevent startup', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('tasks', '{ definitely not json');
    localStorage.setItem('stemsplat.editorTrackDrafts', '{ broken');
  });
  await page.goto('/');
  await expect(page.locator('#dropzone')).toBeVisible();
  const legacyTasks = await page.evaluate(() => localStorage.getItem('tasks'));
  expect(legacyTasks).toBeNull();
});

test('keyboard opens settings and escape restores the application', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'desktop settings are host-only');
  await page.goto('/');
  await page.locator('#settings-btn').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#settings-overlay')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#settings-overlay')).toBeHidden();
  await expect(page.locator('#settings-btn')).toBeFocused();
});

test('custom settings listbox supports arrow keys and selection', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop', 'desktop settings are host-only');
  await page.goto('/');
  await page.locator('#settings-btn').click();
  const button = page.locator('#output-format-button');
  await button.focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.locator('#output-format-menu')).toHaveClass(/open/);
  await page.keyboard.press('Enter');
  await expect(page.locator('#output-format-label')).toContainText('320kb mp3');
  await expect.poll(async () => page.evaluate(async () => {
    const response = await fetch('/api/settings');
    return (await response.json()).output_format;
  })).toBe('mp3_320');
});
