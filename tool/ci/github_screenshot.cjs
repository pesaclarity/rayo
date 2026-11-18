// Screenshots of rayo's README as GitHub shows it, light and dark: the part from the
// "What rayo can do" heading to the end of the benchmarks, and the top of the page.
//   node tool/ci/github_screenshot.cjs OUT_DIR [URL]
// Waits for the benchmark image (from the bench branch) to load; screenshots anyway, with
// a warning, if it does not. Playwright comes from NODE_PATH or a global install.
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

// The first heading of the part shot, and the last section of it.
const FIRST = new RegExp(process.env.SHOT_FIRST || 'what rayo can do', 'i');
const LAST = new RegExp(process.env.SHOT_LAST || 'benchmarks', 'i');
// Through a proxy (HTTPS_PROXY) when trying it outside CI.
const launch = process.env.SHOT_PROXY
  ? { proxy: { server: process.env.SHOT_PROXY } }
  : {};

(async () => {
  const dir = path.resolve(process.argv[2] || '.');
  const url = process.argv[3] || 'https://github.com/pesaclarity/rayo';
  fs.mkdirSync(dir, { recursive: true });
  const browser = await chromium.launch(launch);
  let status = 0;
  try {
    for (const scheme of ['light', 'dark']) {
      const page = await browser.newPage({
        colorScheme: scheme,
        deviceScaleFactor: 2,
        viewport: { width: 1280, height: 900 },
        ignoreHTTPSErrors: Boolean(process.env.SHOT_PROXY),
      });
      // A fresh copy of the image, not one a cache kept.
      await page.route(/raw\.githubusercontent\.com|camo\.githubusercontent\.com/, (route) =>
        route.continue({ headers: { ...route.request().headers(), 'cache-control': 'no-cache' } }),
      );
      await page.goto(url, { waitUntil: 'load', timeout: 90000 });
      await page.waitForLoadState('networkidle', { timeout: 30000 }).catch(() => {});
      const readme = page.locator('article.markdown-body').first();
      await readme.waitFor({ timeout: 60000 });
      const img = readme.locator('img[alt^="rayo benchmarks"]').first();
      try {
        await img.scrollIntoViewIfNeeded({ timeout: 15000 });
        await page.waitForFunction(
          (el) => el && el.complete && el.naturalWidth > 0,
          await img.elementHandle({ timeout: 15000 }),
          { timeout: 60000 },
        );
      } catch (err) {
        console.warn(`::warning::the benchmark image did not load (${scheme}): ${err.message}`);
      }
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(1000);
      // The page's top, as a visitor first sees it.
      await page.screenshot({ path: path.join(dir, `github-top-${scheme}.png`) });
      // From the first heading to the heading after the benchmarks.
      const clip = await page.evaluate(
        ([first, last]) => {
          const article = document.querySelector('article.markdown-body');
          const hs = [...article.querySelectorAll('h2')];
          const i = hs.findIndex((h) => new RegExp(first, 'i').test(h.textContent));
          if (i < 0) return null;
          let j = hs.findIndex((h, k) => k > i && new RegExp(last, 'i').test(h.textContent));
          if (j < 0) j = i;
          const top = (hs[i].closest('.markdown-heading') || hs[i]).getBoundingClientRect();
          const end = hs[j + 1]
            ? (hs[j + 1].closest('.markdown-heading') || hs[j + 1]).getBoundingClientRect().top
            : article.getBoundingClientRect().bottom;
          const box = article.getBoundingClientRect();
          return {
            x: Math.max(0, box.left - 16) + window.scrollX,
            y: top.top - 16 + window.scrollY,
            width: box.width + 32,
            height: end - top.top + 8,
          };
        },
        [FIRST.source, LAST.source],
      );
      if (clip) {
        await page.screenshot({
          path: path.join(dir, `github-readme-${scheme}.png`),
          clip,
          fullPage: true,
        });
      } else {
        console.warn(`::warning::no "${FIRST.source}" heading in the README (${scheme})`);
        status = 1;
      }
      console.log(`${scheme}: done`);
      await page.close();
    }
  } finally {
    await browser.close();
  }
  process.exit(status);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
