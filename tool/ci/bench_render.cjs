// The benchmark table (table.html, from tool/ci/bench_results.py) as PNGs, light and
// dark, at twice the CSS size, with Playwright's Chromium.
//   node tool/ci/bench_render.cjs DIR     (writes DIR/table-light.png, DIR/table-dark.png)
// Playwright comes from NODE_PATH or a global install.
'use strict';
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

(async () => {
  const dir = path.resolve(process.argv[2] || '.');
  const url = pathToFileURL(path.join(dir, 'table.html')).href;
  const browser = await chromium.launch();
  try {
    for (const scheme of ['light', 'dark']) {
      const page = await browser.newPage({
        colorScheme: scheme,
        deviceScaleFactor: 2,
        viewport: { width: 980, height: 800 },
      });
      await page.goto(url);
      await page.evaluate(() => document.fonts.ready);
      const out = path.join(dir, `table-${scheme}.png`);
      await page.locator('#card').screenshot({ path: out });
      console.log(`wrote ${out}`);
      await page.close();
    }
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
