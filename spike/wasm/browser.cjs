// node browser.cjs <models dir>: runs check-style comparison in headless Chromium (Playwright)
const { chromium } = require("playwright");
const fs = require("fs"), path = require("path"), http = require("http");
const dir = process.argv[2], here = __dirname;
const files = { "/": path.join(here, "rayo.mjs"), "/rayo.mjs": path.join(here, "rayo.mjs"), "/rayo.wasm": path.join(here, "out/rayo.wasm"), "/rayo-simd.wasm": path.join(here, "out/rayo-simd.wasm") };
for (const f of fs.readdirSync(dir)) files["/m/" + f] = path.join(dir, f);
const srv = http.createServer((q, s) => { const f = files[q.url]; if (!f) { s.writeHead(404); return s.end(); }
  s.writeHead(200, { "content-type": q.url === "/" ? "text/plain" : q.url.endsWith(".mjs") ? "text/javascript" : q.url.endsWith(".wasm") ? "application/wasm" : "application/octet-stream" });
  s.end(fs.readFileSync(f)); }).listen(0, "127.0.0.1", async () => {
  const base = `http://127.0.0.1:${srv.address().port}`;
  const browser = await chromium.launch(); const page = await browser.newPage();
  await page.goto(base + "/");
  const res = await page.evaluate(async (base) => {
    const { instantiate, MODELS, runCase } = await import(base + "/rayo.mjs");
    const gf = { "ask-reader": "ask_reader_decodes.json", reader: "reader_decodes.json", categoriser: "categoriser_decodes.json" };
    const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
    const out = {};
    for (const w of ["rayo.wasm", "rayo-simd.wasm"]) {
      const t0 = performance.now();
      const mod = await WebAssembly.compileStreaming(fetch(`${base}/${w}`));
      const rt = instantiate(mod); const startMs = performance.now() - t0;
      const r = { startMs: +startMs.toFixed(1) };
      for (const [name, spec] of Object.entries(MODELS)) {
        const gold = await (await fetch(`${base}/m/${gf[name]}`)).json();
        const m = rt.open(new Uint8Array(await (await fetch(`${base}/m/${name}.weights`)).arrayBuffer()));
        let maxAbs = 0, miss = 0, bitExact = true;
        for (const g of gold.readings) { const o = runCase(m, spec, g.input);
          for (const k of spec.outputs) { const rows = Array.isArray(g[k][0]) ? g[k] : [g[k]], wd = rows[0].length;
            rows.forEach((row, t) => { const seg = o[k].subarray(t * wd, (t + 1) * wd);
              row.forEach((v, i) => { maxAbs = Math.max(maxAbs, Math.abs(v - seg[i])); if (Math.fround(v) !== seg[i]) bitExact = false; });
              if (argmax(row) !== argmax(seg)) miss++; }); } }
        const inp = gold.readings.at(-1).input; for (let i = 0; i < 10; i++) runCase(m, spec, inp);
        const N = 200, s = performance.now(); for (let i = 0; i < N; i++) runCase(m, spec, inp);
        r[name] = { maxAbs, miss, bitExact, msPerRun: +((performance.now() - s) / N).toFixed(4), kernels: m.kernels() };
        m.close();
      }
      r.memoryMB = rt.memoryBytes() / 1048576; out[w] = r;
    }
    return out;
  }, base);
  console.log(JSON.stringify({ browser: browser.version(), ...res }, null, 1));
  await browser.close(); srv.close();
});
