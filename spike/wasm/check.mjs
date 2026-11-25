// node spike/wasm/check.mjs <rayo.wasm> <models dir>  -> compares with the app's goldens, times runs
import { readFileSync } from "node:fs";
import { instantiate, MODELS, runCase } from "./rayo.mjs";
const [wasmPath, dir] = process.argv.slice(2);
const t0 = performance.now();
const mod = new WebAssembly.Module(readFileSync(wasmPath));
const rt = instantiate(mod);
const startMs = performance.now() - t0;
const files = { "ask-reader": "ask_reader_decodes.json", reader: "reader_decodes.json", categoriser: "categoriser_decodes.json" };
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
const report = { wasm: wasmPath, startMs: +startMs.toFixed(2), models: {} };
for (const [name, spec] of Object.entries(MODELS)) {
  const gold = JSON.parse(readFileSync(`${dir}/${files[name]}`, "utf8"));
  const m = rt.open(readFileSync(`${dir}/${name}.weights`));
  let maxAbs = 0, top1Miss = 0, tokMiss = 0, bitExact = true;
  for (const r of gold.readings) {
    const out = runCase(m, spec, r.input);
    for (const o of spec.outputs) {
      const g = r[o], got = out[o];
      const rows = Array.isArray(g[0]) ? g : [g];
      const w = rows[0].length;
      rows.forEach((row, t) => {
        const seg = got.subarray(t * w, (t + 1) * w);
        row.forEach((v, i) => { const d = Math.abs(v - seg[i]); if (d > maxAbs) maxAbs = d; if (Math.fround(v) !== seg[i]) bitExact = false; });
        if (argmax(row) !== argmax(seg)) (rows.length > 1 ? tokMiss++ : top1Miss++);
      });
    }
  }
  const N = 200, inp = gold.readings[gold.readings.length - 1].input;
  for (let i = 0; i < 10; i++) runCase(m, spec, inp);
  const s = performance.now(); for (let i = 0; i < N; i++) runCase(m, spec, inp);
  const perMs = (performance.now() - s) / N;
  report.models[name] = { cases: gold.readings.length, maxAbs, top1Miss, tokMiss, bitExact, msPerRun: +perMs.toFixed(4), kernels: m.kernels() };
  m.close();
}
report.memoryMB = +(rt.memoryBytes() / 1048576).toFixed(1);
console.log(JSON.stringify(report, null, 1));
