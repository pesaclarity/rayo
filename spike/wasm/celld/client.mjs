// node client.mjs <models dir> : uploads weights, checks /ask outputs against the goldens, times it
import { readFileSync } from "node:fs";
const dir = process.argv[2], base = "http://127.0.0.1:9876";
const files = { "ask-reader": "ask_reader_decodes.json", reader: "reader_decodes.json", categoriser: "categoriser_decodes.json" };
const outs = { "ask-reader": ["slots", "intent"], reader: ["fields", "kind"], categoriser: ["category"] };
const argmax = (a) => a.indexOf(Math.max(...a));
const res = {};
for (const name of Object.keys(files)) {
  await fetch(`${base}/model/${name}`, { method: "PUT", body: readFileSync(`${dir}/${name}.weights`) });
  const gold = JSON.parse(readFileSync(`${dir}/${files[name]}`, "utf8"));
  let maxAbs = 0, miss = 0, bitExact = true, first = null;
  for (const [i, r] of gold.readings.entries()) {
    const body = name === "ask-reader" ? { text: r.question } : { input: r.input };
    const t = performance.now();
    const got = await (await fetch(`${base}/ask?model=${name}`, { method: "POST", body: JSON.stringify(body) })).json();
    if (i === 0) first = { ms: +(performance.now() - t).toFixed(1), cold: got.cold, kernels: got.kernels };
    for (const o of outs[name]) {
      const g = (Array.isArray(r[o][0]) ? r[o] : [r[o]]), w = g[0].length;
      g.forEach((row, t2) => { const seg = got[o].slice(t2 * w, (t2 + 1) * w);
        row.forEach((v, k) => { maxAbs = Math.max(maxAbs, Math.abs(v - seg[k])); if (Math.fround(v) !== seg[k]) bitExact = false; });
        if (argmax(row) !== argmax(seg)) miss++; });
    }
  }
  const inp = gold.readings.at(-1).input, N = 500;
  const call = async (n) => { const t = performance.now(); await (await fetch(`${base}/ask?model=${name}&n=${n}`, { method: "POST", body: JSON.stringify({ input: inp }) })).json(); return performance.now() - t; };
  for (let i = 0; i < 5; i++) await call(1);
  let one = Infinity; for (let i = 0; i < 20; i++) one = Math.min(one, await call(1));
  let many = Infinity; for (let i = 0; i < 3; i++) many = Math.min(many, await call(N));
  res[name] = { cases: gold.readings.length, maxAbs, labelMiss: miss, bitExact, firstRequest: first,
    requestMs: +one.toFixed(2), msPerRunInDO: +((many - one) / (N - 1)).toFixed(4) };
}
console.log(JSON.stringify(res, null, 1));
