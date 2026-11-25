// The Ask reader (tests/ask) run by rayo as WebAssembly in Node, checked
// against TFLite's recorded outputs as tests/test_ask.c checks the native
// build: every label TFLite's, every probability within 1e-4. Given more
// than one module, their outputs must also be the same floats, bit for bit.
//
//   node examples/wasm/ask_reader.mjs wasm/out/rayo.wasm [wasm/out/rayo-simd.wasm]
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { loadRayo } from "../../wasm/rayo.mjs";

const dir = fileURLToPath(new URL("../../tests/ask/", import.meta.url));
const gold = JSON.parse(readFileSync(dir + "ask_reader_decodes.json", "utf8")).readings;
const weights = readFileSync(dir + "ask-reader.weights");
const top = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

let failed = false;
let first = null;
for (const path of process.argv.slice(2)) {
  const rayo = await loadRayo(readFileSync(path));
  const model = rayo.open(weights);
  let labels = 0;
  let worst = 0;
  const all = [];
  for (const r of gold) {
    const ids = Int32Array.from(r.input);
    const [slots, intent] = model.run([ids.subarray(0, 96), ids.subarray(96)]);
    all.push(slots, intent);
    let ok = top(intent.subarray(0, r.intent.length)) === top(r.intent);
    r.intent.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - intent[i]))));
    r.slots.forEach((row, t) => {
      const got = slots.subarray(t * row.length, (t + 1) * row.length);
      ok &&= top(got) === top(row);
      row.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - got[i]))));
    });
    labels += ok;
  }
  const ms = (() => {
    const ids = Int32Array.from(gold[0].input);
    const ins = [ids.subarray(0, 96), ids.subarray(96)];
    const n = 100;
    const t = performance.now();
    for (let i = 0; i < n; i++) model.run(ins);
    return (performance.now() - t) / n;
  })();
  let same = "";
  if (first) {
    const bits = all.every((a, i) =>
      Buffer.from(a.buffer).equals(Buffer.from(first[i].buffer)));
    same = bits ? ", the same floats as the first module" : ", NOT the first module's floats";
    failed ||= !bits;
  } else first = all;
  console.log(`${path} (${model.kernels()}): labels ${labels} of ${gold.length} as TFLite, ` +
    `probabilities within ${worst.toPrecision(2)}${same}, ${ms.toFixed(2)} ms a run`);
  failed ||= labels !== gold.length || worst > 1e-4;
  model.close();
}
process.exit(failed ? 1 : 0);
