// Spike host for rayo.wasm: no imports, works in Node, Deno, celld and browsers.
// instantiate(module) -> { open(bytes) -> model }, model.run(binds, inputs:Int32Array[]) -> Float32Array[]
export function instantiate(module) {
  const inst = new WebAssembly.Instance(module, {});
  const x = inst.exports;
  x._initialize();
  const mem = () => x.memory.buffer;
  const cstr = (p) => { const b = new Uint8Array(mem(), p); let n = 0; while (b[n]) n++; return new TextDecoder().decode(b.subarray(0, n)); };
  const tmp = x.rw_alloc(64);
  const put = (s) => { const b = new TextEncoder().encode(s + "\0"); const p = x.rw_alloc(b.length); new Uint8Array(mem(), p, b.length).set(b); return p; };
  return {
    exports: x,
    memoryBytes: () => mem().byteLength,
    open(bytes) {
      const u = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      const p = x.rw_alloc(u.length);
      new Uint8Array(mem(), p, u.length).set(u);
      const h = x.rw_open(p, u.length);
      if (!h) throw new Error("rw_open: " + cstr(x.rw_error()));
      const bound = new Map();
      return {
        kernels: () => cstr(x.rw_kernels(h)),
        run(binds, inputs, outputs) {
          for (const [k, v] of Object.entries(binds || {})) {
            if (bound.get(k) === v) continue;
            const s = put(k); const r = x.rw_bind(h, s, v); x.rw_free(s);
            if (r) throw new Error("bind: " + cstr(x.rw_error()));
            bound.set(k, v);
          }
          inputs.forEach((arr, i) => {
            const p = x.rw_input(h, i, tmp); const n = new Uint32Array(mem(), tmp, 1)[0];
            if (!p || n !== arr.byteLength) throw new Error(`input ${i}: ${n} bytes, got ${arr.byteLength}`);
            new Uint8Array(mem(), p, n).set(new Uint8Array(arr.buffer, arr.byteOffset, n));
          });
          if (x.rw_run(h)) throw new Error("run: " + cstr(x.rw_error()));
          const out = [];
          for (let i = 0; i < outputs; i++) {
            const p = x.rw_output(h, i, tmp); const n = new Uint32Array(mem(), tmp, 1)[0];
            out.push(new Float32Array(mem().slice(p, p + n)));
          }
          return out;
        },
        close() { x.rw_close(h); },
      };
    },
  };
}

// The three app models: how inputs split and outputs come back (T = 96).
export const MODELS = {
  "ask-reader": { binds: { T: 96 }, split: 96, outputs: ["slots", "intent"] },
  "reader": { binds: { T: 96 }, split: 96, outputs: ["fields", "kind"] },
  "categoriser": { binds: {}, split: 0, outputs: ["category"] },
};
export function runCase(model, spec, input) {
  const ids = Int32Array.from(input);
  const ins = spec.split ? [ids.subarray(0, spec.split), ids.subarray(spec.split)] : [ids];
  const outs = model.run(spec.binds, ins, spec.outputs.length);
  const r = {}; spec.outputs.forEach((n, i) => (r[n] = outs[i])); return r;
}
