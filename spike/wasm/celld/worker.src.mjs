// Spike Worker: PUT /model/<name> stores weights in R2; POST /ask {text|input} runs the Ask
// reader in a Durable Object (one per model) that keeps the model open.
// build.sh inlines rayo.mjs above this file (no_bundle mode does not resolve JS imports).
import rayoModule from "./rayo.wasm";

let rt; // one wasm instance per isolate
const runtime = () => (rt ??= instantiate(rayoModule));

export class Reader {
  constructor(state, env) { this.state = state; this.env = env; this.model = null; }
  async load(name) {
    if (this.model) return false;
    const obj = await this.env.MODELS.get(`${name}.weights`);
    if (!obj) throw new Error(`no ${name}.weights in R2`);
    this.model = runtime().open(new Uint8Array(await obj.arrayBuffer()));
    return true;
  }
  async fetch(request) {
    const url = new URL(request.url);
    const name = url.searchParams.get("model") || "ask-reader";
    const cold = await this.load(name);
    const body = await request.json();
    const spec = MODELS[name];
    const input = body.input ?? QUESTIONS[body.text];
    if (!input) return Response.json({ error: "no encoder in the spike: send input, or a known test text" }, { status: 400 });
    const n = Math.max(1, Math.min(10000, Number(url.searchParams.get("n") || 1)));
    let out;
    for (let i = 0; i < n; i++) out = runCase(this.model, spec, input);
    const r = { cold, kernels: this.model.kernels(), wasmMemoryMB: runtime().memoryBytes() / 1048576 };
    for (const [k, v] of Object.entries(out)) r[k] = Array.from(v);
    return Response.json(r);
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const m = url.pathname.match(/^\/model\/([a-z-]+)$/);
    if (m && request.method === "PUT") {
      await env.MODELS.put(`${m[1]}.weights`, request.body);
      return new Response(null, { status: 204 });
    }
    if (url.pathname === "/ask" && request.method === "POST") {
      const name = url.searchParams.get("model") || "ask-reader";
      const stub = env.READER.get(env.READER.idFromName(name));
      return stub.fetch(request);
    }
    return new Response("PUT /model/<name>, POST /ask {text|input}", { status: 404 });
  },
};
