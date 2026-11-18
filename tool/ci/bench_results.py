#!/usr/bin/env python3
"""rayo's benchmark results: one machine's raw timings to JSON, and a run's JSON files to
the files of the `bench` branch (the results, history.csv, README.md, table.html and
table.svg; tool/ci/bench_render.mjs makes the PNGs from table.html). Standard library only.

  bench_results.py collect RAW_DIR OUT.json           (tool/ci/bench_run.sh calls it)
  bench_results.py sizes OUT.json ABI=BYTES...        (tool/ci/android.sh calls it)
  bench_results.py publish --incoming DIR --out DIR --commit SHA [--date ISO] [--repo R]
                           [--run-url URL] [--keep N] [--history N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import platform
import re
import statistics
import sys
from pathlib import Path

SCHEMA = 1
SYNTHETIC_NOTE = ("SmolLM2-135M shape with synthetic (random) weights, written by "
                  "tool/make_synthetic_slm.py: every matrix product, the cache and the memory "
                  "are the real model's size, but the text it writes means nothing.")
VM_NOTE = ("GitHub-hosted runners are shared virtual machines: these numbers are indicative "
           "and vary from run to run. They are not a phone's.")
X86_FLAGS = ["sse4_1", "avx2", "fma", "avx_vnni", "avx512f", "avx512_vnni"]
ARM_FLAGS = ["asimd", "asimddp", "i8mm", "sve", "sve2", "bf16"]


# --- collect -----------------------------------------------------------------------------

def _floats(pattern: str, text: str) -> list[list[float]]:
    return [[float(x) for x in m] for m in re.findall(pattern, text)]


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 4) if xs else None


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _cpu_model(lscpu: str) -> str:
    for key in ("Model name", "model name"):
        m = re.search(rf"^{key}\s*:\s*(.+)$", lscpu, re.M)
        if m and m.group(1).strip() not in ("", "-"):
            return m.group(1).strip()
    return platform.processor() or "unknown"


def clean_cpu(name: str) -> str:
    """Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz -> Intel Xeon Platinum 8370C @ 2.80GHz."""
    name = re.sub(r"\((R|TM|tm|r)\)", "", name)
    name = re.sub(r"\b(CPU|Processor)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def collect(raw: Path, out: Path) -> dict:
    machine = dict(re.findall(r"^(\w+): (.*)$", _read(raw / "machine.txt"), re.M))
    arch = (machine.get("uname", "").split() or ["", platform.machine()])[-1]
    have = set(_read(raw / "cpuflags.txt").split(":", 1)[-1].split())
    wanted = X86_FLAGS if arch in ("x86_64", "amd64") else ARM_FLAGS
    reader, writer = _read(raw / "reader.txt"), _read(raw / "writer.txt")
    rq = _floats(r"per question median ([0-9.]+) ms, p90 ([0-9.]+) ms", reader)
    wr = _floats(r"per reply median ([0-9.]+) ms, p90 ([0-9.]+) ms; per step median ([0-9.]+)",
                 writer)
    kern = re.findall(r"^\S+ ([\w-]+): ", reader, re.M)
    doc: dict = {
        "kind": "rayo-bench-runner",
        "schema": SCHEMA,
        "runner": {
            "label": os.environ.get("BENCH_RUNNER", platform.node()),
            "arch": arch,
            "cpu": _cpu_model(_read(raw / "lscpu.txt")),
            "cores": int(machine.get("cores", "0") or 0),
            "cpu_flags": [f for f in wanted if f in have],
            "cc": machine.get("cc", ""),
            "cflags": machine.get("cflags", ""),
        },
        "kernels": kern[0] if kern else None,
        "ask": {
            "reader": {"per_question_median_ms": _median([r[0] for r in rq]),
                       "per_question_p90_ms": _median([r[1] for r in rq]), "runs": len(rq)},
            "writer": {"per_reply_median_ms": _median([r[0] for r in wr]),
                       "per_reply_p90_ms": _median([r[1] for r in wr]),
                       "per_step_median_ms": _median([r[2] for r in wr]), "runs": len(wr)},
        },
        "slm": [],
        "library_so_bytes": None,
    }
    for weights in ("q4", "int8"):
        rows = [dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                for line in _read(raw / f"slm-{weights}.txt").splitlines() if "threads=" in line]
        for threads in sorted({int(r["threads"]) for r in rows}):
            rs = [r for r in rows if int(r["threads"]) == threads]
            doc["slm"].append({
                "model": "SmolLM2-135M shape", "weights": weights, "synthetic": True,
                "threads": threads, "prompt_tokens": int(rs[0]["prompt"]),
                "decode_tokens": int(rs[0]["decode"]),
                "prompt_tok_s": _median([float(r["prompt_tok_s"]) for r in rs]),
                "decode_tok_s": _median([float(r["decode_tok_s"]) for r in rs]),
                "peak_rss_mb": _median([float(r["peak_rss_mb"]) for r in rs]),
                "file_mb": float(rs[0]["file_mb"]), "runs": len(rs)})
    so = _read(raw / "so_bytes.txt").strip()
    doc["library_so_bytes"] = int(so) if so.isdigit() else None
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=2))
    return doc


def sizes(out: Path, pairs: list[str]) -> None:
    doc = {"kind": "rayo-android-sizes", "schema": SCHEMA,
           "bytes": {k: int(v) for k, v in (p.split("=", 1) for p in pairs)}}
    out.write_text(json.dumps(doc, indent=2) + "\n")


# --- the table ---------------------------------------------------------------------------

def _ms(x: float | None) -> str:
    return "–" if x is None else (f"{x:.2f} ms" if x < 10 else f"{x:.1f} ms")


def _rate(x: float | None) -> str:
    return "–" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}")


def _slm(r: dict, weights: str, threads: int) -> dict | None:
    for s in r.get("slm", []):
        if s["weights"] == weights and s["threads"] == threads:
            return s
    return None


def _pair(r: dict, weights: str, key: str) -> str:
    a, b = _slm(r, weights, 1), _slm(r, weights, 2)
    if not a and not b:
        return "–"
    return f"{_rate(a and a[key])} / {_rate(b and b[key])}"


def _peak(r: dict, weights: str) -> str:
    xs = [s["peak_rss_mb"] for s in r.get("slm", []) if s["weights"] == weights
          and s.get("peak_rss_mb") is not None]
    return f"{max(xs):.0f} MB" if xs else "–"


def _kb(n: int | None) -> str:
    return "–" if n is None else f"{(n + 1023) // 1024} KB"


def runner_title(r: dict) -> tuple[str, str]:
    run = r["runner"]
    arch = {"aarch64": "arm64", "arm64": "arm64"}.get(run["arch"], run["arch"])
    return arch, f"{run['label']}, {run['cores']} vCPU" if run.get("cores") else run["label"]


def table(doc: dict) -> dict:
    """The table every format draws: column titles, then sections of (label, cells) rows,
    where cells is one string a runner, or one string that spans them."""
    rs = doc["runners"]
    sections = [
        ("Machine", [
            ("CPU", [clean_cpu(r["runner"]["cpu"]) for r in rs]),
            ("Kernels chosen at run time", [r.get("kernels") or "–" for r in rs]),
        ]),
        ("Pesa Clarity's Ask models · int8 · 1 thread · median", [
            ("Reader, per question", [_ms(r["ask"]["reader"]["per_question_median_ms"])
                                      for r in rs]),
            ("Writer, per reply (≈13 tokens)", [_ms(r["ask"]["writer"]["per_reply_median_ms"])
                                                for r in rs]),
        ]),
        ("SmolLM2-135M shape · synthetic weights† · 128-token prompt, 64 decoded", [
            ("q4 prompt tokens/s (1 / 2 threads)", [_pair(r, "q4", "prompt_tok_s") for r in rs]),
            ("q4 decode tokens/s (1 / 2 threads)", [_pair(r, "q4", "decode_tok_s") for r in rs]),
            ("q4 peak memory", [_peak(r, "q4") for r in rs]),
            ("int8 prompt tokens/s (1 / 2 threads)",
             [_pair(r, "int8", "prompt_tok_s") for r in rs]),
            ("int8 decode tokens/s (1 / 2 threads)",
             [_pair(r, "int8", "decode_tok_s") for r in rs]),
            ("int8 peak memory", [_peak(r, "int8") for r in rs]),
        ]),
    ]
    lib = doc.get("library_sizes") or {}
    if lib.get("source") == "android-ndk":
        b = lib["bytes"]
        abis = [a for a in ("arm64-v8a", "armeabi-v7a", "x86_64") if a in b]
        sections.append(("Library size · stripped · whole runtime with the language-model path", [
            ("Android library (.so)", " · ".join(f"{a} {_kb(b[a])}" for a in abis)),
        ]))
    else:
        sections.append(("Library size · stripped · whole runtime with the language-model path", [
            ("Linux .so", [_kb(r.get("library_so_bytes")) for r in rs]),
        ]))
    return {"columns": [runner_title(r) for r in rs], "sections": sections}


def markdown_table(doc: dict) -> str:
    t = table(doc)
    cols = t["columns"]
    lines = ["| | " + " | ".join(f"**{a}**<br>{b}" for a, b in cols) + " |",
             "| --- |" + " --- |" * len(cols)]
    for title, rows in t["sections"]:
        lines.append(f"| **{title}** |" + " |" * len(cols))
        for label, cells in rows:
            if isinstance(cells, str):
                cells = [cells] + [""] * (len(cols) - 1)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


CSS = """
:root { --bg: #ffffff; --card: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0;
        --band: #f6f8fa; --accent: #d9480f; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --card: #0d1117; --fg: #f0f6fc; --muted: #9198a1; --line: #3d444d;
          --band: #151b23; --accent: #ff8a4c; }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica,
       Arial, sans-serif; color: var(--fg); }
#card { width: 900px; padding: 28px 32px 22px; background: var(--card); }
header { display: flex; justify-content: space-between; align-items: flex-end;
         border-bottom: 2px solid var(--accent); padding-bottom: 12px; margin-bottom: 6px; }
.brand { font-size: 34px; font-weight: 800; letter-spacing: -0.5px; line-height: 1; }
.brand span { color: var(--accent); }
.brand small { display: block; font-size: 14px; font-weight: 500; color: var(--muted);
               letter-spacing: 0; margin-top: 6px; }
.meta { text-align: right; font-size: 13px; color: var(--muted); line-height: 1.55; }
.meta b { color: var(--fg); font-weight: 600; }
.meta code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
             color: var(--fg); }
table { width: 100%; border-collapse: collapse; font-size: 14px;
        font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; text-align: left; vertical-align: top; }
thead th { font-size: 16px; padding-top: 12px; }
thead th small { display: block; font-size: 12px; font-weight: 400; color: var(--muted); }
tr.section td { background: var(--band); color: var(--muted); font-size: 12px;
                font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
                padding-top: 6px; padding-bottom: 6px; }
tbody tr:not(.section) td { border-bottom: 1px solid var(--line); }
td.label { color: var(--muted); width: 38%; }
td.num { font-weight: 600; }
footer { margin-top: 12px; font-size: 12px; color: var(--muted); line-height: 1.5; }
"""


def html_page(doc: dict) -> str:
    t, e = table(doc), html.escape
    head = "".join(f"<th>{e(a)}<small>{e(b)}</small></th>" for a, b in t["columns"])
    body = []
    n = len(t["columns"])
    for title, rows in t["sections"]:
        body.append(f'<tr class="section"><td colspan="{n + 1}">{e(title)}</td></tr>')
        for label, cells in rows:
            if isinstance(cells, str):
                tds = f'<td class="num" colspan="{n}">{e(cells)}</td>'
            else:
                tds = "".join(f'<td class="num">{e(c)}</td>' for c in cells)
            body.append(f'<tr><td class="label">{e(label)}</td>{tds}</tr>')
    runs = max([s.get("runs", 0) for r in doc["runners"] for s in r.get("slm", [])] or [0])
    ask_runs = max([r["ask"]["reader"].get("runs", 0) for r in doc["runners"]] or [0])
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>rayo benchmarks</title>
<meta name="color-scheme" content="light dark"><style>{CSS}</style></head>
<body><div id="card">
<header>
  <div class="brand">rayo<span>.</span><small>On-device inference in plain C89 · live
  benchmarks</small></div>
  <div class="meta"><b>{e(doc["date"][:10])}</b> · commit <code>{e(doc["commit"][:7])}</code><br>
  GitHub Actions · ubuntu-24.04 runners</div>
</header>
<table><thead><tr><th></th>{head}</tr></thead><tbody>
{chr(10).join(body)}
</tbody></table>
<footer>† {e(SYNTHETIC_NOTE)} Medians of {ask_runs} runs (Ask) and {runs} (language model). {e(VM_NOTE)}</footer>
</div></body></html>
"""


def _text_w(s: str, size: float, bold: bool = False) -> float:
    return len(s) * size * (0.58 if bold else 0.54)


def svg_image(doc: dict) -> str:
    """The same table as an SVG, light or dark as the viewer's colour scheme is."""
    t, e = table(doc), html.escape
    n = len(t["columns"])
    W, pad, label_w = 900, 32, 340
    col_w = (W - 2 * pad - label_w) / max(n, 1)
    y, out = 40.0, []

    def text(x: float, yy: float, s: str, cls: str, anchor: str = "start") -> None:
        out.append(f'<text x="{x:.1f}" y="{yy:.1f}" class="{cls}" '
                   f'text-anchor="{anchor}">{e(s)}</text>')

    text(pad, y + 18, "rayo", "brand")
    out.append(f'<circle cx="{pad + _text_w("rayo", 34, True) + 5:.1f}" cy="{y + 15:.1f}" '
               'r="4" class="accent"/>')
    text(pad, y + 40, "On-device inference in plain C89 · live benchmarks", "muted")
    text(W - pad, y + 18, f'{doc["date"][:10]} · commit {doc["commit"][:7]}', "meta",
         "end")
    text(W - pad, y + 40, "GitHub Actions · ubuntu-24.04 runners", "muted", "end")
    y += 54
    out.append(f'<rect x="{pad}" y="{y:.1f}" width="{W - 2 * pad}" height="2" class="accent"/>')
    y += 30
    for i, (a, b) in enumerate(t["columns"]):
        x = pad + label_w + i * col_w + 10
        text(x, y, a, "th")
        text(x, y + 17, b, "small")
    y += 32
    for title, rows in t["sections"]:
        out.append(f'<rect x="{pad}" y="{y:.1f}" width="{W - 2 * pad}" height="26" '
                   'class="band"/>')
        text(pad + 10, y + 17, title.upper(), "section")
        y += 26
        for label, cells in rows:
            text(pad + 10, y + 21, label, "label")
            if isinstance(cells, str):
                cells = [cells]
            for i, c in enumerate(cells):
                text(pad + label_w + i * col_w + 10, y + 21, c, "num")
            y += 32
            out.append(f'<rect x="{pad}" y="{y - 1:.1f}" width="{W - 2 * pad}" height="1" '
                       'class="line"/>')
    y += 10
    note = f"† {SYNTHETIC_NOTE} {VM_NOTE}"
    words, line, lines = note.split(), "", []
    for w in words:
        if _text_w(line + " " + w, 12) > W - 2 * pad:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    lines.append(line)
    for ln in lines:
        y += 17
        text(pad, y, ln, "small")
    H = y + 24
    style = """
  .bg { fill: #ffffff; } text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
  "Noto Sans", Helvetica, Arial, sans-serif; fill: #1f2328; font-size: 14px; }
  .brand { font-size: 34px; font-weight: 800; } .accent { fill: #d9480f; }
  .muted, .small, .label, .section { fill: #59636e; } .muted { font-size: 14px; }
  .meta { font-size: 14px; font-weight: 600; } .th { font-size: 16px; font-weight: 700; }
  .small { font-size: 12px; } .section { font-size: 12px; font-weight: 600;
  letter-spacing: 0.4px; } .num { font-weight: 600; } .band { fill: #f6f8fa; }
  .line { fill: #d1d9e0; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; } text { fill: #f0f6fc; } .accent { fill: #ff8a4c; }
    .muted, .small, .label, .section { fill: #9198a1; } .band { fill: #151b23; }
    .line { fill: #3d444d; }
  }"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" '
            f'viewBox="0 0 {W} {H:.0f}" role="img" aria-label="rayo benchmarks">\n'
            f"<style>{style}\n</style>\n"
            f'<rect width="{W}" height="{H:.0f}" class="bg"/>\n' + "\n".join(out) + "\n</svg>\n")


# --- publish -----------------------------------------------------------------------------

HISTORY_FIELDS = ["date", "commit", "runner", "arch", "cpu", "kernels", "reader_ms",
                  "writer_ms", "q4_prompt_tok_s_1t", "q4_prompt_tok_s_2t", "q4_decode_tok_s_1t",
                  "q4_decode_tok_s_2t", "q4_peak_mb", "int8_prompt_tok_s_1t",
                  "int8_prompt_tok_s_2t", "int8_decode_tok_s_1t", "int8_decode_tok_s_2t",
                  "int8_peak_mb", "so_bytes"]


def history_rows(doc: dict) -> list[dict]:
    rows = []
    for r in doc["runners"]:
        row = {"date": doc["date"], "commit": doc["commit"], "runner": r["runner"]["label"],
               "arch": r["runner"]["arch"], "cpu": clean_cpu(r["runner"]["cpu"]),
               "kernels": r.get("kernels") or "",
               "reader_ms": r["ask"]["reader"]["per_question_median_ms"],
               "writer_ms": r["ask"]["writer"]["per_reply_median_ms"]}
        for w in ("q4", "int8"):
            for th in (1, 2):
                s = _slm(r, w, th) or {}
                row[f"{w}_prompt_tok_s_{th}t"] = s.get("prompt_tok_s")
                row[f"{w}_decode_tok_s_{th}t"] = s.get("decode_tok_s")
            peaks = [s["peak_rss_mb"] for s in r.get("slm", []) if s["weights"] == w]
            row[f"{w}_peak_mb"] = max(peaks) if peaks else None
        row["so_bytes"] = r.get("library_so_bytes")
        rows.append({k: ("" if v is None else v) for k, v in row.items()})
    return rows


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def readme(doc: dict, history: list[dict], repo: str, n_history: int) -> str:
    sha = doc["commit"]
    lines = [
        "# rayo benchmarks",
        "",
        f"Written by CI after every push to `main` ([`.github/workflows/ci.yml`]"
        f"(https://github.com/{repo}/blob/main/.github/workflows/ci.yml), `bench` and "
        f"`bench-publish`); nothing here is edited by hand. {VM_NOTE}",
        "",
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" srcset="table-dark.png">',
        '  <img alt="rayo benchmarks" src="table-light.png">',
        "</picture>",
        "",
        f"## Latest: {doc['date'][:10]}, [`{sha[:7]}`](https://github.com/{repo}/commit/{sha})",
        "",
        markdown_table(doc),
        "",
        f"† {SYNTHETIC_NOTE}",
        "",
    ]
    if doc.get("run_url"):
        lines += [f"The run: {doc['run_url']}", ""]
    lines += [f"## History (the last {n_history} runs; all in [history.csv](history.csv))", "",
              "| date | commit | runner | reader ms | writer ms | q4 decode tok/s 1 / 2 threads "
              "| int8 decode tok/s 1 / 2 threads |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    recent = sorted({(h["date"], h["commit"]) for h in history}, reverse=True)[:n_history]
    keep = {c for _, c in recent}
    for h in sorted(history, key=lambda h: (h["date"], h["runner"]), reverse=True):
        if h["commit"] not in keep:
            continue
        lines.append(
            f"| {h['date'][:10]} | [`{h['commit'][:7]}`](https://github.com/{repo}/commit/"
            f"{h['commit']}) | {h['runner']} | {_rate(_num(h['reader_ms']))} | "
            f"{_rate(_num(h['writer_ms']))} | {_rate(_num(h['q4_decode_tok_s_1t']))} / "
            f"{_rate(_num(h['q4_decode_tok_s_2t']))} | {_rate(_num(h['int8_decode_tok_s_1t']))}"
            f" / {_rate(_num(h['int8_decode_tok_s_2t']))} |")
    lines += ["", "Each run's full numbers (CPU flags, p90s, per-step times, file sizes) are in "
              "[results/](results/), one JSON file a run; the oldest are removed as new ones "
              "come.", ""]
    return "\n".join(lines)


def publish(a: argparse.Namespace) -> int:
    incoming, out = Path(a.incoming), Path(a.out)
    runners, android = [], None
    for f in sorted(incoming.rglob("*.json")) if incoming.exists() else []:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError) as err:
            print(f"skipping {f}: {err}", file=sys.stderr)
            continue
        if d.get("kind") == "rayo-bench-runner":
            runners.append(d)
        elif d.get("kind") == "rayo-android-sizes":
            android = d
    if not runners:
        print("no benchmark results to publish", file=sys.stderr)
        return 1
    runners.sort(key=lambda r: (r["runner"]["arch"] != "x86_64", r["runner"]["label"]))
    date = a.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = {
        "kind": "rayo-bench-run", "schema": SCHEMA, "commit": a.commit, "date": date,
        "run_url": a.run_url, "synthetic_note": SYNTHETIC_NOTE, "note": VM_NOTE,
        "runners": runners,
        "library_sizes": ({"source": "android-ndk", "bytes": android["bytes"]} if android else
                          {"source": "linux", "bytes": {r["runner"]["arch"]:
                                                        r.get("library_so_bytes")
                                                        for r in runners}}),
    }
    for r in runners:
        r.pop("kind", None)
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{date[:10]}-{a.commit[:7]}.json").write_text(json.dumps(doc, indent=2) + "\n")
    dated = []
    for f in results.glob("*.json"):
        try:
            dated.append((json.loads(f.read_text()).get("date", ""), f))
        except (OSError, ValueError):
            dated.append(("", f))
    for _, f in sorted(dated)[:-a.keep] if len(dated) > a.keep else []:
        f.unlink()
    hist_path = out / "history.csv"
    history = []
    if hist_path.exists():
        with hist_path.open(newline="") as fh:
            history = [h for h in csv.DictReader(fh) if h.get("commit") != a.commit]
    history += [{k: str(v) for k, v in h.items()} for h in history_rows(doc)]
    history.sort(key=lambda h: (h.get("date", ""), h.get("runner", "")))
    history = history[-a.history:]
    with hist_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)
    (out / "README.md").write_text(readme(doc, history, a.repo, 10))
    (out / "table.html").write_text(html_page(doc))
    (out / "table.svg").write_text(svg_image(doc))
    (out / "latest.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(markdown_table(doc))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("raw")
    c.add_argument("out")
    s = sub.add_parser("sizes")
    s.add_argument("out")
    s.add_argument("pairs", nargs="+")
    p = sub.add_parser("publish")
    p.add_argument("--incoming", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--date")
    p.add_argument("--repo", default="pesaclarity/rayo")
    p.add_argument("--run-url")
    p.add_argument("--keep", type=int, default=60, help="results files kept")
    p.add_argument("--history", type=int, default=600, help="history.csv rows kept")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(Path(a.raw), Path(a.out))
        return 0
    if a.cmd == "sizes":
        sizes(Path(a.out), a.pairs)
        return 0
    return publish(a)


if __name__ == "__main__":
    sys.exit(main())
