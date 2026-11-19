#!/usr/bin/env python3
"""Checks the developer guide against the code, so it cannot drift:

- every public C function (RAYO_API in include/rayo/) has a heading in docs/API-C.md and
  its declaration there, as the header writes it;
- every name lib/rayo.dart exports has a heading in docs/API-DART.md;
- the C quickstart in the README and docs/GETTING-STARTED.md is examples/c/quickstart.c;
- every relative link in README.md and docs/ finds its file, and its #anchor a heading.

  python3 tool/check_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
errors: list[str] = []


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def c_functions() -> dict[str, str]:
    """Every RAYO_API declaration, by function name."""
    out = {}
    for h in sorted((ROOT / "include/rayo").glob("*.h")):
        text = re.sub(r"/\*.*?\*/", " ", h.read_text(), flags=re.S)
        for m in re.finditer(r"RAYO_API\s[^;]*\);", text):
            decl = norm(m.group(0))
            name = re.search(r"(\w+)\s*\(", decl).group(1)
            out[name] = decl
    return out


def code_blocks(md: str) -> list[str]:
    return re.findall(r"^```[^\n]*\n(.*?)^```", md, flags=re.S | re.M)


def check_c_api() -> None:
    doc = (ROOT / "docs/API-C.md").read_text()
    blocks = norm(" ".join(code_blocks(doc)))
    for name, decl in c_functions().items():
        if f"### `{name}`" not in doc:
            errors.append(f"docs/API-C.md: no heading for {name}")
        if decl not in blocks:
            errors.append(f"docs/API-C.md: {name}'s declaration is not the header's: {decl}")


def check_dart_api() -> None:
    lib = (ROOT / "lib/rayo.dart").read_text()
    doc = (ROOT / "docs/API-DART.md").read_text()
    for shown in re.findall(r"\bshow\s+([^;]+);", lib):
        for name in (n.strip() for n in shown.split(",")):
            if f"### `{name}`" not in doc:
                errors.append(f"docs/API-DART.md: no heading for {name}")


def check_quickstart() -> None:
    src = (ROOT / "examples/c/quickstart.c").read_text()
    body = src[src.index("#include <stdio.h>"):]
    for md in ("README.md", "docs/GETTING-STARTED.md"):
        if not any(b == body for b in code_blocks((ROOT / md).read_text())):
            errors.append(f"{md}: the C quickstart is not examples/c/quickstart.c")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).replace("`", "").replace("*", "")
    text = re.sub(r"[^\w\- ]", "", text.lower())
    return text.replace(" ", "-")


def anchors(md: str) -> set[str]:
    out, seen = set(), {}
    text = re.sub(r"^```.*?^```", "", md, flags=re.S | re.M)
    for h in re.findall(r"^#{1,6}\s+(.*?)\s*#*\s*$", text, flags=re.M):
        s = slug(h)
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.add(f"{s}-{n}" if n else s)
    return out


def check_links() -> None:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for f in files:
        md = f.read_text()
        prose = re.sub(r"^```.*?^```", "", md, flags=re.S | re.M)
        prose = re.sub(r"`[^`\n]*`", "", prose)
        for href in re.findall(r"\]\(([^)\s]+)\)", prose):
            if re.match(r"^(https?:|mailto:)", href):
                continue
            path, _, anchor = href.partition("#")
            target = (f.parent / path).resolve() if path else f
            where = f"{f.relative_to(ROOT)}: link {href}"
            if not target.exists():
                errors.append(f"{where}: no such file")
            elif anchor and target.suffix == ".md" and anchor not in anchors(target.read_text()):
                errors.append(f"{where}: no heading for #{anchor}")


check_c_api()
check_dart_api()
check_quickstart()
check_links()
for e in errors:
    print(e, file=sys.stderr)
if errors:
    sys.exit(f"check_docs: {len(errors)} problems")
print(f"check_docs: {len(c_functions())} C functions documented, links and quickstart ok")
