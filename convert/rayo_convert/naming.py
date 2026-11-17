"""Names for the converted file's weights, read from the source's tensor names.

TFLite keeps the Keras scopes in its tensor names: ``model_1/conv_1_1/convolution`` is the
kernel of layer ``conv_1``. Keras 3 adds ``_1`` to every scope it traces (the model's too);
those are taken off. The weights are named ``<layer>.<role>`` (``conv_1.kernel``,
``conv_1.bias``, ``token_embedding.table``, ``ln_0_a.gamma``), nested layers joined by dots
(``attention_0.out.kernel``), so the file reads like the model.
"""

from __future__ import annotations

import re
from collections import Counter

# Path components that are an op of TensorFlow's, not a layer.
_OPISH = re.compile(r"^(?:[A-Z]|convolution|strided_slice|moments|embedding_lookup|truediv|"
                    r"add|mul|sub|sequential$|arith\.)")


class Namer:
    def __init__(self, names: list[str]) -> None:
        firsts = Counter(n.split(";")[0].split("/")[0] for n in names if "/" in n)
        self.scope = firsts.most_common(1)[0][0] if firsts else ""
        self.keras3 = self.scope.endswith("_1")
        self.used: set[str] = set()

    def layer(self, name: str) -> str | None:
        """The layer a source tensor belongs to, or None."""
        first = name.split(";")[0]
        parts = first.split("/")
        if len(parts) < 2:
            return None
        if parts[0] == self.scope:
            parts = parts[1:]
        out = []
        for p in parts[:-1]:
            if self.keras3:
                if not p.endswith("_1"):
                    break
                out.append(p[:-2])
            else:
                if _OPISH.match(p):
                    break
                out.append(p)
        if not out:
            return None
        return ".".join(_clean(p) for p in out)

    def unique(self, base: str) -> str:
        name, n = base, 2
        while name in self.used:
            name = f"{base}_{n}"
            n += 1
        self.used.add(name)
        return name


def _clean(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)
