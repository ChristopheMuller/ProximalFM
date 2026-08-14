"""estimands.py — what a prediction answers, spelled out as a tag.

A method output is labelled by the quantity it estimates, in full:

    cate(x)        per-row effect conditioning on X alone — the primary target
    cate(x,w)      ... and on the query row's own W: a strictly finer conditional
    cate(x,w,z)
    cate(x,u)      an oracle handed the true confounder
    ate(x)         one scalar per episode

The *kind* says whether the output varies per row (``cate``) or is a constant
(``ate``). The parenthesis lists the variables the prediction reads at
**predict** time — using a variable only to fit a nuisance does not count, which
is why ``PLearner`` is ``cate(x)`` despite being fitted on both proxies.
That is exactly the property :mod:`benchmarks.metrics` routes its tables on.

This lives in its own module because two packages need it and neither may import
the other: ``benchmarks.methods`` derives a tag from a config
(``method_estimand``), and ``benchmarks.metrics`` decides which table a tag lands
in — and metrics must stay free of the modelling stack (``methods`` drags in
torch).
"""

from __future__ import annotations

ESTIMAND_KINDS = ("cate", "ate")
# Canonical order, so the same conditioning set always prints the same string.
ESTIMAND_INPUTS = ("x", "w", "z", "u")

# The pre-spelled-out tags. Old configs (`estimand: x|xw|ate`), old results.csv
# files and old preds caches keep meaning exactly what they did.
LEGACY_ESTIMANDS = {"x": "cate(x)", "xw": "cate(x,w)", "ate": "ate(x)"}


def format_estimand(kind: str, inputs) -> str:
    """``("cate", ["w", "x"]) -> "cate(x,w)"`` — the canonical spelling."""
    if kind not in ESTIMAND_KINDS:
        raise ValueError(f"estimand kind must be one of {ESTIMAND_KINDS}, got {kind!r}")
    seen = set(inputs)
    unknown = seen - set(ESTIMAND_INPUTS)
    if unknown:
        raise ValueError(
            f"unknown estimand input(s) {sorted(unknown)}; "
            f"expected a subset of {ESTIMAND_INPUTS}"
        )
    return f"{kind}({','.join(i for i in ESTIMAND_INPUTS if i in seen)})"


def parse_estimand(tag: str) -> tuple[str, tuple[str, ...]]:
    """``"cate(x,w)" -> ("cate", ("x", "w"))``, inputs in canonical order.

    Accepts the legacy tags too, so a config or a results.csv written before the
    inputs were spelled out still parses.
    """
    tag = str(tag).strip()
    tag = LEGACY_ESTIMANDS.get(tag, tag)
    if not tag.endswith(")") or "(" not in tag:
        raise ValueError(
            f"estimand must look like 'cate(x,w)' or be one of "
            f"{sorted(LEGACY_ESTIMANDS)}, got {tag!r}"
        )
    kind, _, rest = tag[:-1].partition("(")
    inputs = {p.strip() for p in rest.split(",") if p.strip()}
    format_estimand(kind, inputs)  # validates both halves; raises on garbage
    return kind, tuple(i for i in ESTIMAND_INPUTS if i in inputs)


def canonical_estimand(tag: str) -> str:
    """Validate a tag and return its canonical spelling (legacy tags included)."""
    return format_estimand(*parse_estimand(tag))
