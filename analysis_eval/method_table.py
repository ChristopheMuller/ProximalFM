"""Build a concise paper-ready table of the methods enabled by a benchmark config.

The benchmark YAML is the source of truth for what was tested and for all
configuration-dependent properties. ``methods.json`` contains only stable,
curated literature metadata. The paper table is deliberately restricted to
outputs whose canonical estimand is exactly ``cate(x)``. A multi-output bridge
method therefore contributes only its CATE(X) variants.

Example::

    uv run python -m analysis_eval.method_table \
        --config benchmarks/config_methods.yaml --print
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.estimands import parse_estimand
from benchmarks.run import load_config, method_output_estimands

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "benchmarks" / "config_methods.yaml"
DEFAULT_METADATA = Path(__file__).with_name("methods.json")
DEFAULT_OUT = Path(__file__).with_name("results") / "methods.tex"
LABEL = "tab:benchmark_methods"
CAPTION = (
    "Methods estimating $\\operatorname{CATE}(X)$ in the benchmark. "
    "The method name is the configured benchmark output; the variant records "
    "the implementation or prediction-time choice. Detailed settings are "
    "reported in Appendix~\\ref{app:results}."
)


@dataclass(frozen=True)
class Citation:
    text: str
    bibtex_key: str | None = None


@dataclass(frozen=True)
class MethodRow:
    method: str
    method_label: str
    configured_method: str
    cls: str
    family: str
    estimator: str
    fit_inputs: str
    estimand: str
    predict_inputs: str
    proximal: bool
    foundation_model: bool
    tuned: bool
    oracle: bool
    citations: tuple[Citation, ...]


def load_metadata(path: Path) -> dict[str, Any]:
    """Load and minimally validate the curated method registry."""
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version 1")
    if not isinstance(metadata.get("classes"), dict):
        raise ValueError(f"{path}: 'classes' must be an object")
    if not isinstance(metadata.get("bases"), dict):
        raise ValueError(f"{path}: 'bases' must be an object")
    return metadata


def _citation(raw: dict[str, Any] | None) -> Citation | None:
    if raw is None:
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("citation.text must be a non-empty string")
    key = raw.get("bibtex_key")
    if key is not None and (not isinstance(key, str) or not key.strip()):
        raise ValueError("citation.bibtex_key must be null or a non-empty string")
    return Citation(text.strip(), key.strip() if key else None)


def _selected_proxy_inputs(cfg: dict[str, Any], *, defaults: bool) -> list[str]:
    inputs = []
    if cfg.get("use_W", defaults):
        inputs.append("W")
    if cfg.get("use_Z", defaults):
        inputs.append("Z")
    return inputs


def _fit_inputs(cfg: dict[str, Any]) -> str:
    """Variables actually available to the estimator during fitting."""
    cls = cfg["cls"]
    inputs = ["X", "A", "Y"]
    if cls in {"P2SLS", "PLearner", "KernelPV", "PMMR", "NMMR", "ProximalFM"}:
        inputs += ["W", "Z"]
    elif cls in {
        "S_Learner",
        "T_Learner",
        "X_Learner",
        "CausalFM",
        "DoPFN",
        "CausalPFN",
    }:
        inputs += _selected_proxy_inputs(cfg, defaults=True)
    elif cls == "CausalForest":
        inputs += _selected_proxy_inputs(cfg, defaults=False)
    elif cls == "ProxyMarginalizer":
        nested = cfg.get("estimator") or {}
        inputs += _selected_proxy_inputs(nested, defaults=True)
    if cfg.get("is_oracle", False):
        inputs.append("U")
    return ", ".join(dict.fromkeys(inputs))


def _is_tuned(cfg: dict[str, Any]) -> bool:
    tune = cfg.get("tune") or {}
    if bool(tune.get("enabled", False)) or bool(cfg.get("base_cv")):
        return True
    bridge_params = cfg.get("bridge_params") or {}
    return bool(bridge_params.get("lam_grid"))


def _estimator_label(
    cfg: dict[str, Any], output: dict[str, Any] | None, metadata: dict[str, Any]
) -> tuple[str, Citation | None]:
    cls = cfg["cls"]
    bases = metadata["bases"]

    if cls in {"S_Learner", "T_Learner", "X_Learner"}:
        base = cfg.get("base", "linear")
        if base not in bases:
            raise KeyError(f"no metadata for base estimator {base!r}")
        return bases[base]["label"], _citation(bases[base].get("citation"))
    if cls == "PLearner":
        base = (output or {}).get("base", cfg.get("base", "linear"))
        if (output or {}).get("kind") == "ate":
            return f"{cfg.get('bridge', 'kernel')} bridge (ATE)", None
        if base not in bases:
            raise KeyError(f"no metadata for base estimator {base!r}")
        return (
            f"{cfg.get('bridge', 'kernel')} bridge + {bases[base]['label']}",
            _citation(bases[base].get("citation")),
        )
    if cls == "ProxyMarginalizer":
        nested = cfg.get("estimator") or {}
        nested_cls = nested.get("cls", "unknown")
        nested_meta = metadata["classes"].get(nested_cls)
        nested_label = nested_meta["label"] if nested_meta else nested_cls
        base = nested.get("base")
        base_label = bases.get(base, {}).get("label") if base else None
        integration = (output or {}).get("integration", cfg.get("integration") or {})
        rule = integration.get("method", "empirical")
        points = integration.get("n_points")
        suffix = f"{rule}-{points}" if points is not None else str(rule)
        label = " / ".join(x for x in (nested_label, base_label, suffix) if x)
        base_cite = bases.get(base, {}).get("citation") if base else None
        return label, _citation(base_cite)

    fixed = {
        "P2SLS": "two-stage linear bridge",
        "KernelPV": "kernel two-stage bridge",
        "PMMR": "kernel MMR bridge",
        "NMMR": "neural MMR bridge",
        "CausalForest": "DROrthoForest",
        "CausalFM": "pre-trained transformer",
        "DoPFN": "pre-trained transformer",
        "CausalPFN": "pre-trained transformer",
        "ProximalFM": "pre-trained transformer",
    }
    return fixed.get(cls, metadata["classes"][cls]["label"]), None


def _deduplicate_citations(
    citations: Iterable[Citation | None],
) -> tuple[Citation, ...]:
    seen: set[tuple[str, str | None]] = set()
    result = []
    for citation in citations:
        if citation is None:
            continue
        identity = (citation.text, citation.bibtex_key)
        if identity not in seen:
            seen.add(identity)
            result.append(citation)
    return tuple(result)


def build_rows(config: dict[str, Any], metadata: dict[str, Any]) -> list[MethodRow]:
    """Resolve configured methods into rows for outputs estimating CATE(X)."""
    rows: list[MethodRow] = []
    classes = metadata["classes"]
    for cfg in config.get("methods") or []:
        cls = cfg.get("cls")
        if cls not in classes:
            raise KeyError(
                f"no analysis_eval metadata for benchmark class {cls!r}; "
                "add it to analysis_eval/methods.json"
            )
        meta = classes[cls]
        required = {"label", "family", "proximal", "foundation_model"}
        missing = required - set(meta)
        if missing:
            raise ValueError(f"metadata for {cls!r} is missing {sorted(missing)}")

        estimands = method_output_estimands(cfg)
        outputs = cfg.get("outputs") or [None]
        for output in outputs:
            name = cfg["name"] if output is None else output["name"]
            # Exact, rather than "any CATE": cate(x,w), cate(x,u), and ATE
            # answer different questions and do not belong in this table.
            if estimands[name] != "cate(x)":
                continue
            estimator, base_citation = _estimator_label(cfg, output, metadata)
            kind, prediction_inputs = parse_estimand(estimands[name])
            rows.append(
                MethodRow(
                    method=name,
                    method_label=meta["label"],
                    configured_method=cfg["name"],
                    cls=cls,
                    family=meta["family"],
                    estimator=estimator,
                    fit_inputs=_fit_inputs(cfg),
                    estimand=estimands[name],
                    predict_inputs=", ".join(v.upper() for v in prediction_inputs),
                    proximal=bool(meta["proximal"]),
                    foundation_model=bool(meta["foundation_model"]),
                    tuned=_is_tuned(cfg),
                    oracle=bool(cfg.get("is_oracle", False)),
                    citations=_deduplicate_citations(
                        (_citation(meta.get("citation")), base_citation)
                    ),
                )
            )
    return rows


_TEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _tex(text: Any) -> str:
    return "".join(_TEX_SPECIALS.get(char, char) for char in str(text))


def _latex_citations(citations: tuple[Citation, ...]) -> str:
    keys = [c.bibtex_key for c in citations if c.bibtex_key]
    text_only = [_tex(c.text) for c in citations if not c.bibtex_key]
    parts = ([f"\\citep{{{','.join(keys)}}}"] if keys else []) + text_only
    return "; ".join(parts) if parts else "--"


def render_latex(
    rows: Sequence[MethodRow], *, caption: str = CAPTION, label: str = LABEL
) -> str:
    lines = [
        "% Generated by `python -m analysis_eval.method_table` -- do not edit by hand.",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\footnotesize",
        "    \\setlength{\\tabcolsep}{2.5pt}",
        "    \\renewcommand{\\arraystretch}{0.96}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "    \\begin{tabularx}{\\linewidth}{@{}p{0.23\\linewidth} p{0.16\\linewidth} X p{0.22\\linewidth}@{}}",
        "        \\toprule",
        "        Method name & Method label & Variant & Citation \\\\",
        "        \\midrule",
    ]
    for row in rows:
        lines.append(
            "        "
            + " & ".join(
                [
                    _tex(row.method),
                    _tex(row.method_label),
                    _tex(row.estimator),
                    _latex_citations(row.citations),
                ]
            )
            + " \\\\"
        )
    lines += [
        "        \\bottomrule",
        "    \\end{tabularx}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_csv(rows: Sequence[MethodRow], path: Path) -> None:
    fields = [
        "method",
        "method_label",
        "configured_method",
        "class",
        "family",
        "estimator",
        "fit_inputs",
        "estimand",
        "predict_inputs",
        "proximal",
        "foundation_model",
        "tuned",
        "oracle",
        "citation",
        "citation_keys",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "method_label": row.method_label,
                    "configured_method": row.configured_method,
                    "class": row.cls,
                    "family": row.family,
                    "estimator": row.estimator,
                    "fit_inputs": row.fit_inputs,
                    "estimand": row.estimand,
                    "predict_inputs": row.predict_inputs,
                    "proximal": row.proximal,
                    "foundation_model": row.foundation_model,
                    "tuned": row.tuned,
                    "oracle": row.oracle,
                    "citation": "; ".join(c.text for c in row.citations),
                    "citation_keys": "; ".join(
                        c.bibtex_key for c in row.citations if c.bibtex_key
                    ),
                }
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Benchmark YAML to describe (default: benchmarks/config_methods.yaml).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Curated method/citation JSON (default: analysis_eval/methods.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .tex path; a sibling .csv is written next to it.",
    )
    parser.add_argument("--label", default=LABEL, help="LaTeX table label.")
    parser.add_argument("--caption", default=None, help="Override the table caption.")
    parser.add_argument(
        "--print", dest="to_stdout", action="store_true", help="Also print LaTeX."
    )
    args = parser.parse_args(argv)

    config = load_config(str(args.config))
    metadata = load_metadata(args.metadata)
    rows = build_rows(config, metadata)
    latex = render_latex(rows, caption=args.caption or CAPTION, label=args.label)

    if args.out.suffix.lower() != ".tex":
        parser.error("--out must be a .tex path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex, encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    write_csv(rows, csv_path)
    print(f"wrote {args.out}")
    print(f"wrote {csv_path}")
    if args.to_stdout:
        print(latex, end="")


if __name__ == "__main__":
    main()
