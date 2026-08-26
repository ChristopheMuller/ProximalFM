"""proximalfm.py — this project's own model, as an ordinary benchmark method.

ProximalFM used to have its own entry point: a `models:` block in the config, a
`run_neural_model` in `run.py`, its own device handling and its own prediction
cache. Nothing about it needs that. It is a `CATEEstimator` like every other
competitor — `fit` stores the context, `predict` scores the query rows — so it
lives here with them, is configured under `methods:` with `cls: ProximalFM`, and
takes its device and thread budget from the method's resolved `ComputeConfig`
like everything else.

Two things follow from being a normal method, and both are deliberate:

- **One episode per forward pass.** The old path batched three episodes into the
  model's own batch dimension. The per-episode interface gives that up; the
  model is invariant to the episode axis (query rows never attend to each other
  and every standardization statistic is context-only), so this changes wall
  time, not numbers. Fanning episodes out across CPU workers recovers more than
  the batching did; on a GPU, say `compute: {device: mps, execution: inline}`
  and the device is held by the one process — the arrangement `compute.py`
  already exists to express.
- **The checkpoint is loaded once per process**, not once per episode:
  `build_method` runs per episode, so the loaded model is cached module-level
  keyed by `(path, device)`, exactly as `CausalFM_Estimator` does.

`checkpoint:` takes a file or a directory (`best_model.ckpt`, else the highest
step). `all: true` scores *every* checkpoint in a directory; `normalize_cfg`
expands that into the config's ordinary `outputs:` list before the run starts,
so the runner's output names, the prediction cache and the skip-if-present check
work on it unchanged, and all of the checkpoints share one stored context.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from benchmarks.methods.base import CATEEstimator

__all__ = ["ProximalFM_Estimator", "normalize_cfg", "resolve_checkpoints"]

# (checkpoint path, device) -> loaded model. Keyed by device as well as path
# because the same weights on two devices are two distinct loaded models, and a
# shared session-wide pool can be asked for both.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _step_of(path: Path) -> int:
    """The training step a checkpoint file name claims, or -1 if it names none."""
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoints(cfg: dict) -> list[tuple[str, Path]]:
    """Which checkpoint(s) a config entry names, as `(output suffix, path)`.

    A file is itself. A directory is `best_model.ckpt` if present, else its
    highest-numbered step — unless `all: true`, which takes every `*.ckpt` in it,
    ordered by step, each getting its own output suffix.
    """
    path = Path(cfg["checkpoint"])
    if path.is_file():
        return [("", path)]

    if not path.is_dir():
        raise FileNotFoundError(
            f"method {cfg.get('name', 'ProximalFM')!r}: checkpoint {path} is "
            "neither a file nor a directory."
        )

    ckpts = sorted(((_step_of(f), f) for f in path.glob("*.ckpt")), key=lambda p: p[0])
    if not ckpts:
        raise FileNotFoundError(f"No *.ckpt found in {path}.")

    if cfg.get("all", False):
        return [
            (f"_step_{step}" if step >= 0 else f"_{f.stem}", f) for step, f in ckpts
        ]

    best = path / "best_model.ckpt"
    if best.exists():
        return [("", best)]
    step, f = ckpts[-1]
    return [(f"_{step}", f)]


def normalize_cfg(cfg: dict) -> dict:
    """Expand `all: true` into an explicit `outputs:` list of checkpoints.

    Done once, before the run touches the entry, because the output *names* are
    what the runner pre-allocates its prediction arrays with and what the
    skip-if-already-computed check globs for. After this, a multi-checkpoint
    entry is an ordinary multi-output method: one `fit` (which only stores the
    context), then one prediction per checkpoint.
    """
    if not cfg.get("all", False) or cfg.get("outputs"):
        return cfg
    name = cfg["name"]
    return {
        **cfg,
        "outputs": [
            {"name": f"{name}{suffix}", "checkpoint": str(path)}
            for suffix, path in resolve_checkpoints(cfg)
        ],
    }


def _load_model(path: Path, device: torch.device):
    """Return the model for this checkpoint on this device, loading it once."""
    key = (str(path), str(device))
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    from proximalfm._model.proximalfm import ProximalFM

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ProximalFM(**ckpt.get("config", {})).to(device)

    # Strip the wrappers a checkpoint may have been saved through
    # (`torch.compile`'s `_orig_mod.`, DDP's `module.`).
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(
        {
            k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in state_dict.items()
        },
        strict=False,
    )
    model.eval()

    _MODEL_CACHE[key] = model
    return model


class ProximalFM_Estimator(CATEEstimator):
    """ProximalFM behind the benchmark's `fit`/`predict` interface.

    `fit` does no work beyond moving the context onto the device — the model is
    in-context, so "fitting" *is* the context. `predict` concatenates the query
    rows onto it and runs one forward pass per checkpoint.

    Args:
        checkpoint: file or directory; see `resolve_checkpoints`.
        device: from the method's resolved `ComputeConfig`, never chosen here.
        ensemble: number of feature-transformed views to average at inference.
            `<= 1` is a single plain `predict`.
        transforms: bijective value transforms cycled across ensemble views
            (e.g. `[sign, signed_log, arcsinh, quantile]`); unset means
            permutation-only.
        swap_diag: diagnostic mode that exchanges the W and Z proxy blocks
            before fitting and prediction. This intentionally gives the model
            the proxies under the wrong structural role labels.
        outputs: the config's `outputs:` list; each entry may name its own
            `checkpoint:`, and all of them share this one stored context.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        ensemble: int = 1,
        transforms: Optional[list[str]] = None,
        outputs: Optional[list[dict]] = None,
        all: bool = False,
        swap_diag: bool = False,
    ):
        self.device = torch.device(device)
        self.n_ensemble = int(ensemble or 1)
        self.transforms = transforms or None
        self.swap_diag = bool(swap_diag)

        base_cfg = {"checkpoint": checkpoint, "all": all}
        if outputs:
            # One named prediction per output. An output without its own
            # `checkpoint:` falls back to the method's, resolved singly — an
            # output names exactly one checkpoint, so `all:` has no say here
            # (`normalize_cfg` has already spent it producing this list).
            self._targets = [
                (o["name"], resolve_checkpoints({**base_cfg, "all": False, **o})[0][1])
                for o in outputs
            ]
        else:
            self._targets = [(None, resolve_checkpoints(base_cfg)[0][1])]

        self._ctx: Optional[dict[str, torch.Tensor]] = None

    # -- interface -----------------------------------------------------------

    def _t(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=self.device)

    def _proxy_inputs(self, W, Z):
        """Return proxy blocks in the roles presented to the model.

        ``swap_diag`` is deliberately applied to both context and query
        inputs. The query proxies are hidden by ProximalFM, but swapping them
        there keeps the adapter's input protocol consistent and matters for
        the inferred ``d_W``/``d_Z`` values.
        """
        return (Z, W) if self.swap_diag else (W, Z)

    def fit(self, X, A, Y, *, W=None, Z=None):
        if W is None or Z is None:
            raise ValueError(
                "ProximalFM requires both W (outcome) and Z (treatment) proxies."
            )
        W, Z = self._proxy_inputs(W, Z)
        self._ctx = {
            "X": self._t(X),
            "W": self._t(W),
            "Z": self._t(Z),
            "A": self._t(np.asarray(A).reshape(-1)),
            "Y": self._t(np.asarray(Y).reshape(-1)),
        }
        return self

    def predict(self, X, *, W=None, Z=None):
        return next(iter(self.predict_variants(X, W=W, Z=Z).values()))

    def predict_variants(self, X, *, W=None, Z=None):
        if self._ctx is None:
            raise ValueError("Call fit() before predict().")
        if W is None or Z is None:
            raise ValueError("ProximalFM requires both W and Z at predict time.")

        ctx = self._ctx
        W, Z = self._proxy_inputs(W, Z)
        # The model reads one sequence of context-then-query rows and infers the
        # split from the length of `y_train`; only the context rows' A is used
        # (`ProximalFM.forward` truncates it), and the query rows' proxies are
        # masked inside the model (`_hide_query_proxies`), which is why this is a
        # `cate(x)` method despite being handed W and Z for every row.
        inputs = {
            "X": torch.cat([ctx["X"], self._t(X)]).unsqueeze(0),
            "W": torch.cat([ctx["W"], self._t(W)]).unsqueeze(0),
            "Z": torch.cat([ctx["Z"], self._t(Z)]).unsqueeze(0),
            "A": ctx["A"].unsqueeze(0),
            "y_train": ctx["Y"].unsqueeze(0),
        }
        dims = {
            f"d_{k}": torch.tensor([inputs[k].shape[-1]], device=self.device)
            for k in ("X", "W", "Z")
        }

        preds: dict[Optional[str], np.ndarray] = {}
        with torch.no_grad():
            for name, ckpt_path in self._targets:
                model = _load_model(ckpt_path, self.device)
                if self.n_ensemble > 1:
                    cate = model.predict_ensemble(
                        **inputs,
                        **dims,
                        target="cate",
                        output_type="mean",
                        n_estimators=self.n_ensemble,
                        value_transforms=self.transforms,
                    )
                else:
                    cate = model.predict(
                        **inputs, **dims, target="cate", output_type="mean"
                    )
                preds[name] = cate.squeeze(0).cpu().numpy()
        return preds
