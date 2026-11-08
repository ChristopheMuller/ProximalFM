"""CATE point and continuous-histogram uncertainty metrics, in outcome units."""
from __future__ import annotations

import numpy as np


def point_metrics(prediction, truth) -> dict[str, float]:
    pred, target = np.asarray(prediction, float).ravel(), np.asarray(truth, float).ravel()
    if pred.shape != target.shape or not pred.size:
        raise ValueError("Prediction and truth must have equal nonempty shapes")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Point metrics require finite predictions and truth")
    residual = pred - target
    mse, bias, target_var = float(np.mean(residual**2)), float(residual.mean()), float(target.var())
    # Relative variance tolerance avoids reporting numerical noise as heterogeneity.
    nonconstant = target_var > 1e-12 * max(1.0, float(np.mean(target**2)))
    result = {"mse": mse, "pehe": np.sqrt(mse), "ate_bias": bias,
              "ate_absolute_error": abs(bias), "true_ate": float(target.mean()),
              "predicted_ate": float(pred.mean()), "true_cate_variance": target_var,
              "predicted_cate_variance": float(pred.var()), "n_query": int(pred.size)}
    result.update({"npehe": np.sqrt(mse / target_var) if nonconstant else np.nan,
                   "shape_r2": 1.0 - float(residual.var()) / target_var if nonconstant else np.nan,
                   "ate_bias_norm": bias / np.sqrt(target_var) if nonconstant else np.nan,
                   "level_error": bias**2 / target_var if nonconstant else np.nan,
                   "shape_error": float(residual.var()) / target_var if nonconstant else np.nan})
    return result


def validate_histogram(weights, edges):
    weights, edges = np.asarray(weights, float), np.asarray(edges, float)
    if weights.ndim != 2 or edges.ndim != 1 or edges.size != weights.shape[1] + 1:
        raise ValueError("Expected weights (queries,bins) and shared edges (bins+1,)")
    if not np.isfinite(weights).all() or not np.isfinite(edges).all():
        raise ValueError("Histogram contains nonfinite values")
    if np.any(weights < 0) or np.any(np.diff(edges) <= 0) or np.any(weights.sum(1) <= 0):
        raise ValueError("Histogram weights/edges are invalid")
    return weights / weights.sum(1, keepdims=True), edges


def histogram_moments(weights, edges):
    p, e = validate_histogram(weights, edges)
    centers, widths = (e[:-1] + e[1:]) / 2, np.diff(e)
    mean = p @ centers
    variance = (p * ((centers[None, :] - mean[:, None])**2 + widths[None, :]**2 / 12)).sum(1)
    return mean, variance


def histogram_quantiles(weights, edges, quantiles):
    p, e = validate_histogram(weights, edges)
    qs = np.asarray(quantiles, float).ravel()
    if not np.isfinite(qs).all() or np.any((qs < 0) | (qs > 1)):
        raise ValueError("Quantiles must be in [0,1]")
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    result = np.empty((p.shape[0], qs.size))
    for j, q in enumerate(qs):
        idx = np.minimum((cdf < q).sum(1), p.shape[1] - 1)
        rows = np.arange(p.shape[0])
        before = np.where(idx > 0, cdf[rows, np.maximum(idx - 1, 0)], 0.0)
        within = np.divide(q - before, p[rows, idx], out=np.zeros(p.shape[0]), where=p[rows, idx] > 0)
        result[:, j] = e[idx] + np.clip(within, 0, 1) * (e[idx + 1] - e[idx])
        if q == 0: result[:, j] = e[0]
        if q == 1: result[:, j] = e[-1]
    return result


def histogram_pit(weights, edges, truth):
    p, e = validate_histogram(weights, edges)
    y = np.asarray(truth, float).ravel()
    if y.size != p.shape[0]: raise ValueError("Truth length differs from histogram rows")
    fractions = np.clip((y[:, None] - e[None, :-1]) / np.diff(e)[None, :], 0, 1)
    return (p * fractions).sum(1)


def uncertainty_metrics(weights, edges, truth, levels=(0.5, 0.8, 0.9, 0.95)):
    p, e = validate_histogram(weights, edges)
    y = np.asarray(truth, float).ravel()
    mean, variance = histogram_moments(p, e)
    output = {"posterior_sd_mean": float(np.sqrt(variance).mean()),
              "posterior_variance_mean": float(variance.mean()),
              "truth_outside_histogram_fraction": float(np.mean((y < e[0]) | (y > e[-1]))),
              "posterior_boundary_mass_mean": float((p[:, 0] + p[:, -1]).mean())}
    intervals = {}
    for level in levels:
        if not 0 < level < 1: raise ValueError("Coverage levels must lie in (0,1)")
        bounds = histogram_quantiles(p, e, [(1-level)/2, (1+level)/2])
        key = f"{level:g}"
        output[f"coverage_{key}"] = float(np.mean((y >= bounds[:, 0]) & (y <= bounds[:, 1])))
        output[f"width_{key}"] = float(np.mean(bounds[:, 1] - bounds[:, 0]))
        intervals[key] = bounds
    return output, {"mean": mean, "variance": variance, "pit": histogram_pit(p,e,y), **intervals}
