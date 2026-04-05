from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, confusion_matrix

from teda import TEDA


@dataclass
class TemporalSplit:
    idx_baseline: np.ndarray
    idx_tune: np.ndarray
    idx_test: np.ndarray
    idx_pretest: np.ndarray


def time_since_last_occurrence(series: Iterable[float]) -> np.ndarray:
    last_seen = {}
    out = np.empty(len(series), dtype=float)
    for idx, value in enumerate(series):
        if pd.isna(value):
            out[idx] = np.nan
            continue
        out[idx] = idx - last_seen[value] if value in last_seen else np.nan
        last_seen[value] = idx
    return out


def build_window_hist2d(
    X,
    y,
    window_size: int = 500,
    step: int = 1,
    x_bins=None,
    y_bins=None,
    y_log: bool = False,
    y_clip: Optional[float] = None,
    fit_samples: Optional[int] = None,
):
    X = np.asarray(X)
    y = np.asarray(y)

    if fit_samples is None:
        X_fit = X
        y_fit = y
    else:
        fit_samples = int(max(1, min(len(X), fit_samples)))
        X_fit = X[:fit_samples]
        y_fit = y[:fit_samples]

    if x_bins is None:
        x_vals = np.unique(X_fit[~pd.isna(X_fit)])
        if len(x_vals) == 0:
            x_vals = np.unique(X[~pd.isna(X)])
    else:
        x_vals = np.asarray(x_bins)
    x_to_idx = {v: i for i, v in enumerate(x_vals)}

    y_work = y_fit[~np.isnan(y_fit)]
    if y_clip is not None:
        y_work = np.clip(y_work, 0, y_clip)
    if y_log:
        y_work = np.log1p(y_work)

    if y_work.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.nanpercentile(y_work, 1), np.nanpercentile(y_work, 99)
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0

    if y_bins is None:
        y_edges = np.linspace(lo, hi, 25)
    else:
        y_edges = np.asarray(y_bins)

    starts = np.arange(0, len(X) - window_size + 1, step, dtype=int)
    H = np.zeros((len(starts), len(x_vals), len(y_edges) - 1), dtype=np.int32)

    for k, start in enumerate(starts):
        end = start + window_size
        Xw = X[start:end]
        yw = y[start:end]

        mask = ~pd.isna(Xw) & ~np.isnan(yw)
        Xw = Xw[mask]
        yw = yw[mask]

        if y_clip is not None:
            yw = np.clip(yw, 0, y_clip)
        if y_log:
            yw = np.log1p(yw)

        y_bin_idx = np.searchsorted(y_edges, yw, side="right") - 1
        valid = (y_bin_idx >= 0) & (y_bin_idx < (len(y_edges) - 1))
        Xw = Xw[valid]
        y_bin_idx = y_bin_idx[valid]

        for xv, yb in zip(Xw, y_bin_idx):
            idx = x_to_idx.get(xv)
            if idx is not None:
                H[k, idx, yb] += 1

    return H, x_vals, y_edges, starts


def normalize_window(Hw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    Hw = Hw.astype(np.float32)
    total = Hw.sum()
    return Hw / max(total, eps)


def sample_to_window_labels(sample_labels: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    sample_labels = np.asarray(sample_labels, dtype=np.uint8)
    starts = np.asarray(starts, dtype=int)
    csum = np.concatenate([[0], np.cumsum(sample_labels)])
    ends = np.minimum(starts + window_size, len(sample_labels))
    counts = csum[ends] - csum[starts]
    return counts > 0


def build_composite_token(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
    fit_samples: Optional[int] = None,
):
    idle = df["Idle(bit)"].to_numpy(dtype=float)
    length = df["Len(byte)"].to_numpy(dtype=float)

    mask = ~np.isnan(idle)
    fit_mask = mask.copy()
    if fit_samples is not None:
        fit_samples = int(max(1, min(len(idle), fit_samples)))
        fit_mask[fit_samples:] = False

    idle_fit = idle[fit_mask]
    if idle_fit.size == 0:
        idle_fit = idle[mask]
    unique_fit = np.unique(idle_fit)
    effective_clusters = max(1, min(int(n_clusters), len(unique_fit)))

    km = KMeans(n_clusters=effective_clusters, n_init=10, random_state=random_state)
    labels = np.full(len(idle), fill_value=-1, dtype=int)
    km.fit(idle_fit.reshape(-1, 1))
    labels[mask] = km.predict(idle[mask].reshape(-1, 1))

    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    idle_class = np.array([remap.get(v, -1) for v in labels], dtype=int)

    x_comp = np.empty(len(df), dtype=object)
    for i in range(len(df)):
        if np.isnan(length[i]) or idle_class[i] < 0:
            x_comp[i] = np.nan
        else:
            x_comp[i] = (int(length[i]), int(idle_class[i]))

    return x_comp, idle_class, np.sort(centers)


def inject_synthetic_anomalies(
    df: pd.DataFrame,
    warmup: int = 20_000,
    n_events: int = 12,
    min_len: int = 250,
    max_len: int = 1200,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    out = df.copy()
    # Force writable copies because pandas may expose read-only views here.
    length = out["Len(byte)"].to_numpy(dtype=float, copy=True)
    idle = out["Idle(bit)"].to_numpy(dtype=float, copy=True)

    n = len(out)
    labels = np.zeros(n, dtype=np.uint8)
    used = np.zeros(n, dtype=bool)
    events = []

    l_min = np.nanmin(length)
    l_max = np.nanmax(length)
    i_min = np.nanmin(idle)
    i_max = np.nanmax(idle)

    len_hi = max(np.nanpercentile(length, 95), l_max * 0.85)
    idle_std = np.nanstd(idle)
    idle_center = np.nanpercentile(idle, 70)

    event_types = ["len_burst", "idle_delay", "period_shift", "combined"]
    gap = min_len // 2

    for i in range(n_events):
        ev_type = event_types[i % len(event_types)]
        ev_len = int(rng.integers(min_len, max_len + 1))

        start = end = None
        for _ in range(4000):
            candidate_start = int(rng.integers(warmup, n - ev_len - 1))
            candidate_end = candidate_start + ev_len
            lo = max(0, candidate_start - gap)
            hi = min(n, candidate_end + gap)
            if not used[lo:hi].any():
                start, end = candidate_start, candidate_end
                break

        if start is None:
            continue

        used[start:end] = True
        labels[start:end] = 1

        if ev_type == "len_burst":
            length[start:end] = rng.choice([len_hi, l_max], size=ev_len, replace=True)
        elif ev_type == "idle_delay":
            factor = float(rng.uniform(2.0, 3.5))
            idle[start:end] = np.clip(idle[start:end] * factor, i_min, i_max)
        elif ev_type == "period_shift":
            noise = rng.normal(0, max(3.0, idle_std * 0.15), size=ev_len)
            idle[start:end] = np.clip(idle_center + noise, i_min, i_max)
            n_spikes = max(1, ev_len // 25)
            spike_pos = start + rng.choice(ev_len, size=n_spikes, replace=False)
            idle[spike_pos] = i_max
        elif ev_type == "combined":
            length[start:end] = rng.choice([len_hi, l_max], size=ev_len, replace=True)
            factor = float(rng.uniform(2.0, 3.0))
            idle[start:end] = np.clip(idle[start:end] * factor, i_min, i_max)

        events.append({"event_id": i, "type": ev_type, "start": start, "end": end, "length": ev_len})

    out["Len(byte)"] = np.clip(np.rint(length), l_min, l_max)
    out["Idle(bit)"] = np.clip(np.rint(idle), i_min, i_max)

    events_df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
    return out, labels, events_df


def prepare_temporal_split(n_windows: int, ref_windows: int) -> TemporalSplit:
    baseline_end = ref_windows
    tune_end = int(0.60 * n_windows)
    if tune_end <= baseline_end + 100:
        raise ValueError("Tune segment became too short; adjust REF_WINDOWS or STEP.")

    return TemporalSplit(
        idx_baseline=np.arange(0, baseline_end),
        idx_tune=np.arange(baseline_end, tune_end),
        idx_test=np.arange(tune_end, n_windows),
        idx_pretest=np.arange(0, tune_end),
    )


def _zscore_from_ref(a: np.ndarray, ref_windows: int, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    ref = a[:ref_windows]
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    return (a - mu) / max(sd, eps)


def _safe_prob_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    s = float(np.sum(x))
    if s <= eps:
        return np.full_like(x, 1.0 / max(x.size, 1), dtype=np.float32)
    x = x / s
    return np.clip(x, eps, None)


def _entropy(p: np.ndarray) -> float:
    p = _safe_prob_vector(p)
    return float(-(p * np.log(p)).sum())


def _bhattacharyya_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    coeff = float(np.sqrt(np.maximum(p, eps) * np.maximum(q, eps)).sum())
    coeff = min(max(coeff, eps), 1.0)
    return float(-np.log(coeff))


def _symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(0.5 * ((p * np.log(p / q)).sum() + (q * np.log(q / p)).sum()))


def _avg_pool_matrix(mat: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    h, w = mat.shape
    out_h, out_w = out_shape
    y_edges = np.linspace(0, h, out_h + 1, dtype=int)
    x_edges = np.linspace(0, w, out_w + 1, dtype=int)
    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            block = mat[y_edges[i]:y_edges[i + 1], x_edges[j]:x_edges[j + 1]]
            pooled[i, j] = float(block.mean()) if block.size else 0.0
    return pooled


def compute_reference_matrix(H: np.ndarray, ref_windows: int, ref_mode: str = "mean") -> np.ndarray:
    H_ref = H[:ref_windows].astype(np.float32)
    if ref_mode == "mean":
        ref = H_ref.mean(axis=0)
    elif ref_mode == "median":
        ref = np.median(H_ref, axis=0)
    else:
        raise ValueError("ref_mode must be 'mean' or 'median'")
    return normalize_window(ref)


def build_quantification_features(
    H: np.ndarray,
    ref_windows: int,
    pool_shape: Tuple[int, int] = (8, 8),
    random_projection_dim: int = 12,
    random_state: int = 42,
) -> Dict[str, object]:
    H_norm = np.stack([normalize_window(h) for h in H]).astype(np.float32)
    n_windows = H_norm.shape[0]
    flat = H_norm.reshape(n_windows, -1)
    ref = compute_reference_matrix(H, ref_windows)
    ref_flat = ref.ravel()
    prev_flat = np.vstack([flat[0], flat[:-1]])

    diff_ref = flat - ref_flat
    diff_prev = flat - prev_flat

    js_ref = np.array([jensenshannon(row, ref_flat) for row in flat], dtype=np.float32)
    fro_ref = np.linalg.norm(H_norm - ref, axis=(1, 2)).astype(np.float32)
    l1_ref = np.abs(diff_ref).sum(axis=1).astype(np.float32)
    l2_ref = np.linalg.norm(diff_ref, axis=1).astype(np.float32)
    linf_ref = np.max(np.abs(diff_ref), axis=1).astype(np.float32)
    cosine_ref = cdist(flat, ref_flat[None, :], metric="cosine").ravel().astype(np.float32)
    corr_ref = cdist(flat, ref_flat[None, :], metric="correlation").ravel().astype(np.float32)
    city_prev = np.abs(diff_prev).sum(axis=1).astype(np.float32)
    l2_prev = np.linalg.norm(diff_prev, axis=1).astype(np.float32)
    cosine_prev = np.array(
        [0.0 if i == 0 else float(cdist(flat[i:i + 1], flat[i - 1:i], metric="cosine")[0, 0]) for i in range(n_windows)],
        dtype=np.float32,
    )
    js_prev = np.array(
        [0.0 if i == 0 else jensenshannon(flat[i], flat[i - 1]) for i in range(n_windows)],
        dtype=np.float32,
    )
    bhat_ref = np.array([_bhattacharyya_distance(row, ref_flat) for row in flat], dtype=np.float32)
    hist_intersection_ref = np.array([1.0 - np.minimum(row, ref_flat).sum() for row in flat], dtype=np.float32)
    skl_ref = np.array([_symmetric_kl(row, ref_flat) for row in flat], dtype=np.float32)

    entropy = np.array([_entropy(row) for row in flat], dtype=np.float32)
    sparsity = np.mean(H_norm <= 0.0, axis=(1, 2)).astype(np.float32)
    energy = np.sum(H_norm ** 2, axis=(1, 2)).astype(np.float32)
    max_bin = np.max(H_norm, axis=(1, 2)).astype(np.float32)

    y_idx = np.arange(H_norm.shape[1], dtype=np.float32)
    x_idx = np.arange(H_norm.shape[2], dtype=np.float32)
    mass_y = H_norm.sum(axis=2)
    mass_x = H_norm.sum(axis=1)
    centroid_y = (mass_y * y_idx).sum(axis=1).astype(np.float32)
    centroid_x = (mass_x * x_idx).sum(axis=1).astype(np.float32)

    diag_mass = np.array(
        [float(np.trace(_avg_pool_matrix(h, pool_shape))) for h in H_norm],
        dtype=np.float32,
    )
    edge_mass = (
        H_norm[:, 0, :].sum(axis=1)
        + H_norm[:, -1, :].sum(axis=1)
        + H_norm[:, :, 0].sum(axis=1)
        + H_norm[:, :, -1].sum(axis=1)
    ).astype(np.float32)
    rare_mass = H_norm[:, :, int(np.floor(0.75 * H_norm.shape[2])):].sum(axis=(1, 2)).astype(np.float32)

    pooled = np.stack([_avg_pool_matrix(h, pool_shape) for h in H_norm]).astype(np.float32)
    pooled_flat = pooled.reshape(n_windows, -1)
    pooled_ref = pooled_flat[:ref_windows].mean(axis=0)
    pooled_l2_ref = np.linalg.norm(pooled_flat - pooled_ref, axis=1).astype(np.float32)
    pooled_l2_prev = np.linalg.norm(pooled_flat - np.vstack([pooled_flat[0], pooled_flat[:-1]]), axis=1).astype(np.float32)

    flat_proj_input = flat.astype(np.float64, copy=False)
    proj_dim = max(1, min(random_projection_dim, flat.shape[1]))
    rng = np.random.default_rng(random_state)
    random_basis = rng.normal(0.0, 1.0 / math.sqrt(flat.shape[1]), size=(flat.shape[1], proj_dim))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        flat_rp = np.matmul(flat_proj_input, random_basis).astype(np.float32)
    rp_ref = flat_rp[:ref_windows].mean(axis=0)
    rp_l2_ref = np.linalg.norm(flat_rp - rp_ref, axis=1).astype(np.float32)

    coord = np.arange(flat.shape[1], dtype=np.float64)
    cosine_basis = np.stack(
        [np.cos(np.pi * (k + 1) * (coord + 0.5) / max(flat.shape[1], 1)) for k in range(proj_dim)],
        axis=1,
    )
    cosine_basis /= np.maximum(np.linalg.norm(cosine_basis, axis=0, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        flat_svd = np.matmul(flat_proj_input, cosine_basis).astype(np.float32)
    svd_ref = flat_svd[:ref_windows].mean(axis=0)
    svd_l2_ref = np.linalg.norm(flat_svd - svd_ref, axis=1).astype(np.float32)

    feature_map = {
        "js_ref": js_ref,
        "fro_ref": fro_ref,
        "l1_ref": l1_ref,
        "l2_ref": l2_ref,
        "linf_ref": linf_ref,
        "cosine_ref": cosine_ref,
        "corr_ref": np.nan_to_num(corr_ref, nan=0.0, posinf=0.0, neginf=0.0),
        "city_prev": city_prev,
        "l2_prev": l2_prev,
        "cosine_prev": cosine_prev,
        "js_prev": js_prev,
        "bhat_ref": bhat_ref,
        "hist_intersection_ref": hist_intersection_ref,
        "skl_ref": skl_ref,
        "entropy": entropy,
        "sparsity": sparsity,
        "energy": energy,
        "max_bin": max_bin,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "diag_mass": diag_mass,
        "edge_mass": edge_mass,
        "rare_mass": rare_mass,
        "pooled_l2_ref": pooled_l2_ref,
        "pooled_l2_prev": pooled_l2_prev,
        "rp_l2_ref": rp_l2_ref,
        "svd_l2_ref": svd_l2_ref,
    }

    feature_df = pd.DataFrame(feature_map)
    delta_cols = []
    z_cols = []
    for col in list(feature_df.columns):
        delta_col = f"d_{col}"
        z_col = f"z_{col}"
        feature_df[delta_col] = np.diff(feature_df[col], prepend=feature_df[col].iloc[0]).astype(np.float32)
        feature_df[z_col] = _zscore_from_ref(feature_df[col].to_numpy(), ref_windows)
        delta_cols.append(delta_col)
        z_cols.append(z_col)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_df = feature_df.clip(-1e4, 1e4).astype(np.float32)

    mv_cols = ["z_js_ref", "z_fro_ref", "z_l2_ref", "z_cosine_ref", "z_rare_mass", "z_pooled_l2_ref"]
    mv_cols = [col for col in mv_cols if col in feature_df.columns]

    return {
        "H": H,
        "H_norm": H_norm,
        "reference": ref,
        "pooled": pooled,
        "pooled_flat": pooled_flat,
        "flat": flat,
        "flat_rp": flat_rp,
        "flat_svd": flat_svd,
        "feature_df": feature_df,
        "feature_names": list(feature_df.columns),
        "mv_cols": mv_cols,
        "mv_feats": feature_df[mv_cols].to_numpy(dtype=np.float32),
    }


def build_representation_outputs(
    name: str,
    X,
    window_size: int,
    step: int,
    y_clip: Optional[float],
    ref_windows: int,
    pool_shape: Tuple[int, int] = (8, 8),
    random_projection_dim: int = 12,
    random_state: int = 42,
    fit_samples: Optional[int] = None,
) -> Dict[str, object]:
    y = time_since_last_occurrence(X)
    H, x_vals, y_edges, starts = build_window_hist2d(
        X,
        y,
        window_size=window_size,
        step=step,
        y_clip=y_clip,
        fit_samples=fit_samples,
    )
    quant = build_quantification_features(
        H=H,
        ref_windows=ref_windows,
        pool_shape=pool_shape,
        random_projection_dim=random_projection_dim,
        random_state=random_state,
    )
    quant.update(
        {
            "name": name,
            "starts": starts,
            "window_size": window_size,
            "x_vals": x_vals,
            "y_edges": y_edges,
            "x_bins_count": int(len(x_vals)),
            "y_bins_count": int(len(y_edges) - 1),
        }
    )
    return quant


def tune_threshold_from_scores(scores: np.ndarray, y_true: np.ndarray, idx_tune: np.ndarray) -> Tuple[float, Dict[str, float]]:
    score_tune = np.asarray(scores, dtype=float)[idx_tune]
    if len(score_tune) == 0:
        return float("nan"), {}

    quantiles = np.linspace(0.80, 0.999, 48)
    thresholds = np.unique(np.quantile(score_tune, quantiles))
    best_thr = thresholds[0]
    best_metrics = None
    best_obj = -np.inf

    for thr in thresholds:
        flags = np.asarray(scores) > thr
        metrics = binary_metrics(y_true[idx_tune], flags[idx_tune], np.asarray(scores)[idx_tune])
        if metrics["objective_score"] > best_obj:
            best_obj = metrics["objective_score"]
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics or {}


def teda_flags(series: np.ndarray, thr: float = 2.0) -> np.ndarray:
    s = np.asarray(series, dtype=np.float32)
    model = TEDA(threshold=thr)
    out = np.zeros(len(s), dtype=np.int8)
    for i, value in enumerate(s):
        out[i] = model.run(np.array([value], dtype=np.float32))
    return out.astype(bool)


def teda_flags_multivariate(features: np.ndarray, thr: float = 2.0) -> np.ndarray:
    F = np.asarray(features, dtype=np.float32)
    model = TEDA(threshold=thr)
    out = np.zeros(F.shape[0], dtype=np.int8)
    for i in range(F.shape[0]):
        out[i] = model.run(F[i])
    return out.astype(bool)


def tune_teda_threshold_univariate(series: np.ndarray, y_true: np.ndarray, idx_tune: np.ndarray, thr_grid: np.ndarray):
    best = None
    best_obj = -np.inf
    for thr in thr_grid:
        flags = teda_flags(series, float(thr))
        metrics = binary_metrics(y_true[idx_tune], flags[idx_tune], np.asarray(series)[idx_tune])
        if metrics["objective_score"] > best_obj:
            best_obj = metrics["objective_score"]
            best = (float(thr), flags, metrics)
    return best


def tune_teda_threshold_multivariate(features: np.ndarray, y_true: np.ndarray, idx_tune: np.ndarray, thr_grid: np.ndarray):
    best = None
    best_obj = -np.inf
    score = np.linalg.norm(np.asarray(features), axis=1)
    for thr in thr_grid:
        flags = teda_flags_multivariate(features, float(thr))
        metrics = binary_metrics(y_true[idx_tune], flags[idx_tune], score[idx_tune])
        if metrics["objective_score"] > best_obj:
            best_obj = metrics["objective_score"]
            best = (float(thr), flags, metrics)
    return best


def binary_metrics(
    y_true: np.ndarray,
    flags: np.ndarray,
    score: Optional[np.ndarray] = None,
    w_f1: float = 1.0,
    w_far: float = 0.35,
    w_bacc: float = 0.25,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    flags = np.asarray(flags, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y_true, flags.astype(np.uint8), labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (recall + tnr)

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    avg_precision = np.nan
    if score is not None and np.unique(y_true).size > 1:
        avg_precision = float(average_precision_score(y_true, np.asarray(score, dtype=float)))

    objective_score = w_f1 * f1 - w_far * far + w_bacc * balanced_accuracy
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(far),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "avg_precision": float(avg_precision) if not np.isnan(avg_precision) else np.nan,
        "objective_score": float(objective_score),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def subsample_indices(y: np.ndarray, max_samples: int, seed: int = 42, stratify: bool = True) -> np.ndarray:
    y = np.asarray(y)
    if len(y) <= max_samples:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    if not stratify or np.unique(y).size < 2:
        return np.sort(rng.choice(len(y), size=max_samples, replace=False))

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n1 = min(len(idx1), max(1, int(max_samples * len(idx1) / len(y))))
    n0 = max_samples - n1
    chosen = np.concatenate(
        [
            rng.choice(idx0, size=min(len(idx0), n0), replace=False),
            rng.choice(idx1, size=min(len(idx1), n1), replace=False),
        ]
    )
    return np.sort(chosen)


def build_pair_dataset(
    H_norm: np.ndarray,
    y_windows: np.ndarray,
    pool_shape: Optional[Tuple[int, int]] = (8, 8),
    selected_pair_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    if pool_shape is None:
        pooled = np.asarray(H_norm, dtype=np.float32)
    else:
        pooled = np.stack([_avg_pool_matrix(h, pool_shape) for h in H_norm]).astype(np.float32)

    all_pair_indices = np.arange(max(pooled.shape[0] - 1, 0), dtype=int)
    if selected_pair_indices is None:
        pair_indices = all_pair_indices
    else:
        pair_indices = np.asarray(selected_pair_indices, dtype=int)
        pair_indices = pair_indices[(pair_indices >= 0) & (pair_indices < len(all_pair_indices))]

    prev_idx = pair_indices
    cur_idx = pair_indices + 1
    pair = np.stack([pooled[prev_idx], pooled[cur_idx]], axis=1) if len(pair_indices) else np.zeros((0, 2, *pooled.shape[1:]), dtype=np.float32)
    pair_flat = pair.reshape(pair.shape[0], -1)
    pair_delta = (pair[:, 1] - pair[:, 0]).reshape(pair.shape[0], -1)
    return {
        "pair_image": pair.astype(np.float32, copy=False),
        "pair_flat": pair_flat.astype(np.float32, copy=False),
        "pair_delta": pair_delta.astype(np.float32, copy=False),
        "labels": np.asarray(y_windows[cur_idx], dtype=np.uint8),
        "current_window_index": cur_idx,
    }


def _topk_union_nonzero_positions(prev_mat: np.ndarray, cur_mat: np.ndarray, top_k: int) -> np.ndarray:
    support = (prev_mat > 0.0) | (cur_mat > 0.0)
    coords = np.argwhere(support)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=int)
    scores = np.array(
        [max(float(prev_mat[r, c]), float(cur_mat[r, c]), abs(float(cur_mat[r, c] - prev_mat[r, c]))) for r, c in coords],
        dtype=np.float32,
    )
    order = np.argsort(scores)[::-1]
    return coords[order[:top_k]]


def build_sparse_token_pair_dataset(
    H_norm: np.ndarray,
    y_windows: np.ndarray,
    top_k: int = 32,
    selected_pair_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    H_norm = np.asarray(H_norm, dtype=np.float32)
    all_pair_indices = np.arange(max(H_norm.shape[0] - 1, 0), dtype=int)
    if selected_pair_indices is None:
        pair_indices = all_pair_indices
    else:
        pair_indices = np.asarray(selected_pair_indices, dtype=int)
        pair_indices = pair_indices[(pair_indices >= 0) & (pair_indices < len(all_pair_indices))]

    n_pairs = len(pair_indices)
    token_features = np.zeros((n_pairs, top_k, 7), dtype=np.float32)
    token_mask = np.zeros((n_pairs, top_k), dtype=np.float32)

    h_denom = float(max(H_norm.shape[1] - 1, 1))
    w_denom = float(max(H_norm.shape[2] - 1, 1))

    for out_idx, pair_idx in enumerate(pair_indices):
        prev_mat = H_norm[pair_idx]
        cur_mat = H_norm[pair_idx + 1]
        coords = _topk_union_nonzero_positions(prev_mat, cur_mat, top_k=top_k)
        for token_idx, (row, col) in enumerate(coords):
            prev_val = float(prev_mat[row, col])
            cur_val = float(cur_mat[row, col])
            token_features[out_idx, token_idx] = np.array(
                [
                    row / h_denom,
                    col / w_denom,
                    prev_val,
                    cur_val,
                    cur_val - prev_val,
                    1.0 if prev_val > 0.0 else 0.0,
                    1.0 if cur_val > 0.0 else 0.0,
                ],
                dtype=np.float32,
            )
            token_mask[out_idx, token_idx] = 1.0

    current_window_index = pair_indices + 1
    return {
        "token_features": token_features,
        "token_mask": token_mask,
        "token_flat": token_features.reshape(n_pairs, -1).astype(np.float32, copy=False),
        "labels": np.asarray(y_windows[current_window_index], dtype=np.uint8),
        "current_window_index": current_window_index,
    }


def build_pair_split(window_split: TemporalSplit, current_window_index: np.ndarray) -> TemporalSplit:
    current_window_index = np.asarray(current_window_index)
    return TemporalSplit(
        idx_baseline=np.where(np.isin(current_window_index, window_split.idx_baseline))[0],
        idx_tune=np.where(np.isin(current_window_index, window_split.idx_tune))[0],
        idx_test=np.where(np.isin(current_window_index, window_split.idx_test))[0],
        idx_pretest=np.where(np.isin(current_window_index, window_split.idx_pretest))[0],
    )


def count_parameters(model) -> int:
    if model is None:
        return 0
    if hasattr(model, "parameters"):
        return int(sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", True)))
    if hasattr(model, "coef_"):
        return int(np.size(model.coef_) + np.size(getattr(model, "intercept_", 0)))
    if hasattr(model, "coefs_"):
        total = sum(np.size(w) for w in model.coefs_)
        total += sum(np.size(b) for b in getattr(model, "intercepts_", []))
        return int(total)
    if hasattr(model, "estimators_"):
        total = 0
        for est in model.estimators_:
            if isinstance(est, np.ndarray):
                for item in est.flat:
                    if hasattr(item, "tree_"):
                        total += int(item.tree_.node_count)
            elif hasattr(est, "tree_"):
                total += int(est.tree_.node_count)
        return int(total)
    if hasattr(model, "support_vectors_"):
        return int(np.size(model.support_vectors_))
    return 0


def attach_cost_metadata(
    results_df: pd.DataFrame,
    priors: Dict[str, float],
) -> pd.DataFrame:
    df = results_df.copy()
    for col in ["estimated_input_dim", "estimated_params", "estimated_ops", "artifact_bytes"]:
        if col not in df.columns:
            df[col] = np.nan

    def inv_log_norm(series: pd.Series) -> pd.Series:
        s = np.log1p(series.fillna(series.max() if series.notna().any() else 1.0))
        if float(s.max() - s.min()) < 1e-12:
            return pd.Series(np.ones(len(s)), index=s.index)
        return 1.0 - (s - s.min()) / (s.max() - s.min())

    df["embedded_prior"] = df["modelo"].map(priors).fillna(0.35)
    df["input_fit"] = inv_log_norm(df["estimated_input_dim"])
    df["param_fit"] = inv_log_norm(df["estimated_params"])
    df["ops_fit"] = inv_log_norm(df["estimated_ops"])
    df["memory_fit"] = inv_log_norm(df["artifact_bytes"])
    df["embedded_score"] = (
        0.35 * df["embedded_prior"]
        + 0.20 * df["input_fit"]
        + 0.20 * df["param_fit"]
        + 0.15 * df["ops_fit"]
        + 0.10 * df["memory_fit"]
    )
    df["final_rank_score"] = 0.70 * df["objective_score"].fillna(0.0) + 0.30 * df["embedded_score"].fillna(0.0)
    return df
