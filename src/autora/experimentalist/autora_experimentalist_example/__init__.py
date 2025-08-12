# Random-Subset-Novelty (RSN)
# Exploit = run greedy max-min novelty inside a small random window

# Updated samplers
# - Random-Subset Novelty (RSN)
# - Stratified (under-covered bins) + RSN
#
# Usage: pick one at the bottom via:  sample = sample_rsn   or   sample = sample_stratified_rsn

import numpy as np
import pandas as pd
from typing import Union, Optional, Sequence


# ---------- helpers ----------

def _anti_join(pool: pd.DataFrame, tested: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Remove rows already in `tested` from `pool` (schema must match)."""
    if tested is None or tested.empty:
        return pool.copy()
    mask = ~pool.apply(tuple, axis=1).isin(tested.apply(tuple, axis=1))
    return pool.loc[mask]


def _grid_scalers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Index]:
    """Return (min, range, numeric_cols) computed on the (full) grid."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    rc = df[num_cols].astype(float)
    rc_min = rc.min()
    rc_rng = (rc.max() - rc_min).replace(0, 1.0)
    return rc_min, rc_rng, num_cols


def _scale(df: pd.DataFrame, rc_min: pd.Series, rc_rng: pd.Series, cols: Sequence[str]) -> np.ndarray:
    """Scale selected numeric cols of df to [0,1] using grid stats."""
    return ((df[cols].astype(float) - rc_min) / rc_rng).to_numpy()


def _greedy_maxmin_subset(
    subset_df: pd.DataFrame,      # candidates subset (rows to choose from)
    tested_arr: np.ndarray,       # scaled tested array (n_tested, d)
    rc_min: pd.Series, rc_rng: pd.Series, cols: Sequence[str],
    k: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Greedy farthest-first (k-center) selection on `subset_df` against `tested_arr`,
    returning `k` rows from `subset_df`. Incremental O(kN).
    """
    sub_arr = _scale(subset_df, rc_min, rc_rng, cols)

    # current min distance to union (starts with tested only)
    if tested_arr.size == 0:
        best_d = np.full(len(sub_arr), np.inf)
    else:
        best_d = np.linalg.norm(sub_arr[:, None, :] - tested_arr[None, :, :], axis=2).min(axis=1)

    # tiny jitter for deterministic tie-breaking
    if np.any(np.isfinite(best_d)):
        best_d = best_d + 1e-12 * rng.standard_normal(best_d.shape)

    chosen: list[int] = []
    k = min(k, len(sub_arr))
    for _ in range(k):
        i = int(np.argmax(best_d))
        chosen.append(i)
        # update distances with just the newly added point
        d_new = np.linalg.norm(sub_arr - sub_arr[i], axis=1)
        best_d = np.minimum(best_d, d_new)
        best_d[chosen] = -np.inf

    return subset_df.iloc[chosen].reset_index(drop=True)


def _bin_keys(df: pd.DataFrame, cols: Sequence[str], bins: int = 10) -> pd.Series:
    """Create a coarse grid key per row by binning numeric cols."""
    if len(cols) == 0 or df.empty:
        return pd.Series(["_all"] * len(df), index=df.index)
    binned = [pd.cut(df[c], bins=bins, labels=False, include_lowest=True, duplicates="drop") for c in cols]
    key = pd.concat(binned, axis=1).astype("Int64").astype(str).agg("-".join, axis=1)
    return key


def _under_coverage_weights(candidates: pd.DataFrame, tested: pd.DataFrame, cols: Sequence[str], bins: int = 10) -> np.ndarray:
    """Weights ∝ 1/(tested_count_in_bin + 1) to favor under-sampled regions."""
    cand_key = _bin_keys(candidates, cols, bins=bins)
    if tested is None or tested.empty:
        weights = np.ones(len(candidates), dtype=float)
        return (weights / weights.sum())
    tested_key = _bin_keys(tested, cols, bins=bins)
    counts = tested_key.value_counts()
    weights = 1.0 / (counts.reindex(cand_key).fillna(0.0).to_numpy() + 1.0)
    s = weights.sum()
    return (weights / s) if s > 0 else np.full_like(weights, 1.0 / len(weights))


# ---------- 2) Random-Subset Novelty (RSN) ----------

def sample_rsn(
    conditions: Union[pd.DataFrame, np.ndarray],          # tested so far
    reference_conditions: Union[pd.DataFrame, np.ndarray],# full grid / pool
    num_samples: int = 1,
    epsilon: float = 0.3,
    subset_factor: int = 5,       # subset size ≈ subset_factor * num_samples
    subset_cap: int = 200,        # hard cap on subset size
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    ε-random; otherwise run greedy max–min novelty **inside a random subset** of the pool.
    Keeps representativeness while still exploiting novelty.
    """
    # normalize inputs
    reference_conditions = pd.DataFrame(reference_conditions).copy()
    conditions = pd.DataFrame(conditions).copy()
    rng = np.random.default_rng(random_state)

    # build candidate pool (exclude tested)
    candidates = _anti_join(reference_conditions, conditions)
    if candidates.empty:
        return pd.DataFrame(columns=reference_conditions.columns).reset_index(drop=True)
    if len(candidates) <= num_samples:
        return candidates.reset_index(drop=True)

    # cold start → random
    if conditions.empty:
        return candidates.sample(n=num_samples, random_state=random_state).reset_index(drop=True)

    # grid scalers + arrays
    rc_min, rc_rng, num_cols = _grid_scalers(reference_conditions)
    tested_arr = _scale(conditions, rc_min, rc_rng, num_cols)

    # ε-explore
    if rng.random() < epsilon:
        return candidates.sample(n=num_samples, random_state=random_state).reset_index(drop=True)

    # exploit within random subset
    m = min(max(num_samples, subset_factor * num_samples), subset_cap, len(candidates))
    sub = candidates.sample(n=m, random_state=random_state, replace=False)

    return _greedy_maxmin_subset(
        subset_df=sub,
        tested_arr=tested_arr,
        rc_min=rc_min, rc_rng=rc_rng, cols=num_cols,
        k=num_samples,
        rng=rng,
    )


# ---------- 3) Stratified (under-covered) + RSN ----------

def sample_stratified_rsn(
    conditions: Union[pd.DataFrame, np.ndarray],          # tested so far
    reference_conditions: Union[pd.DataFrame, np.ndarray],# full grid / pool
    num_samples: int = 1,
    epsilon: float = 0.4,          # slightly higher explore rate
    bins: int = 10,                # stratification granularity
    subset_factor: int = 4,        # subset size ≈ subset_factor * num_samples
    subset_cap: int = 200,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Exploration: weighted random toward **under-covered bins** (1/(count+1)).
    Exploitation: draw a **weighted subset** by those weights, then run RSN within it.
    """
    # normalize inputs
    reference_conditions = pd.DataFrame(reference_conditions).copy()
    conditions = pd.DataFrame(conditions).copy()
    rng = np.random.default_rng(random_state)

    # build candidate pool (exclude tested)
    candidates = _anti_join(reference_conditions, conditions)
    if candidates.empty:
        return pd.DataFrame(columns=reference_conditions.columns).reset_index(drop=True)
    if len(candidates) <= num_samples:
        return candidates.reset_index(drop=True)

    # scalers + weights
    rc_min, rc_rng, num_cols = _grid_scalers(reference_conditions)
    tested_arr = _scale(conditions, rc_min, rc_rng, num_cols)
    weights = _under_coverage_weights(candidates, conditions, num_cols, bins=bins)
    w_series = pd.Series(weights, index=candidates.index)

    # cold start → weighted random
    if conditions.empty:
        return candidates.sample(n=num_samples, weights=w_series, random_state=random_state, replace=False).reset_index(drop=True)

    # ε-explore (weighted)
    if rng.random() < epsilon:
        return candidates.sample(n=num_samples, weights=w_series, random_state=random_state, replace=False).reset_index(drop=True)

    # exploit within a weighted subset, then greedy max–min
    m = min(max(num_samples, subset_factor * num_samples), subset_cap, len(candidates))
    sub = candidates.sample(n=m, weights=w_series, random_state=random_state, replace=False)

    return _greedy_maxmin_subset(
        subset_df=sub,
        tested_arr=tested_arr,
        rc_min=rc_min, rc_rng=rc_rng, cols=num_cols,
        k=num_samples,
        rng=rng,
    )


# ---------- pick one strategy for your pipeline ----------
# Default to RSN; switch to stratified by changing the alias.
#sample = sample_rsn
sample = sample_stratified_rsn
