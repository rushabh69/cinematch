"""Model evaluation: rating accuracy (RMSE/MAE) and ranking (precision/recall@k).

Rating metrics come from the held-out test ratings. Ranking metrics use a
top-N protocol: for each test user, the movies they rated >= LIKE_THRESHOLD in
the test set are the "relevant" set; we ask the model for its top-k and measure
the overlap.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config
from .data import Dataset


def rating_metrics(model, dataset: Dataset):
    """RMSE + MAE over the test split, one prediction-row per user."""
    preds, truths = [], []
    for uid, grp in dataset.test.groupby("userId"):
        if uid not in dataset.uid_to_idx:
            continue
        row = model.predict_all_ratings(uid)          # length n_movies, rating scale
        for mid, r in zip(grp.movieId.to_numpy(), grp.rating.to_numpy()):
            j = dataset.mid_to_idx.get(mid)
            if j is None:
                continue
            preds.append(row[j])
            truths.append(r)
    preds = np.asarray(preds, dtype=float)
    truths = np.asarray(truths, dtype=float)
    if len(preds) == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae = float(np.mean(np.abs(preds - truths)))
    return rmse, mae


def ranking_metrics(model, dataset: Dataset, k: int = config.EVAL_K,
                    like_threshold: float = config.LIKE_THRESHOLD):
    """Mean precision@k, recall@k and hit-rate over users with test 'likes'."""
    # relevant test movies per user (rated highly, and a column we know)
    relevant = defaultdict(set)
    for uid, mid, r in dataset.test[["userId", "movieId", "rating"]].itertuples(index=False):
        if r >= like_threshold and mid in dataset.mid_to_idx and uid in dataset.uid_to_idx:
            relevant[uid].add(mid)

    precisions, recalls, hits = [], [], []
    for uid, rel in relevant.items():
        if not rel:
            continue
        recs = model.recommend(uid, n=k, exclude_seen=True)
        rec_ids = {mid for mid, _ in recs}
        n_hit = len(rec_ids & rel)
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(rel))
        hits.append(1.0 if n_hit > 0 else 0.0)

    if not precisions:
        return {"precision@k": float("nan"), "recall@k": float("nan"),
                "hit_rate@k": float("nan"), "n_eval_users": 0}
    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "hit_rate@k": float(np.mean(hits)),
        "n_eval_users": len(precisions),
    }


def evaluate_model(name: str, model, dataset: Dataset, k: int = config.EVAL_K,
                   fit: bool = True) -> dict:
    t0 = time.perf_counter()
    if fit:
        model.fit(dataset)
    fit_time = time.perf_counter() - t0

    rmse, mae = rating_metrics(model, dataset)
    rank = ranking_metrics(model, dataset, k=k)

    backend = getattr(model, "backend", None)
    return {
        "model": name + (f" [{backend}]" if backend else ""),
        "RMSE": rmse,
        "MAE": mae,
        f"Precision@{k}": rank["precision@k"],
        f"Recall@{k}": rank["recall@k"],
        f"HitRate@{k}": rank["hit_rate@k"],
        "fit_seconds": round(fit_time, 2),
    }


def compare_models(models: dict, dataset: Dataset, k: int = config.EVAL_K) -> pd.DataFrame:
    """Fit + evaluate a {name: model} dict, return a tidy comparison table."""
    rows = []
    for name, model in models.items():
        print(f"  evaluating {name} ...", flush=True)
        rows.append(evaluate_model(name, model, dataset, k=k))
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("RMSE")