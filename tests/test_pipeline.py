"""Sanity tests for the recommendation engine.

Run:  python -m pytest -q
  or: python tests/test_pipeline.py   (runs without pytest installed)

The dataset is loaded once and shared across tests (module-scoped) to keep the
suite fast.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.data import load_dataset
from src import models as M
from src.recommender import RecommenderSystem


# ---------------------------------------------------------------------------
# shared fixtures (works with or without pytest)
# ---------------------------------------------------------------------------
_DS = None
_REC = None


def get_ds():
    global _DS
    if _DS is None:
        _DS = load_dataset(verbose=False)
    return _DS


def get_rec():
    global _REC
    if _REC is None:
        _REC = RecommenderSystem.build(dataset=get_ds(), verbose=False)
    return _REC


# ---------------------------------------------------------------------------
# data pipeline
# ---------------------------------------------------------------------------
def test_filtering_thresholds():
    ds = get_ds()
    uc = ds.train.userId.value_counts()
    mc = ds.train.movieId.value_counts()
    assert uc.min() >= 1                                   # users survive
    # every movie kept in the full cleaned set met the >=5 threshold
    full_mc = ds.ratings.movieId.value_counts()
    assert full_mc.min() >= config.MIN_RATINGS_PER_MOVIE
    full_uc = ds.ratings.userId.value_counts()
    assert full_uc.min() >= config.MIN_RATINGS_PER_USER


def test_no_train_test_leakage():
    ds = get_ds()
    train_pairs = set(zip(ds.train.userId, ds.train.movieId))
    test_pairs = set(zip(ds.test.userId, ds.test.movieId))
    assert train_pairs.isdisjoint(test_pairs)


def test_split_ratio_roughly_80_20():
    ds = get_ds()
    frac = len(ds.test) / (len(ds.train) + len(ds.test))
    assert 0.15 < frac < 0.25


def test_matrix_shape_and_maps():
    ds = get_ds()
    assert ds.R.shape == (ds.n_users, ds.n_movies)
    assert len(ds.uid_to_idx) == ds.n_users
    assert len(ds.mid_to_idx) == ds.n_movies


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def test_all_models_fit_and_predict_in_range():
    ds = get_ds()
    lo, hi = config.RATING_SCALE
    for name, model in M.build_all_models().items():
        model.fit(ds)
        p = model.predict(int(ds.user_ids[0]), int(ds.movie_ids[0]))
        assert lo <= p <= hi, f"{name} predicted out of range: {p}"
        recs = model.recommend(int(ds.user_ids[0]), n=5)
        assert len(recs) == 5
        # recommendations must exclude already-seen movies
        seen = ds.seen_movie_indices(int(ds.user_ids[0]))
        seen_ids = {int(ds.movie_ids[j]) for j in seen}
        assert all(mid not in seen_ids for mid, _ in recs)


def test_svd_backend_available():
    ds = get_ds()
    svd = M.MatrixFactorizationCF().fit(ds)
    assert svd.backend in {"surprise", "numpy-sgd"}


def test_recommendations_are_ranked_desc():
    ds = get_ds()
    model = M.UserBasedCF().fit(ds)
    scores = [s for _, s in model.recommend(int(ds.user_ids[0]), n=10)]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# facade
# ---------------------------------------------------------------------------
def test_get_recommendations_shape():
    rec = get_rec()
    uid = int(rec.ds.user_ids[0])
    df = rec.get_recommendations(uid, n=10)
    assert len(df) == 10
    assert {"title", "score", "why"}.issubset(df.columns)


def test_cold_start_fallback():
    rec = get_rec()
    df = rec.get_recommendations(10_000_000, n=5)     # unknown user
    assert len(df) == 5
    assert bool(df["cold_start"].iloc[0]) is True


def test_similar_movies_excludes_self():
    rec = get_rec()
    mid = int(rec.ds.movie_ids[0])
    for method in ("content", "cf", "hybrid"):
        sim = rec.get_similar_movies(mid, n=8, method=method)
        assert mid not in set(sim["movieId"]), method
        assert len(sim) > 0, method


def test_new_user_fold_in():
    rec = get_rec()
    liked = [(int(rec.ds.movie_ids[0]), 5.0), (int(rec.ds.movie_ids[1]), 4.5)]
    df = rec.recommend_for_new_user(liked, n=8)
    assert len(df) == 8
    rated = {m for m, _ in liked}
    assert rated.isdisjoint(set(df["movieId"]))


# ---------------------------------------------------------------------------
# manual runner (no pytest needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)