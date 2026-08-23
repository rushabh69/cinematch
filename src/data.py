"""Data pipeline: load, clean, filter, split, and index the ratings.

The main entry point is load_dataset(), which hands back a Dataset object
bundling the cleaned ratings, an 80/20 train/test split, the movie metadata,
and a sparse user x item matrix plus the id<->index maps every model uses.
"""
from __future__ import annotations

import io
import zipfile
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from . import config


def ensure_dataset() -> None:
    """Download + extract ml-latest-small if we don't already have it."""
    if config.RATINGS_CSV.exists():
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens dataset from {config.DATASET_URL} ...")
    with urllib.request.urlopen(config.DATASET_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(config.DATA_DIR)
    print(f"Extracted to {config.DATASET_DIR}")


@dataclass
class Dataset:
    """Everything a model needs, computed once from the training split."""

    ratings: pd.DataFrame          # full cleaned ratings (train + test)
    train: pd.DataFrame            # 80%
    test: pd.DataFrame             # 20%
    movies: pd.DataFrame           # movieId, title, genres, year
    tags: pd.DataFrame             # raw tags

    # id <-> matrix index maps
    user_ids: np.ndarray = field(default=None)
    movie_ids: np.ndarray = field(default=None)
    uid_to_idx: dict = field(default=None)
    mid_to_idx: dict = field(default=None)

    # training user x item sparse matrix (CSR) of raw ratings
    R: sparse.csr_matrix = field(default=None)

    # stats derived from the training matrix
    global_mean: float = 0.0
    user_means: np.ndarray = field(default=None)
    movie_means: np.ndarray = field(default=None)

    def has_user(self, user_id: int) -> bool:
        return user_id in self.uid_to_idx

    def has_movie(self, movie_id: int) -> bool:
        return movie_id in self.mid_to_idx

    def title(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies.movieId == movie_id, "title"]
        return row.iloc[0] if len(row) else f"movie {movie_id}"

    def genres(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies.movieId == movie_id, "genres"]
        return row.iloc[0] if len(row) else ""

    def seen_movie_indices(self, user_id: int) -> set:
        """Column indices this user rated in the training matrix."""
        if user_id not in self.uid_to_idx:
            return set()
        u = self.uid_to_idx[user_id]
        return set(self.R[u].indices)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)


def _extract_year(title: str):
    """Pull the release year out of a title like 'Toy Story (1995)'."""
    import re

    m = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    return int(m.group(1)) if m else np.nan


def clean_ratings(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows and any rating pointing at a movie we don't know."""
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    # keep only ratings inside the valid scale that point at a known movie
    lo, hi = config.RATING_SCALE
    ratings = ratings[(ratings.rating >= lo) & (ratings.rating <= hi)]
    ratings = ratings[ratings.movieId.isin(set(movies.movieId))]
    return ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")


def filter_sparse(ratings: pd.DataFrame,
                  min_user: int = config.MIN_RATINGS_PER_USER,
                  min_movie: int = config.MIN_RATINGS_PER_MOVIE) -> pd.DataFrame:
    """Drop users/movies with too few ratings.

    One pass isn't enough: removing thinly-rated movies can push some users
    below the user threshold, so we loop until nothing else gets dropped.
    """
    cur = ratings
    while True:
        uc = cur.userId.value_counts()
        mc = cur.movieId.value_counts()
        keep_u = uc[uc >= min_user].index
        keep_m = mc[mc >= min_movie].index
        nxt = cur[cur.userId.isin(keep_u) & cur.movieId.isin(keep_m)]
        if len(nxt) == len(cur):
            return nxt.reset_index(drop=True)
        cur = nxt


def train_test_split_per_user(ratings: pd.DataFrame,
                              test_size: float = config.TEST_SIZE,
                              seed: int = config.RANDOM_SEED):
    """Hold out test_size of *each user's* ratings.

    Splitting per-user (instead of globally) means every user keeps some
    training history, so no user is impossible to score at evaluation time.
    """
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("userId"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = min(n_test, len(idx) - 1)   # never move a user's whole history to test
        test_idx.extend(idx[:n_test])
    test_mask = ratings.index.isin(test_idx)
    train = ratings[~test_mask].reset_index(drop=True)
    test = ratings[test_mask].reset_index(drop=True)
    return train, test


def _build_matrix(train: pd.DataFrame, user_ids, movie_ids):
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    mid_to_idx = {m: i for i, m in enumerate(movie_ids)}
    rows = train.userId.map(uid_to_idx).to_numpy()
    cols = train.movieId.map(mid_to_idx).to_numpy()
    vals = train.rating.to_numpy(dtype=np.float64)
    R = sparse.csr_matrix((vals, (rows, cols)),
                          shape=(len(user_ids), len(movie_ids)))
    return R, uid_to_idx, mid_to_idx


def _matrix_stats(R: sparse.csr_matrix):
    """Global mean plus per-user and per-movie means over observed entries."""
    counts_u = np.asarray((R != 0).sum(axis=1)).ravel()
    sums_u = np.asarray(R.sum(axis=1)).ravel()
    user_means = np.divide(sums_u, counts_u, out=np.zeros_like(sums_u),
                           where=counts_u > 0)
    counts_m = np.asarray((R != 0).sum(axis=0)).ravel()
    sums_m = np.asarray(R.sum(axis=0)).ravel()
    movie_means = np.divide(sums_m, counts_m, out=np.zeros_like(sums_m),
                            where=counts_m > 0)
    global_mean = float(R.data.mean()) if R.nnz else 0.0
    # movies/users unseen in training fall back to the global mean
    movie_means[counts_m == 0] = global_mean
    user_means[counts_u == 0] = global_mean
    return global_mean, user_means, movie_means


def load_dataset(min_user: int = config.MIN_RATINGS_PER_USER,
                 min_movie: int = config.MIN_RATINGS_PER_MOVIE,
                 test_size: float = config.TEST_SIZE,
                 seed: int = config.RANDOM_SEED,
                 verbose: bool = True) -> Dataset:
    ensure_dataset()

    ratings = pd.read_csv(config.RATINGS_CSV)
    movies = pd.read_csv(config.MOVIES_CSV)
    tags = pd.read_csv(config.TAGS_CSV)

    raw_shape = ratings.shape
    ratings = clean_ratings(ratings, movies)
    ratings = filter_sparse(ratings, min_user, min_movie)

    # movie metadata: derive year, blank out the "(no genres listed)" marker
    movies = movies.copy()
    movies["year"] = movies["title"].map(_extract_year)
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    # keep non-empty tags
    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag"] = tags["tag"].astype(str).str.strip()
    tags = tags[tags.tag != ""]

    train, test = train_test_split_per_user(ratings, test_size, seed)
    # only score test rows whose user AND movie are known from training
    known_users = set(train.userId)
    known_movies = set(train.movieId)
    test = test[test.userId.isin(known_users) & test.movieId.isin(known_movies)]
    test = test.reset_index(drop=True)

    user_ids = np.sort(train.userId.unique())
    movie_ids = np.sort(train.movieId.unique())
    R, uid_to_idx, mid_to_idx = _build_matrix(train, user_ids, movie_ids)
    global_mean, user_means, movie_means = _matrix_stats(R)

    ds = Dataset(
        ratings=ratings, train=train, test=test,
        movies=movies, tags=tags,
        user_ids=user_ids, movie_ids=movie_ids,
        uid_to_idx=uid_to_idx, mid_to_idx=mid_to_idx,
        R=R, global_mean=global_mean,
        user_means=user_means, movie_means=movie_means,
    )

    if verbose:
        density = R.nnz / (ds.n_users * ds.n_movies) * 100
        print("Data pipeline")
        print(f"  raw ratings         : {raw_shape[0]:,}")
        print(f"  after clean+filter  : {len(ratings):,}")
        print(f"  users x movies      : {ds.n_users:,} x {ds.n_movies:,} "
              f"(density {density:.2f}%)")
        print(f"  train / test        : {len(train):,} / {len(test):,}")
        print(f"  global mean rating  : {global_mean:.3f}")
    return ds