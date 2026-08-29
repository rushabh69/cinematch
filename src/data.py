from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from . import config


def ensure_dataset():
    if config.RATINGS_CSV.exists():
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MovieLens dataset from {config.DATASET_URL} ...")

    with urllib.request.urlopen(config.DATASET_URL) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(config.DATA_DIR)

    print(f"Extracted to {config.DATASET_DIR}")


@dataclass
class Dataset:
    ratings: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    movies: pd.DataFrame
    tags: pd.DataFrame

    user_ids: np.ndarray = field(default=None)
    movie_ids: np.ndarray = field(default=None)

    uid_to_idx: dict = field(default=None)
    mid_to_idx: dict = field(default=None)

    R: sparse.csr_matrix = field(default=None)

    global_mean: float = 0.0
    user_means: np.ndarray = field(default=None)
    movie_means: np.ndarray = field(default=None)

    def has_user(self, user_id: int) -> bool:
        return user_id in self.uid_to_idx

    def has_movie(self, movie_id: int) -> bool:
        return movie_id in self.mid_to_idx

    def title(self, movie_id: int) -> str:
        match = self.movies.loc[
            self.movies.movieId == movie_id, "title"
        ]

        if match.empty:
            return f"movie {movie_id}"

        return match.iloc[0]

    def genres(self, movie_id: int) -> str:
        match = self.movies.loc[
            self.movies.movieId == movie_id, "genres"
        ]

        if match.empty:
            return ""

        return match.iloc[0]

    def seen_movie_indices(self, user_id: int) -> set:
        if user_id not in self.uid_to_idx:
            return set()

        user_idx = self.uid_to_idx[user_id]
        return set(self.R[user_idx].indices)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_movies(self) -> int:
        return len(self.movie_ids)


def _extract_year(title: str):
    match = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    return int(match.group(1)) if match else np.nan


def clean_ratings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    ).copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    low, high = config.RATING_SCALE

    ratings = ratings[
        ratings["rating"].between(low, high)
    ]

    movie_ids = set(movies["movieId"])
    ratings = ratings[ratings["movieId"].isin(movie_ids)]

    return ratings.drop_duplicates(
        subset=["userId", "movieId"],
        keep="last",
    )


def filter_sparse(
    ratings: pd.DataFrame,
    min_user: int = config.MIN_RATINGS_PER_USER,
    min_movie: int = config.MIN_RATINGS_PER_MOVIE,
) -> pd.DataFrame:
    current = ratings

    while True:
        user_counts = current["userId"].value_counts()
        movie_counts = current["movieId"].value_counts()

        users = user_counts[
            user_counts >= min_user
        ].index

        movies = movie_counts[
            movie_counts >= min_movie
        ].index

        filtered = current[
            current["userId"].isin(users)
            & current["movieId"].isin(movies)
        ]

        if len(filtered) == len(current):
            return filtered.reset_index(drop=True)

        current = filtered


def train_test_split_per_user(
    ratings: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    test_indices = []

    for _, group in ratings.groupby("userId"):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)

        test_count = round(len(indices) * test_size)
        test_count = min(test_count, len(indices) - 1)

        test_indices.extend(indices[:test_count])

    test_mask = ratings.index.isin(test_indices)

    train = ratings.loc[~test_mask].reset_index(drop=True)
    test = ratings.loc[test_mask].reset_index(drop=True)

    return train, test


def _build_matrix(train, user_ids, movie_ids):
    uid_to_idx = {
        user_id: i
        for i, user_id in enumerate(user_ids)
    }

    mid_to_idx = {
        movie_id: i
        for i, movie_id in enumerate(movie_ids)
    }

    rows = train["userId"].map(uid_to_idx).to_numpy()
    cols = train["movieId"].map(mid_to_idx).to_numpy()
    values = train["rating"].to_numpy(dtype=np.float64)

    matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )

    return matrix, uid_to_idx, mid_to_idx


def _matrix_stats(matrix):
    user_counts = np.asarray(
        (matrix != 0).sum(axis=1)
    ).ravel()
    user_sums = np.asarray(
        matrix.sum(axis=1)
    ).ravel()

    user_means = np.divide(
        user_sums,
        user_counts,
        out=np.zeros_like(user_sums),
        where=user_counts > 0,
    )

    movie_counts = np.asarray(
        (matrix != 0).sum(axis=0)
    ).ravel()
    movie_sums = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    movie_means = np.divide(
        movie_sums,
        movie_counts,
        out=np.zeros_like(movie_sums),
        where=movie_counts > 0,
    )

    global_mean = (
        float(matrix.data.mean())
        if matrix.nnz
        else 0.0
    )

    user_means[user_counts == 0] = global_mean
    movie_means[movie_counts == 0] = global_mean

    return global_mean, user_means, movie_means


def load_dataset(
    min_user: int = config.MIN_RATINGS_PER_USER,
    min_movie: int = config.MIN_RATINGS_PER_MOVIE,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
    verbose: bool = True,
) -> Dataset:
    ensure_dataset()

    ratings = pd.read_csv(config.RATINGS_CSV)
    movies = pd.read_csv(config.MOVIES_CSV)
    tags = pd.read_csv(config.TAGS_CSV)

    raw_count = len(ratings)

    ratings = clean_ratings(ratings, movies)
    ratings = filter_sparse(
        ratings,
        min_user=min_user,
        min_movie=min_movie,
    )

    movies = movies.copy()
    movies["year"] = movies["title"].map(_extract_year)
    movies["genres"] = movies["genres"].replace(
        "(no genres listed)",
        "",
    )

    tags = tags.dropna(subset=["tag"]).copy()
    tags["tag"] = tags["tag"].astype(str).str.strip()
    tags = tags[tags["tag"] != ""]

    train, test = train_test_split_per_user(
        ratings,
        test_size=test_size,
        seed=seed,
    )

    train_users = set(train["userId"])
    train_movies = set(train["movieId"])

    test = test[
        test["userId"].isin(train_users)
        & test["movieId"].isin(train_movies)
    ].reset_index(drop=True)

    user_ids = np.sort(train["userId"].unique())
    movie_ids = np.sort(train["movieId"].unique())

    matrix, uid_to_idx, mid_to_idx = _build_matrix(
        train,
        user_ids,
        movie_ids,
    )

    global_mean, user_means, movie_means = _matrix_stats(matrix)

    dataset = Dataset(
        ratings=ratings,
        train=train,
        test=test,
        movies=movies,
        tags=tags,
        user_ids=user_ids,
        movie_ids=movie_ids,
        uid_to_idx=uid_to_idx,
        mid_to_idx=mid_to_idx,
        R=matrix,
        global_mean=global_mean,
        user_means=user_means,
        movie_means=movie_means,
    )

    if verbose:
        density = (
            matrix.nnz
            / (dataset.n_users * dataset.n_movies)
            * 100
        )

        print("Data pipeline")
        print(f"  raw ratings       : {raw_count:,}")
        print(f"  after filtering   : {len(ratings):,}")
        print(
            f"  users x movies    : "
            f"{dataset.n_users:,} x {dataset.n_movies:,} "
            f"(density {density:.2f}%)"
        )
        print(
            f"  train / test      : "
            f"{len(train):,} / {len(test):,}"
        )
        print(f"  global mean       : {global_mean:.3f}")

    return dataset
