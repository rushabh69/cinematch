from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Dataset


class ContentModel:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.movie_ids = dataset.movie_ids
        self.mid_to_idx = dataset.mid_to_idx

        self.features: sparse.csr_matrix | None = None
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self) -> "ContentModel":
        tags = self.ds.tags.copy()
        tags["tag"] = (
            tags["tag"]
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", "", regex=True)
        )

        tag_docs = (
            tags.groupby("movieId")["tag"]
            .apply(" ".join)
            .to_dict()
        )

        genres = (
            self.ds.movies
            .set_index("movieId")["genres"]
            .to_dict()
        )

        documents = []

        for movie_id in self.movie_ids:
            movie_genres = str(
                genres.get(movie_id, "")
            ).replace("|", " ").lower()

            movie_tags = tag_docs.get(movie_id, "")

            # Give genres a little more weight than individual tags.
            documents.append(
                f"{movie_genres} {movie_genres} {movie_tags}".strip()
            )

        self.vectorizer = TfidfVectorizer(
            token_pattern=r"[^\s]+",
            min_df=1,
        )
        self.features = self.vectorizer.fit_transform(documents)

        return self

    def similar_by_index(self, idx: int, n: int = 10):
        similarities = cosine_similarity(
            self.features[idx],
            self.features,
        ).ravel()

        similarities[idx] = -np.inf

        count = min(n, len(similarities) - 1)
        indices = np.argpartition(
            -similarities,
            count,
        )[:count]

        indices = indices[
            np.argsort(-similarities[indices])
        ]

        return [
            (int(movie_idx), float(similarities[movie_idx]))
            for movie_idx in indices
            if np.isfinite(similarities[movie_idx])
        ]

    def user_profile(self, liked_indices, weights=None):
        if len(liked_indices) == 0:
            return None

        movies = self.features[liked_indices]

        if weights is None:
            profile = movies.mean(axis=0)
        else:
            weights = np.asarray(
                weights,
                dtype=float,
            ).reshape(-1, 1)

            profile = (
                np.asarray(
                    movies.multiply(weights).sum(axis=0)
                )
                / max(weights.sum(), 1e-9)
            )

        return np.asarray(profile).ravel()

    def score_all(self, profile: np.ndarray) -> np.ndarray:
        if profile is None:
            return np.zeros(len(self.movie_ids))

        return cosine_similarity(
            profile.reshape(1, -1),
            self.features,
        ).ravel()