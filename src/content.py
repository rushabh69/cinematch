"""Content-based movie features from genres + user tags.

Builds one TF-IDF vector per movie by joining its genres with any tags users
have attached to it. These vectors power the hybrid model and the content-based
path of get_similar_movies (which still works for movies that have too few
ratings for collaborative similarity to mean anything).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Dataset


class ContentModel:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.movie_ids = dataset.movie_ids            # aligned with the CF matrix
        self.mid_to_idx = dataset.mid_to_idx
        self.features: sparse.csr_matrix | None = None
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self) -> "ContentModel":
        ds = self.ds
        # one tag "document" per movie (spaces inside a tag are removed so a
        # multi-word tag stays a single token)
        tag_docs = (
            ds.tags.assign(tok=ds.tags.tag.str.lower().str.replace(r"\s+", "", regex=True))
            .groupby("movieId")["tok"].apply(lambda s: " ".join(s))
            .to_dict()
        )
        genre_map = ds.movies.set_index("movieId")["genres"].to_dict()

        corpus = []
        for mid in self.movie_ids:
            genres = str(genre_map.get(mid, "")).replace("|", " ").lower()
            tags = tag_docs.get(mid, "")
            # repeat genres so a shared genre counts a bit more than a shared tag
            corpus.append(f"{genres} {genres} {tags}".strip())

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+", min_df=1)
        self.features = self.vectorizer.fit_transform(corpus)   # n_movies x n_terms
        return self

    def similar_by_index(self, idx: int, n: int = 10):
        """[(movie_idx, cosine)] most similar to a given movie column index."""
        sims = cosine_similarity(self.features[idx], self.features).ravel()
        sims[idx] = -np.inf
        top = np.argpartition(-sims, min(n, len(sims) - 1))[:n]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i])]

    def user_profile(self, liked_indices, weights=None):
        """Weighted average of the content vectors of the movies a user liked."""
        if len(liked_indices) == 0:
            return None
        mat = self.features[liked_indices]
        if weights is None:
            profile = mat.mean(axis=0)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1, 1)
            profile = np.asarray(mat.multiply(w).sum(axis=0)) / max(w.sum(), 1e-9)
        return np.asarray(profile).ravel()

    def score_all(self, profile: np.ndarray) -> np.ndarray:
        """Cosine of every movie against a user profile -> length n_movies."""
        if profile is None:
            return np.zeros(len(self.movie_ids))
        p = profile.reshape(1, -1)
        return cosine_similarity(p, self.features).ravel()