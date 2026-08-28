from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from . import models as M
from .content import ContentModel
from .data import Dataset, load_dataset


DEFAULT_MODEL = "Hybrid (CF + Content)"


class RecommenderSystem:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.models: dict[str, M.BaseRecommender] = {}
        self.content: ContentModel | None = None
        self.metrics: pd.DataFrame | None = None

        self._movie_counts = np.asarray(
            (dataset.R != 0).sum(axis=0)
        ).ravel()

    @classmethod
    def build(
        cls,
        dataset: Dataset | None = None,
        which: list[str] | None = None,
        verbose: bool = True,
    ) -> "RecommenderSystem":
        ds = dataset or load_dataset(verbose=verbose)
        rec = cls(ds)

        model_list = M.build_all_models()
        model_list["Implicit CF"] = M.ImplicitCF()

        if which is not None:
            model_list = {
                name: model
                for name, model in model_list.items()
                if name in which
            }

        for name, model in model_list.items():
            if verbose:
                print(f"  fitting {name} ...", flush=True)

            model.fit(ds)
            rec.models[name] = model

        rec.content = ContentModel(ds).fit()
        return rec

    def save(self, path=None):
        path = path or (config.ARTIFACTS_DIR / "recommender.pkl")
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        return path

    @staticmethod
    def load(path=None) -> "RecommenderSystem":
        path = path or (config.ARTIFACTS_DIR / "recommender.pkl")

        with open(path, "rb") as f:
            return pickle.load(f)

    def _movie_row(self, movie_id: int) -> dict:
        idx = self.ds.mid_to_idx.get(movie_id)

        if idx is None:
            count = 0
            average = np.nan
        else:
            count = int(self._movie_counts[idx])
            average = float(self.ds.movie_means[idx])

        return {
            "movieId": int(movie_id),
            "title": self.ds.title(movie_id),
            "genres": self.ds.genres(movie_id),
            "n_ratings": count,
            "avg_rating": round(average, 2) if idx is not None else np.nan,
        }

    def model_names(self):
        return list(self.models)

    def known_user_ids(self):
        return list(self.ds.user_ids)

    def is_cold_start(self, user_id: int) -> bool:
        if user_id not in self.ds.uid_to_idx:
            return True

        return not self.ds.seen_movie_indices(user_id)

    def search_movies(self, query: str, n: int = 25) -> pd.DataFrame:
        query = str(query).strip().lower()

        if not query:
            return pd.DataFrame()

        movies = self.ds.movies
        matches = movies[
            movies["title"].str.lower().str.contains(
                query,
                regex=False,
            )
        ]

        matches = matches[
            matches.movieId.isin(self.ds.mid_to_idx)
        ]

        rows = [
            self._movie_row(movie_id)
            for movie_id in matches.movieId.tolist()[:n]
        ]

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(
            "n_ratings",
            ascending=False,
        )

    def user_top_rated(
        self,
        user_id: int,
        n: int = 10,
    ) -> pd.DataFrame:
        if user_id not in self.ds.uid_to_idx:
            return pd.DataFrame()

        user_idx = self.ds.uid_to_idx[user_id]
        row = self.ds.R[user_idx]

        rated = sorted(
            zip(row.indices, row.data),
            key=lambda item: item[1],
            reverse=True,
        )[:n]

        rows = []

        for movie_idx, rating in rated:
            movie_id = int(self.ds.movie_ids[movie_idx])
            movie = self._movie_row(movie_id)
            movie["your_rating"] = float(rating)
            rows.append(movie)

        return pd.DataFrame(rows)

    def popular_movies(self, n: int = 10) -> pd.DataFrame:
        model = self.models.get("Popularity")

        if model is None:
            model = M.PopularityRecommender().fit(self.ds)

        recommendations = model.recommend(
            user_id=-1,
            n=n,
            exclude_seen=False,
        )

        rows = []

        for movie_id, score in recommendations:
            movie = self._movie_row(movie_id)
            movie["score"] = round(score, 3)
            rows.append(movie)

        return pd.DataFrame(rows)

    def get_recommendations(
        self,
        user_id: int,
        n: int = config.DEFAULT_N_RECOMMENDATIONS,
        model: str = DEFAULT_MODEL,
        explain: bool = True,
    ) -> pd.DataFrame:

        if self.is_cold_start(user_id):
            result = self.popular_movies(n)
            result["cold_start"] = True

            if explain:
                result["why"] = (
                    "Popular pick - no rating history yet (cold start)"
                )

            return result

        recommender = self.models.get(model)

        if recommender is None:
            recommender = self.models[DEFAULT_MODEL]

        recommendations = recommender.recommend(
            user_id,
            n=n,
            exclude_seen=True,
        )

        rows = []

        for movie_id, score in recommendations:
            movie = self._movie_row(movie_id)
            movie["score"] = round(float(score), 3)
            movie["cold_start"] = False

            if explain:
                movie["why"] = self.explain(user_id, movie_id)

            rows.append(movie)

        return pd.DataFrame(rows)

    def explain(
        self,
        user_id: int,
        movie_id: int,
        top: int = 3,
    ) -> str:
        if (
            user_id not in self.ds.uid_to_idx
            or movie_id not in self.ds.mid_to_idx
        ):
            return ""

        target_idx = self.ds.mid_to_idx[movie_id]
        user_idx = self.ds.uid_to_idx[user_id]
        row = self.ds.R[user_idx]

        liked = [
            (movie_idx, rating)
            for movie_idx, rating
            in zip(row.indices, row.data)
            if rating >= config.LIKE_THRESHOLD
        ]

        if not liked:
            liked = list(zip(row.indices, row.data))

        if not liked:
            return ""

        liked_indices = [movie_idx for movie_idx, _ in liked]

        similarities = cosine_similarity(
            self.content.features[target_idx],
            self.content.features[liked_indices],
        ).ravel()

        order = np.argsort(-similarities)[:top]

        titles = [
            self.ds.title(
                int(self.ds.movie_ids[liked_indices[pos]])
            )
            for pos in order
            if similarities[pos] > 0
        ]

        if not titles:
            return "Recommended from users with similar taste"

        return "Because you liked " + ", ".join(titles)

    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10,
        method: str = "hybrid",
    ) -> pd.DataFrame:
        if movie_id not in self.ds.mid_to_idx:
            return pd.DataFrame()

        movie_idx = self.ds.mid_to_idx[movie_id]

        content_scores = None
        cf_scores = None

        if method in ("content", "hybrid"):
            content_scores = cosine_similarity(
                self.content.features[movie_idx],
                self.content.features,
            ).ravel()

        item_model = self.models.get("Item-based CF")

        if method in ("cf", "hybrid") and item_model is not None:
            cf_scores = np.asarray(
                item_model.sim.getrow(movie_idx).todense()
            ).ravel().astype(float)

        if method == "content":
            scores = content_scores

        elif method == "cf":
            scores = cf_scores

        elif content_scores is not None and cf_scores is not None:
            scores = (
                0.5 * M._minmax(content_scores)
                + 0.5 * M._minmax(cf_scores)
            )

        elif content_scores is not None:
            scores = content_scores

        else:
            scores = cf_scores

        if scores is None:
            return pd.DataFrame()

        scores = scores.copy()
        scores[movie_idx] = -np.inf

        order = np.argsort(-scores)[:n]

        rows = []

        for idx in order:
            score = scores[idx]

            if not np.isfinite(score) or score <= 0:
                continue

            movie = self._movie_row(
                int(self.ds.movie_ids[idx])
            )
            movie["similarity"] = round(float(score), 3)
            rows.append(movie)

        return pd.DataFrame(rows)

    def recommend_for_new_user(
        self,
        ratings: list[tuple[int, float]],
        n: int = 10,
    ) -> pd.DataFrame:
        valid = [
            (movie_id, rating)
            for movie_id, rating in ratings
            if movie_id in self.ds.mid_to_idx
        ]

        if not valid:
            result = self.popular_movies(n)
            result["cold_start"] = True
            return result

        liked = [
            (movie_id, rating)
            for movie_id, rating in valid
            if rating >= config.LIKE_THRESHOLD
        ]

        if liked:
            liked_indices = [
                self.ds.mid_to_idx[movie_id]
                for movie_id, _ in liked
            ]
            weights = [rating for _, rating in liked]
        else:
            liked_indices = [
                self.ds.mid_to_idx[movie_id]
                for movie_id, _ in valid
            ]
            weights = [rating for _, rating in valid]

        profile = self.content.user_profile(
            liked_indices,
            weights,
        )
        content_scores = M._minmax(
            self.content.score_all(profile)
        )

        cf_scores = np.zeros(self.ds.n_movies)

        item_model = self.models.get("Item-based CF")

        if item_model is not None:
            rating_vector = np.zeros(self.ds.n_movies)

            for movie_id, rating in valid:
                idx = self.ds.mid_to_idx[movie_id]
                rating_vector[idx] = rating

            cf_scores = M._minmax(
                item_model.sim @ rating_vector
            )

        scores = 0.5 * content_scores + 0.5 * cf_scores

        for movie_id, _ in valid:
            scores[self.ds.mid_to_idx[movie_id]] = -np.inf

        order = np.argsort(-scores)[:n]

        rows = []

        for idx in order:
            if not np.isfinite(scores[idx]):
                continue

            movie = self._movie_row(
                int(self.ds.movie_ids[idx])
            )
            movie["score"] = round(float(scores[idx]), 3)
            movie["cold_start"] = True
            rows.append(movie)

        return pd.DataFrame(rows)