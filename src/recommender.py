"""High-level recommender facade used by the CLI and the Streamlit app.

Wraps the individual models behind a small, friendly API:

    rec = RecommenderSystem.build()          # load data + fit models (cached)
    rec.get_recommendations(user_id, n=10)   # personalised top-N (+ explanations)
    rec.get_similar_movies(movie_id, n=10)   # "movies similar to this"
    rec.recommend_for_new_user([...])        # cold-start fold-in
    rec.search_movies("matrix")              # title search
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import config
from .data import Dataset, load_dataset
from .content import ContentModel
from . import models as M


# hybrid gives the best top-N ranking in our eval; SVD is the best pure rating
# predictor. the app/CLI default to the best top-N experience.
DEFAULT_MODEL = "Hybrid (CF + Content)"


class RecommenderSystem:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.models: dict[str, M.BaseRecommender] = {}
        self.content: ContentModel | None = None
        self.metrics: pd.DataFrame | None = None
        # per-movie rating count, handy for the UI + confidence
        R = dataset.R
        self._movie_counts = np.asarray((R != 0).sum(axis=0)).ravel()

    @classmethod
    def build(cls, dataset: Dataset | None = None, which: list[str] | None = None,
              verbose: bool = True) -> "RecommenderSystem":
        ds = dataset or load_dataset(verbose=verbose)
        self = cls(ds)
        registry = M.build_all_models()
        registry["Implicit CF"] = M.ImplicitCF()          # bonus model available too
        if which is not None:
            registry = {k: v for k, v in registry.items() if k in which}
        for name, model in registry.items():
            if verbose:
                print(f"  fitting {name} ...", flush=True)
            model.fit(ds)
            self.models[name] = model
        self.content = ContentModel(ds).fit()
        return self

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
        cnt = int(self._movie_counts[idx]) if idx is not None else 0
        avg = float(self.ds.movie_means[idx]) if idx is not None else np.nan
        return {
            "movieId": int(movie_id),
            "title": self.ds.title(movie_id),
            "genres": self.ds.genres(movie_id),
            "n_ratings": cnt,
            "avg_rating": round(avg, 2) if idx is not None else np.nan,
        }

    def model_names(self):
        return list(self.models.keys())

    def known_user_ids(self):
        return list(self.ds.user_ids)

    def is_cold_start(self, user_id: int) -> bool:
        return (user_id not in self.ds.uid_to_idx) or \
               (len(self.ds.seen_movie_indices(user_id)) == 0)

    def search_movies(self, query: str, n: int = 25) -> pd.DataFrame:
        q = str(query).strip().lower()
        if not q:
            return pd.DataFrame()
        m = self.ds.movies
        hit = m[m.title.str.lower().str.contains(q, regex=False)]
        # only surface movies the models actually know (present in training)
        hit = hit[hit.movieId.isin(self.ds.mid_to_idx)]
        rows = [self._movie_row(mid) for mid in hit.movieId.tolist()[:n]]
        df = pd.DataFrame(rows)
        return df.sort_values("n_ratings", ascending=False) if len(df) else df

    def user_top_rated(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.ds.uid_to_idx:
            return pd.DataFrame()
        u = self.ds.uid_to_idx[user_id]
        row = self.ds.R[u]
        pairs = sorted(zip(row.indices, row.data), key=lambda t: -t[1])[:n]
        rows = []
        for j, r in pairs:
            d = self._movie_row(int(self.ds.movie_ids[j]))
            d["your_rating"] = float(r)
            rows.append(d)
        return pd.DataFrame(rows)

    def popular_movies(self, n: int = 10) -> pd.DataFrame:
        pop = self.models.get("Popularity") or M.PopularityRecommender().fit(self.ds)
        recs = pop.recommend(user_id=-1, n=n, exclude_seen=False)
        rows = []
        for mid, score in recs:
            d = self._movie_row(mid)
            d["score"] = round(score, 3)
            rows.append(d)
        return pd.DataFrame(rows)

    def get_recommendations(self, user_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS,model: str = DEFAULT_MODEL,explain: bool = True) -> pd.DataFrame:

        """Top-N movies for a user.

        Cold start: an unknown user (or one with no ratings) falls back to
        popularity, flagged via the cold_start column so the UI can say so.
        """
        if self.is_cold_start(user_id):
            df = self.popular_movies(n)
            df["cold_start"] = True
            if explain:
                df["why"] = "Popular pick - no rating history yet (cold start)"
            return df

        mdl = self.models.get(model) or self.models[DEFAULT_MODEL]
        recs = mdl.recommend(user_id, n=n, exclude_seen=True)
        rows = []
        for mid, score in recs:
            d = self._movie_row(mid)
            d["score"] = round(float(score), 3)
            d["cold_start"] = False
            if explain:
                d["why"] = self.explain(user_id, mid)
            rows.append(d)
        return pd.DataFrame(rows)

    def explain(self, user_id: int, movie_id: int, top: int = 3) -> str:
        """'Because you liked X, Y, Z' using content similarity."""
        if user_id not in self.ds.uid_to_idx or movie_id not in self.ds.mid_to_idx:
            return ""
        target = self.ds.mid_to_idx[movie_id]
        u = self.ds.uid_to_idx[user_id]
        row = self.ds.R[u]
        liked = [(j, r) for j, r in zip(row.indices, row.data)
                 if r >= config.LIKE_THRESHOLD]
        if not liked:
            liked = list(zip(row.indices, row.data))
        if not liked:
            return ""
        idxs = [j for j, _ in liked]
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.content.features[target],
                                 self.content.features[idxs]).ravel()
        order = np.argsort(-sims)[:top]
        names = [self.ds.title(int(self.ds.movie_ids[idxs[o]]))
                 for o in order if sims[o] > 0]
        if not names:
            return "Recommended from users with similar taste"
        return "Because you liked " + ", ".join(names)

    def get_similar_movies(self, movie_id: int, n: int = 10,
                           method: str = "hybrid") -> pd.DataFrame:
        """Movies similar to movie_id.

        method='content'  -> genres + tags TF-IDF cosine (works for any movie)
        method='cf'       -> item-based collaborative similarity (co-rating)
        method='hybrid'   -> blend of both (default)
        """
        if movie_id not in self.ds.mid_to_idx:
            return pd.DataFrame()
        idx = self.ds.mid_to_idx[movie_id]

        content_sim = None
        if method in ("content", "hybrid"):
            from sklearn.metrics.pairwise import cosine_similarity
            content_sim = cosine_similarity(
                self.content.features[idx], self.content.features).ravel()

        cf_sim = None
        item_cf = self.models.get("Item-based CF")
        if method in ("cf", "hybrid") and item_cf is not None:
            cf_sim = np.asarray(item_cf.sim.getrow(idx).todense()).ravel().astype(float)

        # combine BEFORE excluding self (so min-max normalisation stays valid)
        if method == "content" or (method == "hybrid" and cf_sim is None):
            score = None if content_sim is None else content_sim.copy()
        elif method == "cf" or (method == "hybrid" and content_sim is None):
            score = None if cf_sim is None else cf_sim.copy()
        else:  # hybrid: blend normalised content + collaborative similarity
            score = 0.5 * M._minmax(content_sim) + 0.5 * M._minmax(cf_sim)
        if score is None:
            return pd.DataFrame()
        score[idx] = -np.inf                    # never return the movie itself

        order = np.argsort(-score)[:n]
        rows = []
        for j in order:
            if not np.isfinite(score[j]) or score[j] <= 0:
                continue
            d = self._movie_row(int(self.ds.movie_ids[j]))
            d["similarity"] = round(float(score[j]), 3)
            rows.append(d)
        return pd.DataFrame(rows)

    def recommend_for_new_user(self, ratings: list[tuple[int, float]],
                               n: int = 10) -> pd.DataFrame:
        """Recommend for a NEW user given a few (movie_id, rating) pairs.

        Combines content similarity to the movies they liked with item-based
        collaborative similarity ("folding in" the new user), so we never need
        to retrain. If we know nothing usable, fall back to popularity.
        """
        valid = [(m, r) for m, r in ratings if m in self.ds.mid_to_idx]
        if not valid:
            df = self.popular_movies(n)
            df["cold_start"] = True
            return df

        liked_idx = [self.ds.mid_to_idx[m] for m, r in valid if r >= config.LIKE_THRESHOLD]
        weights = [r for _, r in valid if r >= config.LIKE_THRESHOLD]
        if not liked_idx:
            liked_idx = [self.ds.mid_to_idx[m] for m, _ in valid]
            weights = [r for _, r in valid]

        profile = self.content.user_profile(liked_idx, weights)
        content_score = M._minmax(self.content.score_all(profile))

        cf_score = np.zeros(self.ds.n_movies)
        item_cf = self.models.get("Item-based CF")
        if item_cf is not None:
            r_vec = np.zeros(self.ds.n_movies)
            for m, r in valid:
                r_vec[self.ds.mid_to_idx[m]] = r
            num = item_cf.sim @ r_vec
            cf_score = M._minmax(num)

        score = 0.5 * content_score + 0.5 * cf_score
        for m, _ in valid:                          # don't recommend rated movies
            score[self.ds.mid_to_idx[m]] = -np.inf

        order = np.argsort(-score)[:n]
        rows = []
        for j in order:
            if not np.isfinite(score[j]):
                continue
            d = self._movie_row(int(self.ds.movie_ids[j]))
            d["score"] = round(float(score[j]), 3)
            d["cold_start"] = True
            rows.append(d)
        return pd.DataFrame(rows)