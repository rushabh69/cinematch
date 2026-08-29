from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .content import ContentModel
from .data import Dataset


def _clip(x):
    lo, hi = config.RATING_SCALE
    return float(np.clip(x, lo, hi))


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)

    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(x)

    return (x - lo) / (hi - lo)


def _keep_topk_per_row(S: np.ndarray, k: int) -> np.ndarray:
    if k >= S.shape[1]:
        return S

    out = np.zeros_like(S)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    out[rows, idx] = S[rows, idx]
    return out


class BaseRecommender:
    name = "base"

    def fit(self, dataset: Dataset):
        self.ds = dataset
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        raise NotImplementedError

    def _score_all(self, user_id: int) -> np.ndarray:
        raise NotImplementedError

    def predict_all_ratings(self, user_id: int) -> np.ndarray:
        lo, hi = config.RATING_SCALE
        return np.clip(self._score_all(user_id), lo, hi)

    def recommend(
        self,
        user_id: int,
        n: int = config.DEFAULT_N_RECOMMENDATIONS,
        exclude_seen: bool = True,
    ):
        scores = self._score_all(user_id)

        if exclude_seen:
            for idx in self.ds.seen_movie_indices(user_id):
                scores[idx] = -np.inf

        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return []

        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]

        return [
            (int(self.ds.movie_ids[idx]), float(scores[idx]))
            for idx in top
            if np.isfinite(scores[idx])
        ]


class PopularityRecommender(BaseRecommender):
    name = "Popularity"

    def fit(self, dataset: Dataset):
        super().fit(dataset)

        R = dataset.R
        counts = np.asarray((R != 0).sum(axis=0)).ravel().astype(float)
        means = dataset.movie_means

        prior = dataset.global_mean
        min_votes = config.POPULARITY_MIN_VOTES

        self.weighted = (
            counts / (counts + min_votes) * means
            + min_votes / (counts + min_votes) * prior
        )
        self.counts = counts

        return self

    def predict(self, user_id, movie_id):
        if movie_id in self.ds.mid_to_idx:
            return _clip(self.weighted[self.ds.mid_to_idx[movie_id]])

        return _clip(self.ds.global_mean)

    def _score_all(self, user_id):
        return self.weighted.copy()


class UserBasedCF(BaseRecommender):
    name = "User-based CF"

    def __init__(self, k: int = config.TOP_K_NEIGHBOURS):
        self.k = k

    def fit(self, dataset: Dataset):
        super().fit(dataset)

        R = dataset.R.tocsr().astype(np.float64)
        self.user_means = dataset.user_means

        centred = R.copy()
        centred.data -= np.repeat(
            self.user_means,
            np.diff(R.indptr),
        )

        self.Rc = centred
        self.mask = (R != 0).astype(np.float64)

        sim = cosine_similarity(centred)
        np.fill_diagonal(sim, 0.0)

        self.sim = _keep_topk_per_row(sim, self.k)
        self.sim_bool = (self.sim != 0).astype(np.float64)

        return self

    def _predict_row(self, u_idx: int) -> np.ndarray:
        similarity = self.sim[u_idx]

        num = similarity @ self.Rc
        den = np.abs(similarity) @ self.mask

        out = np.full(self.ds.n_movies, self.user_means[u_idx])
        den = np.asarray(den).ravel()
        num = np.asarray(num).ravel()

        valid = den > 1e-9
        out[valid] = self.user_means[u_idx] + num[valid] / den[valid]

        return out

    def _support_row(self, u_idx: int) -> np.ndarray:
        return np.asarray(self.sim_bool[u_idx] @ self.mask).ravel()

    def predict(self, user_id, movie_id):
        if user_id not in self.ds.uid_to_idx:
            return _clip(self.ds.global_mean)

        if movie_id not in self.ds.mid_to_idx:
            return _clip(self.user_means[self.ds.uid_to_idx[user_id]])

        u_idx = self.ds.uid_to_idx[user_id]
        movie_idx = self.ds.mid_to_idx[movie_id]

        return _clip(self._predict_row(u_idx)[movie_idx])

    def predict_all_ratings(self, user_id):
        lo, hi = config.RATING_SCALE

        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, self.ds.global_mean)

        predictions = self._predict_row(self.ds.uid_to_idx[user_id])
        return np.clip(predictions, lo, hi)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)

        u_idx = self.ds.uid_to_idx[user_id]
        predictions = self._predict_row(u_idx)

        support = self._support_row(u_idx)
        shrink = support / (support + config.RANK_SHRINKAGE_BETA)

        base = self.user_means[u_idx]
        return base + (predictions - base) * shrink


class ItemBasedCF(BaseRecommender):
    name = "Item-based CF"

    def __init__(self, k: int = config.TOP_K_NEIGHBOURS):
        self.k = k

    def fit(self, dataset: Dataset):
        super().fit(dataset)

        R = dataset.R.tocsr().astype(np.float64)

        centred = R.copy()
        centred.data -= np.repeat(
            dataset.user_means,
            np.diff(R.indptr),
        )

        sim = cosine_similarity(centred.T)
        np.fill_diagonal(sim, 0.0)

        self.sim = sparse.csr_matrix(
            _keep_topk_per_row(sim, self.k).astype(np.float32)
        )
        self.sim_abs = abs(self.sim)

        sim_bool = self.sim.copy()
        sim_bool.data = np.ones_like(sim_bool.data)
        self.sim_bool = sim_bool

        self.R = R
        self.mask = (R != 0).astype(np.float64)

        return self

    def _predict_row(self, u_idx: int):
        ratings = self.R[u_idx].toarray().ravel()
        rated = self.mask[u_idx].toarray().ravel()

        num = self.sim @ ratings
        den = self.sim_abs @ rated

        out = self.ds.movie_means.copy()
        valid = den > 1e-9
        out[valid] = num[valid] / den[valid]

        return out, rated

    def predict(self, user_id, movie_id):
        if user_id not in self.ds.uid_to_idx:
            return _clip(self.ds.global_mean)

        if movie_id not in self.ds.mid_to_idx:
            return _clip(self.ds.global_mean)

        u_idx = self.ds.uid_to_idx[user_id]
        movie_idx = self.ds.mid_to_idx[movie_id]

        return _clip(self._predict_row(u_idx)[0][movie_idx])

    def predict_all_ratings(self, user_id):
        lo, hi = config.RATING_SCALE

        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, self.ds.global_mean)

        predictions = self._predict_row(self.ds.uid_to_idx[user_id])[0]
        return np.clip(predictions, lo, hi)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)

        u_idx = self.ds.uid_to_idx[user_id]
        predictions, rated = self._predict_row(u_idx)

        support = self.sim_bool @ rated
        shrink = support / (support + config.RANK_SHRINKAGE_BETA)

        base = self.ds.user_means[u_idx]
        return base + (predictions - base) * shrink

    def similar_items(self, movie_id: int, n: int = 10):
        if movie_id not in self.ds.mid_to_idx:
            return []

        idx = self.ds.mid_to_idx[movie_id]
        sims = np.asarray(self.sim.getrow(idx).todense()).ravel()
        order = np.argsort(-sims)

        result = []

        for other_idx in order:
            if other_idx == idx or sims[other_idx] <= 0:
                continue

            result.append(
                (
                    int(self.ds.movie_ids[other_idx]),
                    float(sims[other_idx]),
                )
            )

            if len(result) >= n:
                break

        return result


class MatrixFactorizationCF(BaseRecommender):
    name = "SVD (Matrix Factorisation)"

    def __init__(
        self,
        factors=config.MF_FACTORS,
        epochs=config.MF_EPOCHS,
        lr=config.MF_LR,
        reg=config.MF_REG,
        seed=config.RANDOM_SEED,
    ):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.backend = None

    def fit(self, dataset: Dataset):
        super().fit(dataset)
        self.mu = dataset.global_mean

        try:
            self._fit_surprise(dataset)
            self.backend = "surprise"
        except Exception:
            self._fit_numpy(dataset)
            self.backend = "numpy-sgd"

        return self

    def _fit_surprise(self, dataset: Dataset):
        from surprise import Dataset as SurpriseDataset
        from surprise import Reader, SVD

        reader = Reader(rating_scale=config.RATING_SCALE)
        data = SurpriseDataset.load_from_df(
            dataset.train[["userId", "movieId", "rating"]],
            reader,
        )
        trainset = data.build_full_trainset()

        algo = SVD(
            n_factors=self.factors,
            n_epochs=self.epochs,
            lr_all=self.lr,
            reg_all=self.reg,
            random_state=self.seed,
        )
        algo.fit(trainset)

        n_users = dataset.n_users
        n_movies = dataset.n_movies
        factors = self.factors

        self.P = np.zeros((n_users, factors))
        self.Q = np.zeros((n_movies, factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_movies)

        for uid, user_idx in dataset.uid_to_idx.items():
            try:
                inner_uid = algo.trainset.to_inner_uid(uid)
                self.P[user_idx] = algo.pu[inner_uid]
                self.bu[user_idx] = algo.bu[inner_uid]
            except ValueError:
                pass

        for movie_id, movie_idx in dataset.mid_to_idx.items():
            try:
                inner_iid = algo.trainset.to_inner_iid(movie_id)
                self.Q[movie_idx] = algo.qi[inner_iid]
                self.bi[movie_idx] = algo.bi[inner_iid]
            except ValueError:
                pass

    def _fit_numpy(self, dataset: Dataset):
        rng = np.random.default_rng(self.seed)

        n_users = dataset.n_users
        n_movies = dataset.n_movies
        factors = self.factors

        P = rng.normal(0, 0.1, (n_users, factors))
        Q = rng.normal(0, 0.1, (n_movies, factors))
        bu = np.zeros(n_users)
        bi = np.zeros(n_movies)

        train = dataset.train
        user_idx = train.userId.map(dataset.uid_to_idx).to_numpy()
        movie_idx = train.movieId.map(dataset.mid_to_idx).to_numpy()
        ratings = train.rating.to_numpy(dtype=np.float64)

        for _ in range(self.epochs):
            order = rng.permutation(len(ratings))

            for row_idx in order:
                u = user_idx[row_idx]
                i = movie_idx[row_idx]
                rating = ratings[row_idx]

                prediction = (
                    self.mu
                    + bu[u]
                    + bi[i]
                    + P[u] @ Q[i]
                )

                error = rating - prediction

                bu[u] += self.lr * (error - self.reg * bu[u])
                bi[i] += self.lr * (error - self.reg * bi[i])

                old_p = P[u].copy()

                P[u] += self.lr * (
                    error * Q[i] - self.reg * P[u]
                )
                Q[i] += self.lr * (
                    error * old_p - self.reg * Q[i]
                )

        self.P = P
        self.Q = Q
        self.bu = bu
        self.bi = bi

    def predict(self, user_id, movie_id):
        if (
            user_id not in self.ds.uid_to_idx
            or movie_id not in self.ds.mid_to_idx
        ):
            return _clip(self.mu)

        user_idx = self.ds.uid_to_idx[user_id]
        movie_idx = self.ds.mid_to_idx[movie_id]

        prediction = (
            self.mu
            + self.bu[user_idx]
            + self.bi[movie_idx]
            + self.P[user_idx] @ self.Q[movie_idx]
        )

        return _clip(prediction)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)

        user_idx = self.ds.uid_to_idx[user_id]

        return (
            self.mu
            + self.bu[user_idx]
            + self.bi
            + self.Q @ self.P[user_idx]
        )

    def item_factors(self):
        return self.Q


class HybridRecommender(BaseRecommender):
    name = "Hybrid (CF + Content)"

    def __init__(
        self,
        cf_model: BaseRecommender | None = None,
        cf_weight: float = config.HYBRID_CF_WEIGHT,
    ):
        self.cf = cf_model
        self.cf_weight = cf_weight

    def fit(self, dataset: Dataset):
        super().fit(dataset)

        if self.cf is None:
            self.cf = UserBasedCF().fit(dataset)
        elif not hasattr(self.cf, "ds"):
            self.cf.fit(dataset)

        self.content = ContentModel(dataset).fit()
        return self

    def _content_scores(self, user_id: int) -> np.ndarray:
        seen = self.ds.seen_movie_indices(user_id)

        if not seen:
            return np.zeros(self.ds.n_movies)

        user_idx = self.ds.uid_to_idx[user_id]
        row = self.ds.R[user_idx]

        liked_idx = []
        weights = []

        for movie_idx, rating in zip(row.indices, row.data):
            if rating >= config.LIKE_THRESHOLD:
                liked_idx.append(movie_idx)
                weights.append(
                    rating - config.LIKE_THRESHOLD + 0.5
                )

        if not liked_idx:
            liked_idx = list(row.indices)
            weights = list(row.data)

        profile = self.content.user_profile(
            liked_idx,
            weights,
        )

        return self.content.score_all(profile)

    def predict(self, user_id, movie_id):
        return self.cf.predict(user_id, movie_id)

    def predict_all_ratings(self, user_id):
        return self.cf.predict_all_ratings(user_id)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)

        cf_scores = _minmax(self.cf._score_all(user_id))
        content_scores = _minmax(self._content_scores(user_id))

        return (
            self.cf_weight * cf_scores
            + (1 - self.cf_weight) * content_scores
        )


class ImplicitCF(BaseRecommender):
    name = "Implicit CF"

    def __init__(
        self,
        k: int = config.TOP_K_NEIGHBOURS,
        alpha: float = 1.0,
    ):
        self.k = k
        self.alpha = alpha

    def fit(self, dataset: Dataset):
        super().fit(dataset)

        R = dataset.R.tocsr()

        conf = R.copy().astype(np.float64)
        conf.data = 1.0 + self.alpha * conf.data

        self.conf = conf

        sim = cosine_similarity(conf.T)
        np.fill_diagonal(sim, 0.0)

        self.sim = sparse.csr_matrix(
            _keep_topk_per_row(sim, self.k).astype(np.float32)
        )

        return self

    def predict(self, user_id, movie_id):
        if movie_id in self.ds.mid_to_idx:
            idx = self.ds.mid_to_idx[movie_id]
            return _clip(self.ds.movie_means[idx])

        return _clip(self.ds.global_mean)

    def predict_all_ratings(self, user_id):
        lo, hi = config.RATING_SCALE
        return np.clip(self.ds.movie_means.copy(), lo, hi)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)

            user_idx = self.ds.uid_to_idx[user_id]
        interacted = (
            self.conf[user_idx].toarray().ravel() > 0
        ).astype(np.float64)

        return self.sim @ interacted


def build_all_models():
    return {
        "Popularity": PopularityRecommender(),
        "User-based CF": UserBasedCF(),
        "Item-based CF": ItemBasedCF(),
    "SVD (Matrix Factorisation)": MatrixFactorizationCF(),
        "Hybrid (CF + Content)": HybridRecommender(),
    }
