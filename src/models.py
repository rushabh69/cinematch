"""Recommendation models.

Every model shares the BaseRecommender interface:

    model.fit(dataset)
    model.predict(user_id, movie_id) -> float            # for RMSE / MAE
    model.recommend(user_id, n) -> [(movie_id, score)]   # for top-N ranking

Implemented:
  * PopularityRecommender   - non-personalised baseline + cold-start fallback
  * UserBasedCF             - cosine similarity between users
  * ItemBasedCF             - adjusted-cosine similarity between movies
  * MatrixFactorizationCF   - SVD via surprise (NumPy SGD fallback)
  * HybridRecommender       - collaborative score blended with content
  * ImplicitCF              - item co-occurrence on implicit feedback (bonus)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data import Dataset
from .content import ContentModel


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
    """Zero out all but the k largest entries in each row of a similarity matrix."""
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
        """A ranking score for every movie column (length n_movies)."""
        raise NotImplementedError

    def predict_all_ratings(self, user_id: int) -> np.ndarray:
        """Rating-scale prediction for every movie (used for RMSE/MAE).

        Default: clip the ranking scores to the rating scale. Models whose
        ranking score isn't on the rating scale (hybrid, implicit) override this.
        """
        lo, hi = config.RATING_SCALE
        return np.clip(self._score_all(user_id), lo, hi)

    def recommend(self, user_id: int, n: int = config.DEFAULT_N_RECOMMENDATIONS,
                  exclude_seen: bool = True):
        scores = self._score_all(user_id)
        if exclude_seen:
            for j in self.ds.seen_movie_indices(user_id):
                scores[j] = -np.inf
        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ds.movie_ids[j]), float(scores[j]))
                for j in top if np.isfinite(scores[j])]


class PopularityRecommender(BaseRecommender):
    """IMDB-style weighted rating so one 5.0 vote can't top the chart."""

    name = "Popularity"

    def fit(self, dataset: Dataset):
        super().fit(dataset)
        R = dataset.R
        counts = np.asarray((R != 0).sum(axis=0)).ravel().astype(float)
        means = dataset.movie_means
        C = dataset.global_mean
        m = config.POPULARITY_MIN_VOTES
        self.weighted = (counts / (counts + m)) * means + (m / (counts + m)) * C
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
        # mean-centre each user's observed ratings (keeps the sparsity)
        Rc = R.copy()
        Rc.data = Rc.data - np.repeat(self.user_means, np.diff(R.indptr))
        self.Rc = Rc
        self.mask = (R != 0).astype(np.float64)       # 1 where rated
        # user-user cosine on the centred vectors
        sim = cosine_similarity(Rc)
        np.fill_diagonal(sim, 0.0)
        self.sim = _keep_topk_per_row(sim, self.k)
        self.sim_bool = (self.sim != 0).astype(np.float64)   # for support counts
        return self

    def _predict_row(self, u_idx: int) -> np.ndarray:
        s = self.sim[u_idx]                           # (n_users,)
        num = s @ self.Rc                             # (n_movies,)
        den = np.abs(s) @ self.mask                   # (n_movies,)
        out = np.full(self.ds.n_movies, self.user_means[u_idx])
        nz = np.asarray(den).ravel() > 1e-9
        out[nz] = self.user_means[u_idx] + np.asarray(num).ravel()[nz] / np.asarray(den).ravel()[nz]
        return out

    def _support_row(self, u_idx: int) -> np.ndarray:
        # how many neighbours actually rated each item
        return np.asarray(self.sim_bool[u_idx] @ self.mask).ravel()

    def predict(self, user_id, movie_id):
        if user_id not in self.ds.uid_to_idx:
            return _clip(self.ds.global_mean)
        if movie_id not in self.ds.mid_to_idx:
            return _clip(self.user_means[self.ds.uid_to_idx[user_id]])
        u = self.ds.uid_to_idx[user_id]
        j = self.ds.mid_to_idx[movie_id]
        return _clip(self._predict_row(u)[j])

    def predict_all_ratings(self, user_id):
        lo, hi = config.RATING_SCALE
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, self.ds.global_mean)
        return np.clip(self._predict_row(self.ds.uid_to_idx[user_id]), lo, hi)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)
        u = self.ds.uid_to_idx[user_id]
        pred = self._predict_row(u)
        support = self._support_row(u)
        shrink = support / (support + config.RANK_SHRINKAGE_BETA)
        return self.user_means[u] + (pred - self.user_means[u]) * shrink


class ItemBasedCF(BaseRecommender):
    name = "Item-based CF"

    def __init__(self, k: int = config.TOP_K_NEIGHBOURS):
        self.k = k

    def fit(self, dataset: Dataset):
        super().fit(dataset)
        R = dataset.R.tocsr().astype(np.float64)
        # adjusted cosine: subtract each USER's mean before comparing items
        Rc = R.copy()
        Rc.data = Rc.data - np.repeat(dataset.user_means, np.diff(R.indptr))
        sim = cosine_similarity(Rc.T)                 # item x item
        np.fill_diagonal(sim, 0.0)
        # top-k pruned -> mostly zeros -> store sparse (big memory win)
        pruned = _keep_topk_per_row(sim, self.k).astype(np.float32)
        self.sim = sparse.csr_matrix(pruned)
        self.sim_abs = abs(self.sim)
        sb = self.sim.copy(); sb.data = np.ones_like(sb.data)
        self.sim_bool = sb                            # for support counts
        self.R = R                                    # raw ratings for weighting
        self.mask = (R != 0).astype(np.float64)
        return self

    def _predict_row(self, u_idx: int):
        r_u = self.R[u_idx].toarray().ravel()         # raw ratings, 0 where unrated
        mask_u = self.mask[u_idx].toarray().ravel()
        num = self.sim @ r_u                          # (n_movies,)
        den = self.sim_abs @ mask_u
        out = self.ds.movie_means.copy()
        nz = den > 1e-9
        out[nz] = num[nz] / den[nz]
        return out, mask_u

    def predict(self, user_id, movie_id):
        if user_id not in self.ds.uid_to_idx:
            return _clip(self.ds.global_mean)
        if movie_id not in self.ds.mid_to_idx:
            return _clip(self.ds.global_mean)
        u = self.ds.uid_to_idx[user_id]
        j = self.ds.mid_to_idx[movie_id]
        return _clip(self._predict_row(u)[0][j])

    def predict_all_ratings(self, user_id):
        lo, hi = config.RATING_SCALE
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, self.ds.global_mean)
        return np.clip(self._predict_row(self.ds.uid_to_idx[user_id])[0], lo, hi)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)
        u = self.ds.uid_to_idx[user_id]
        pred, mask_u = self._predict_row(u)
        support = self.sim_bool @ mask_u              # neighbours the user has rated
        shrink = support / (support + config.RANK_SHRINKAGE_BETA)
        # shrink toward the stable per-user mean (movie means are noisy for
        # thinly-rated movies and would bring back the obscure-item problem)
        base = self.ds.user_means[u]
        return base + (pred - base) * shrink

    def similar_items(self, movie_id: int, n: int = 10):
        if movie_id not in self.ds.mid_to_idx:
            return []
        i = self.ds.mid_to_idx[movie_id]
        sims = np.asarray(self.sim.getrow(i).todense()).ravel().astype(float)
        order = np.argsort(-sims)
        out = []
        for j in order:
            if j == i or sims[j] <= 0:
                continue
            out.append((int(self.ds.movie_ids[j]), float(sims[j])))
            if len(out) >= n:
                break
        return out


class MatrixFactorizationCF(BaseRecommender):
    """Funk-SVD matrix factorisation.

    Uses the surprise library's SVD when it's installed (fast, Cython);
    otherwise falls back to an equivalent NumPy SGD so the project still runs
    with no native build step. self.backend records which path was taken.
    """

    name = "SVD (Matrix Factorisation)"

    def __init__(self, factors=config.MF_FACTORS, epochs=config.MF_EPOCHS,
                 lr=config.MF_LR, reg=config.MF_REG, seed=config.RANDOM_SEED):
        self.factors, self.epochs, self.lr, self.reg, self.seed = \
            factors, epochs, lr, reg, seed
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
        from surprise import SVD, Dataset as SDataset, Reader

        reader = Reader(rating_scale=config.RATING_SCALE)
        data = SDataset.load_from_df(
            dataset.train[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        algo = SVD(n_factors=self.factors, n_epochs=self.epochs,
                   lr_all=self.lr, reg_all=self.reg, random_state=self.seed)
        algo.fit(trainset)
        # materialise P, Q, biases in OUR index order for fast scoring. We do
        # NOT keep `algo` (it holds a full copy of the trainset); prediction
        # below only needs P/Q/bu/bi.
        n_u, n_m, f = dataset.n_users, dataset.n_movies, self.factors
        self.P = np.zeros((n_u, f)); self.bu = np.zeros(n_u)
        self.Q = np.zeros((n_m, f)); self.bi = np.zeros(n_m)
        for uid, u in dataset.uid_to_idx.items():
            try:
                iu = algo.trainset.to_inner_uid(uid)
                self.P[u] = algo.pu[iu]; self.bu[u] = algo.bu[iu]
            except ValueError:
                pass
        for mid, i in dataset.mid_to_idx.items():
            try:
                ii = algo.trainset.to_inner_iid(mid)
                self.Q[i] = algo.qi[ii]; self.bi[i] = algo.bi[ii]
            except ValueError:
                pass

    def _fit_numpy(self, dataset: Dataset):
        rng = np.random.default_rng(self.seed)
        n_u, n_m, f = dataset.n_users, dataset.n_movies, self.factors
        P = rng.normal(0, 0.1, (n_u, f))
        Q = rng.normal(0, 0.1, (n_m, f))
        bu = np.zeros(n_u); bi = np.zeros(n_m)
        mu = self.mu

        tr = dataset.train
        u_idx = tr.userId.map(dataset.uid_to_idx).to_numpy()
        i_idx = tr.movieId.map(dataset.mid_to_idx).to_numpy()
        r = tr.rating.to_numpy(dtype=np.float64)
        lr, reg = self.lr, self.reg

        for _ in range(self.epochs):
            order = rng.permutation(len(r))
            for n in order:
                u, i, rui = u_idx[n], i_idx[n], r[n]
                pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
                err = rui - pred
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                pu = P[u].copy()
                P[u] += lr * (err * Q[i] - reg * P[u])
                Q[i] += lr * (err * pu - reg * Q[i])
        self.P, self.Q, self.bu, self.bi = P, Q, bu, bi

    def predict(self, user_id, movie_id):
        if user_id not in self.ds.uid_to_idx or movie_id not in self.ds.mid_to_idx:
            return _clip(self.mu)
        u = self.ds.uid_to_idx[user_id]
        i = self.ds.mid_to_idx[movie_id]
        return _clip(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)
        u = self.ds.uid_to_idx[user_id]
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]

    def item_factors(self):
        return self.Q


class HybridRecommender(BaseRecommender):
    """Blend a collaborative model's ranking with content similarity.

    recommend() mixes min-max-normalised CF scores with a content score built
    from the user's liked-movie profile. predict() (for RMSE/MAE) just delegates
    to the collaborative model, whose output is on the real rating scale.
    """

    name = "Hybrid (CF + Content)"

    def __init__(self, cf_model: BaseRecommender | None = None,
                 cf_weight: float = config.HYBRID_CF_WEIGHT):
        self.cf = cf_model
        self.cf_weight = cf_weight

    def fit(self, dataset: Dataset):
        super().fit(dataset)
        if self.cf is None:
            # user-based CF is the strongest top-N ranker here; blending content
            # on top gives the best precision@k of all our models.
            self.cf = UserBasedCF().fit(dataset)
        elif not hasattr(self.cf, "ds"):
            self.cf.fit(dataset)
        self.content = ContentModel(dataset).fit()
        return self

    def _content_scores(self, user_id: int) -> np.ndarray:
        seen = self.ds.seen_movie_indices(user_id)
        if not seen:
            return np.zeros(self.ds.n_movies)
        u = self.ds.uid_to_idx[user_id]
        row = self.ds.R[u]
        liked_idx, weights = [], []
        for j, val in zip(row.indices, row.data):
            if val >= config.LIKE_THRESHOLD:
                liked_idx.append(j)
                weights.append(val - config.LIKE_THRESHOLD + 0.5)
        if not liked_idx:                                # nothing rated highly
            liked_idx = list(row.indices); weights = list(row.data)
        profile = self.content.user_profile(liked_idx, weights)
        return self.content.score_all(profile)

    def predict(self, user_id, movie_id):
        return self.cf.predict(user_id, movie_id)

    def predict_all_ratings(self, user_id):
        return self.cf.predict_all_ratings(user_id)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)
        cf = _minmax(self.cf._score_all(user_id))
        content = _minmax(self._content_scores(user_id))
        return self.cf_weight * cf + (1 - self.cf_weight) * content


class ImplicitCF(BaseRecommender):
    """Item-to-item recommendations from *implicit* feedback.

    Real implicit signals are watch time / clicks / plays. MovieLens has none,
    so we derive an implicit signal from the explicit data: any rating is an
    "interaction", weighted by a confidence that grows with the rating.
    Similarity is cosine over the confidence-weighted matrix -> "people who
    watched X also watched Y".
    """

    name = "Implicit CF"

    def __init__(self, k: int = config.TOP_K_NEIGHBOURS, alpha: float = 1.0):
        self.k = k
        self.alpha = alpha

    def fit(self, dataset: Dataset):
        super().fit(dataset)
        R = dataset.R.tocsr()
        # confidence = 1 + alpha * rating (ALS-style), over binarised interactions
        conf = R.copy().astype(np.float64)
        conf.data = 1.0 + self.alpha * conf.data
        self.conf = conf
        sim = cosine_similarity(conf.T)
        np.fill_diagonal(sim, 0.0)
        self.sim = sparse.csr_matrix(
            _keep_topk_per_row(sim, self.k).astype(np.float32))
        return self

    def predict(self, user_id, movie_id):
        # implicit models rank, they don't estimate a star rating; expose a
        # popularity-style fallback so the interface stays uniform.
        if movie_id in self.ds.mid_to_idx:
            return _clip(self.ds.movie_means[self.ds.mid_to_idx[movie_id]])
        return _clip(self.ds.global_mean)

    def predict_all_ratings(self, user_id):
        lo, hi = config.RATING_SCALE
        return np.clip(self.ds.movie_means.copy(), lo, hi)

    def _score_all(self, user_id):
        if user_id not in self.ds.uid_to_idx:
            return np.full(self.ds.n_movies, -np.inf)
        u = self.ds.uid_to_idx[user_id]
        interacted = (self.conf[u].toarray().ravel() > 0).astype(np.float64)
        return self.sim @ interacted


def build_all_models():
    """Fresh, unfitted instances of every comparable model, keyed by name."""
    return {
        "Popularity": PopularityRecommender(),
        "User-based CF": UserBasedCF(),
        "Item-based CF": ItemBasedCF(),
        "SVD (Matrix Factorisation)": MatrixFactorizationCF(),
        "Hybrid (CF + Content)": HybridRecommender(),
    }