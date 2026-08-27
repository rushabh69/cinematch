"""Command-line interface for the movie recommender.

Examples:
    python cli.py recommend 42                 # top-10 for user 42
    python cli.py recommend 42 -n 5 --model "SVD (Matrix Factorisation)"
    python cli.py similar "Matrix"             # movies similar to The Matrix
    python cli.py similar 1 --method content   # by movieId, content-based
    python cli.py search "star wars"
    python cli.py popular                       # cold-start / popularity chart
    python cli.py newuser "Toy Story" "Matrix"  # simulate a brand-new user
    python cli.py profile 42                    # a user's top-rated movies
    python cli.py compare                       # model comparison table

The first run builds + caches the models (a few seconds); later runs load the
cache instantly.
"""
from __future__ import annotations

import argparse
import sys
import io
import warnings

warnings.filterwarnings("ignore")

# make Unicode titles safe on the Windows console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

from src import config
from src.recommender import RecommenderSystem, DEFAULT_MODEL

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 50)


def get_system(rebuild: bool = False) -> RecommenderSystem:
    cache = config.ARTIFACTS_DIR / "recommender.pkl"
    if cache.exists() and not rebuild:
        try:
            return RecommenderSystem.load(cache)
        except Exception as e:  # stale/incompatible cache -> rebuild
            print(f"(cache unreadable: {e}; rebuilding)")
    print("Building recommender (first run) ...")
    rec = RecommenderSystem.build(verbose=True)
    rec.save()
    return rec


def _resolve_movie(rec: RecommenderSystem, token: str) -> int | None:
    """Accept either a numeric movieId or a title substring."""
    token = token.strip()
    if token.isdigit() and int(token) in rec.ds.mid_to_idx:
        return int(token)
    hits = rec.search_movies(token, n=1)
    if len(hits):
        return int(hits.iloc[0]["movieId"])
    return None


def _show(df: pd.DataFrame, cols=None):
    if df is None or len(df) == 0:
        print("  (no results)")
        return
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    print(df.to_string(index=False))


def cmd_recommend(rec, args):
    if rec.is_cold_start(args.user_id):
        print(f"User {args.user_id} is a COLD START (no history) -> popularity fallback\n")
    else:
        print(f"Top {args.n} recommendations for user {args.user_id} "
              f"(model: {args.model}):\n")
    df = rec.get_recommendations(args.user_id, n=args.n, model=args.model)
    _show(df, ["title", "genres", "score", "avg_rating", "n_ratings", "why"])


def cmd_similar(rec, args):
    mid = _resolve_movie(rec, args.movie)
    if mid is None:
        print(f"  no movie matched '{args.movie}'")
        return
    print(f"Movies similar to: {rec.ds.title(mid)}  (method: {args.method})\n")
    df = rec.get_similar_movies(mid, n=args.n, method=args.method)
    _show(df, ["title", "genres", "similarity", "avg_rating", "n_ratings"])


def cmd_search(rec, args):
    print(f"Search results for '{args.query}':\n")
    _show(rec.search_movies(args.query, n=args.n),
          ["movieId", "title", "genres", "avg_rating", "n_ratings"])


def cmd_popular(rec, args):
    print(f"Top {args.n} popular movies (cold-start fallback):\n")
    _show(rec.popular_movies(args.n), ["title", "genres", "avg_rating", "n_ratings", "score"])


def cmd_newuser(rec, args):
    ratings = []
    print("Simulating a new user who likes:")
    for token in args.movies:
        mid = _resolve_movie(rec, token)
        if mid is None:
            print(f"  ? no match for '{token}'")
            continue
        ratings.append((mid, args.rating))
        print(f"  + {rec.ds.title(mid)}  (rated {args.rating})")
    if not ratings:
        print("  nothing matched; showing popular movies instead")
    print(f"\nRecommended for this new user (cold-start fold-in):\n")
    _show(rec.recommend_for_new_user(ratings, n=args.n),
          ["title", "genres", "score", "avg_rating", "n_ratings"])


def cmd_profile(rec, args):
    if args.user_id not in rec.ds.uid_to_idx:
        print(f"  user {args.user_id} not in dataset")
        return
    print(f"User {args.user_id} — top rated movies:\n")
    _show(rec.user_top_rated(args.user_id, n=args.n),
          ["title", "genres", "your_rating", "avg_rating"])


def cmd_compare(rec, args):
    m = rec.metrics
    if m is None:
        path = config.ARTIFACTS_DIR / "metrics.csv"
        if path.exists():
            m = pd.read_csv(path, index_col=0)
    if m is None:
        print("  no metrics found; run:  python -m scripts.train")
        return
    print("Model comparison (test set):\n")
    print(m.round(4).to_string())
    print("\nLower RMSE/MAE = better rating accuracy; "
          "higher Precision/Recall = better top-N ranking.")


def build_parser():
    p = argparse.ArgumentParser(
        description="MovieLens collaborative-filtering recommender (CLI).")
    p.add_argument("--rebuild", action="store_true",
                   help="ignore the cache and refit all models")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("recommend", help="top-N recommendations for a user")
    r.add_argument("user_id", type=int)
    r.add_argument("-n", type=int, default=config.DEFAULT_N_RECOMMENDATIONS)
    r.add_argument("--model", default=DEFAULT_MODEL)
    r.set_defaults(func=cmd_recommend)

    s = sub.add_parser("similar", help="movies similar to a given movie")
    s.add_argument("movie", help="movieId or title substring")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--method", choices=["hybrid", "content", "cf"], default="hybrid")
    s.set_defaults(func=cmd_similar)

    se = sub.add_parser("search", help="search movies by title")
    se.add_argument("query")
    se.add_argument("-n", type=int, default=20)
    se.set_defaults(func=cmd_search)

    po = sub.add_parser("popular", help="popularity chart (cold-start fallback)")
    po.add_argument("-n", type=int, default=10)
    po.set_defaults(func=cmd_popular)

    nu = sub.add_parser("newuser", help="simulate a brand-new user from liked movies")
    nu.add_argument("movies", nargs="+", help="movie titles or ids the user likes")
    nu.add_argument("-n", type=int, default=10)
    nu.add_argument("--rating", type=float, default=5.0)
    nu.set_defaults(func=cmd_newuser)

    pr = sub.add_parser("profile", help="a user's top-rated movies")
    pr.add_argument("user_id", type=int)
    pr.add_argument("-n", type=int, default=10)
    pr.set_defaults(func=cmd_profile)

    cp = sub.add_parser("compare", help="show the model comparison table")
    cp.set_defaults(func=cmd_compare)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = get_system(rebuild=args.rebuild)
    print()
    args.func(rec, args)


if __name__ == "__main__":
    main()