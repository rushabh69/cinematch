"""Export recommender data to docs/data/*.json for the static site."""
import sys
import json
import concurrent.futures as cf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.recommender import RecommenderSystem
from src import posters, config

OUT = config.PROJECT_ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)

rec = RecommenderSystem.load()
ds = rec.ds
movie_ids = [int(m) for m in ds.movie_ids]
print(f"movies {len(movie_ids)}, users {ds.n_users}, models {len(rec.model_names())}")

# TMDB is slow/rate-limited, so reuse existing posters unless --posters is set.
poster = {}
if "--posters" in sys.argv:
    print("fetching posters ...")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(posters.poster_url, mid, 1): mid for mid in movie_ids}
        for fut in cf.as_completed(futs):
            mid = futs[fut]
            try:
                poster[mid] = fut.result()
            except Exception:
                poster[mid] = None
            done += 1
            if done % 400 == 0:
                print(f"  posters {done}/{len(movie_ids)}")
    print(f"  got {sum(1 for v in poster.values() if v)}/{len(movie_ids)} posters")
else:
    existing = OUT / "movies.json"
    if existing.exists():
        prev = json.loads(existing.read_text(encoding="utf-8"))
        poster = {int(k): v.get("p") for k, v in prev.items()}
    print("skipping posters (run with --posters to fetch)")

movies = {}
for mid in movie_ids:
    row = rec._movie_row(mid)
    movies[str(mid)] = {"t": row["title"], "g": row["genres"],
                        "a": row["avg_rating"], "n": row["n_ratings"],
                        "p": poster.get(mid)}
(OUT / "movies.json").write_text(json.dumps(movies, ensure_ascii=False), encoding="utf-8")
print("wrote movies.json")

print("building recommendations ...")
models = rec.model_names()
recs = {}
for i, uid in enumerate(ds.user_ids):
    uid = int(uid)
    recs[str(uid)] = {
        name: [{"m": int(r.movieId), "s": round(float(r.score), 3), "w": r.why}
               for r in rec.get_recommendations(uid, n=10, model=name).itertuples()]
        for name in models
    }
    if (i + 1) % 150 == 0:
        print(f"  users {i + 1}/{ds.n_users}")
(OUT / "recs.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
print("wrote recs.json")

print("building similar movies ...")
similar = {}
for i, mid in enumerate(movie_ids):
    df = rec.get_similar_movies(mid, n=10, method="hybrid")
    similar[str(mid)] = [{"m": int(r.movieId), "s": round(float(r.similarity), 3)}
                         for r in df.itertuples()] if len(df) else []
    if (i + 1) % 1000 == 0:
        print(f"  movies {i + 1}/{len(movie_ids)}")
(OUT / "similar.json").write_text(json.dumps(similar, ensure_ascii=False), encoding="utf-8")
print("wrote similar.json")

if rec.metrics is not None:
    m = rec.metrics.reset_index()
    m["model"] = m["model"].str.replace(r" \[.*\]", "", regex=True)
    (OUT / "metrics.json").write_text(m.to_json(orient="records"), encoding="utf-8")
    print("wrote metrics.json")

meta = {"models": models, "default_model": "Hybrid (CF + Content)",
        "users": [int(u) for u in ds.user_ids], "n_movies": len(movie_ids),
        "n_ratings": int(len(ds.ratings))}
(OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
print("wrote meta.json")

for f in sorted(OUT.glob("*.json")):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
print("done")