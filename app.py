"""Streamlit web app for the MovieLens recommender.

Run:  streamlit run app.py

Four sections:
  * Recommendations   - top-N for a selected user (with explanations)
  * Similar Movies     - "movies similar to this" (content / CF / hybrid)
  * New User           - cold-start: build a taste profile on the fly
  * Model Comparison   - RMSE / MAE / precision / recall across models
"""
from __future__ import annotations
from src import posters

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.recommender import RecommenderSystem, DEFAULT_MODEL

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")


# load / build the system once, cached across reruns
@st.cache_resource(show_spinner=True)
def load_system() -> RecommenderSystem:
    cache = config.ARTIFACTS_DIR / "recommender.pkl"
    if cache.exists():
        try:
            return RecommenderSystem.load(cache)
        except Exception:
            pass
    with st.spinner("Training models for the first time (a few seconds)…"):
        rec = RecommenderSystem.build(verbose=False)
        rec.save()
    return rec


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    path = config.ARTIFACTS_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


rec = load_system()
metrics = load_metrics()

STAR = "⭐"


def stars(x):
    try:s
        return STAR * int(round(float(x)))
    except Exception:
        return ""


def show_movies(df: pd.DataFrame, score_col=None, score_label="score"):
    """Render a movie table with a couple of friendly columns."""
    if df is None or len(df) == 0:
        st.info("No results.")
        return
    view = df.copy()
    cols = {"title": "Title", "genres": "Genres",
            "avg_rating": "Avg", "n_ratings": "# ratings"}
    order = ["title", "genres"]
    if score_col and score_col in view.columns:
        view[score_label] = view[score_col].round(3)
        order.append(score_label)
    if "why" in view.columns:
        order.append("why"); cols["why"] = "Why recommended"
    order += ["avg_rating", "n_ratings"]
    order = [c for c in order if c in view.columns]
    view = view[order].rename(columns=cols)
    st.dataframe(view, width='stretch', hide_index=True)

def render_movie_cards(df, score_col=None, score_label="Score", per_row=5):
    """Poster-card grid; falls back to the text table if there's no API key."""
    if df is None or len(df) == 0:
        st.info("No results.")
        return
    if not posters.has_key():
        show_movies(df, score_col, score_label)
        return
    rows = df.to_dict("records")
    for i in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, item in zip(cols, rows[i:i + per_row]):
            with col:
                url = posters.poster_url(int(item["movieId"]))
                if url:
                    st.image(url, width='stretch')
                else:
                    st.markdown(
                        "<div style='aspect-ratio:2/3;background:#1c2333;border-radius:8px;"
                        "display:flex;align-items:center;justify-content:center;font-size:2rem'>🎬</div>",
                        unsafe_allow_html=True)
                st.markdown(f"**{item['title']}**")
                bits = []
                if score_col and score_col in item and pd.notna(item.get(score_col)):
                    bits.append(f"{score_label} {round(float(item[score_col]), 3)}")
                if "avg_rating" in item and pd.notna(item.get("avg_rating")):
                    bits.append(f"⭐ {item['avg_rating']}")
                if bits:
                    st.caption(" · ".join(bits))
                if item.get("why"):
                    st.caption(item["why"])
# sidebar
st.sidebar.title("🎬 MovieLens Recommender")
st.sidebar.caption("Collaborative filtering on ml-latest-small")
page = st.sidebar.radio(
    "Explore",
    ["🎯 Recommendations", "🔎 Similar Movies", "🆕 New User (cold start)",
     "📊 Model Comparison", "ℹ️ About"],
)
st.sidebar.markdown("---")
st.sidebar.metric("Users", f"{rec.ds.n_users:,}")
st.sidebar.metric("Movies", f"{rec.ds.n_movies:,}")
st.sidebar.metric("Ratings (train+test)", f"{len(rec.ds.ratings):,}")
if metrics is not None:
    best = metrics["RMSE"].idxmin()
    st.sidebar.markdown(f"**Best RMSE:** {best.split(' [')[0]} "
                        f"({metrics['RMSE'].min():.3f})")


# 1. Recommendations
if page == "🎯 Recommendations":
    st.header("🎯 Personalised recommendations")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        user_id = st.selectbox("Select a user", rec.known_user_ids(),
                               index=0, help="Every user in the dataset")
    with c2:
        model = st.selectbox("Model", rec.model_names(),
                             index=rec.model_names().index(DEFAULT_MODEL)
                             if DEFAULT_MODEL in rec.model_names() else 0)
    with c3:
        n = st.slider("How many", 5, 25, 10)

    left, right = st.columns([3, 2])
    with right:
        st.subheader("This user's favourites")
        prof = rec.user_top_rated(user_id, n=8)
        if len(prof):
            prof = prof.copy()
            prof["★"] = prof["your_rating"].map(stars)
            st.dataframe(prof[["title", "★", "your_rating"]]
                         .rename(columns={"title": "Title", "your_rating": "Rating"}),
                         width='stretch', hide_index=True)
    with left:
        st.subheader(f"Top {n} picks")
        if rec.is_cold_start(user_id):
            st.warning("Cold start: this user has no ratings — showing popular movies.")
        recs = rec.get_recommendations(user_id, n=n, model=model)
        render_movie_cards(recs, score_col="score", score_label="score")


# 2. Similar movies
elif page == "🔎 Similar Movies":
    st.header("🔎 Movies similar to…")
    c1, c2, c3 = st.columns([3, 1.5, 1])
    with c1:
        query = st.text_input("Search a movie", "Matrix")
    with c3:
        n = st.slider("How many", 5, 25, 10, key="sim_n")
    matches = rec.search_movies(query, n=50) if query else pd.DataFrame()
    if len(matches) == 0:
        st.info("Type part of a movie title to search.")
    else:
        labels = {f"{r.title}  ·  {r.n_ratings} ratings": int(r.movieId)
                  for r in matches.itertuples()}
        choice = st.selectbox("Pick the movie you meant", list(labels.keys()))
        with c2:
            method = st.radio("Similarity", ["hybrid", "content", "cf"],
                              help="content = genres+tags · cf = co-rating · hybrid = both")
        movie_id = labels[choice]
        st.subheader(f"Because you're looking at: {rec.ds.title(movie_id)}")
        st.caption(f"Genres: {rec.ds.genres(movie_id) or '—'}")
        sim = rec.get_similar_movies(movie_id, n=n, method=method)
        render_movie_cards(sim, score_col="similarity", score_label="Similarity")


# 3. New user cold start
elif page == "🆕 New User (cold start)":
    st.header("🆕 New user — build a taste profile")
    st.write("Pick a few movies you love. We fold you into the model **without "
             "retraining** using content + collaborative similarity.")
    # offer well-known movies first for a nice UX
    catalog = rec.ds.movies[rec.ds.movies.movieId.isin(rec.ds.mid_to_idx)]
    counts = rec._movie_counts
    order = np.argsort(-counts)
    popular_titles = [rec.ds.title(int(rec.ds.movie_ids[i])) for i in order[:400]]
    title_to_id = {rec.ds.title(int(rec.ds.movie_ids[i])): int(rec.ds.movie_ids[i])
                   for i in order[:400]}
    picks = st.multiselect("Movies you like", popular_titles,
                           default=popular_titles[:2] if popular_titles else [])
    rating = st.slider("How much do you like them?", 3.0, 5.0, 5.0, 0.5)
    n = st.slider("How many recommendations", 5, 25, 10, key="nu_n")
    if st.button("Recommend for me", type="primary"):
        ratings = [(title_to_id[t], rating) for t in picks]
        if not ratings:
            st.warning("Pick at least one movie (or see the Popular fallback below).")
        recs = rec.recommend_for_new_user(ratings, n=n)
        render_movie_cards(recs, score_col="score", score_label="Score")


# 4. Model comparison
elif page == "📊 Model Comparison":
    st.header("📊 Model comparison")
    if metrics is None:
        st.warning("No metrics found. Run `python -m scripts.train` first.")
    else:
        m = metrics.copy()
        m.index = [i.split(" [")[0] for i in m.index]
        k = config.EVAL_K
        cprec, cerr = st.columns(2)
        with cerr:
            fig = go.Figure()
            fig.add_bar(name="RMSE", x=m.index, y=m["RMSE"], marker_color="#C44E52")
            fig.add_bar(name="MAE", x=m.index, y=m["MAE"], marker_color="#4C72B0")
            fig.update_layout(barmode="group", title="Rating accuracy (lower is better)",
                              height=420, legend=dict(orientation="h"))
            st.plotly_chart(fig, width='stretch')
        with cprec:
            fig = go.Figure()
            fig.add_bar(name=f"Precision@{k}", x=m.index, y=m[f"Precision@{k}"],
                        marker_color="#55A868")
            fig.add_bar(name=f"Recall@{k}", x=m.index, y=m[f"Recall@{k}"],
                        marker_color="#DD8452")
            fig.update_layout(barmode="group",
                              title=f"Top-{k} ranking (higher is better)",
                              height=420, legend=dict(orientation="h"))
            st.plotly_chart(fig, width='stretch')

        st.subheader("Full metrics")
        st.dataframe(m.round(4), width='stretch')
        st.markdown(
            "**Takeaways**\n"
            "- **SVD** is the best *rating predictor* (lowest RMSE/MAE).\n"
            "- **Hybrid / User-based CF** are the best *rankers* (top precision@k, "
            "recall@k, hit-rate) — RMSE isn't the whole story for top-N.\n"
            "- **Popularity** is a surprisingly strong ranking baseline (popularity "
            "bias in the test set), which is exactly why it's the cold-start fallback."
        )


# 5. About
else:
    st.header("ℹ️ About this project")
    st.markdown(
        """
This is a movie recommendation engine built on the **MovieLens ml-latest-small**
dataset (~100K ratings). It implements and compares several collaborative-filtering
approaches and wraps them in a friendly API + this app.

**Models**
- Popularity (IMDB weighted rating) — baseline & cold-start fallback
- User-based collaborative filtering (cosine similarity, support-shrunk ranking)
- Item-based collaborative filtering (adjusted cosine)
- SVD matrix factorisation (via the `surprise` library)
- Hybrid = collaborative + content (genres + tags TF-IDF)
- Implicit-feedback CF (bonus)

**Features**
- `get_recommendations(user_id, n)` with *"because you liked X, Y, Z"* explanations
- `get_similar_movies(movie_id, n)` — content, collaborative, or hybrid
- Cold-start handling — popularity fallback + new-user fold-in with no retraining
- Evaluation — RMSE, MAE, precision@k, recall@k, hit-rate

Built with pandas, numpy, scipy, scikit-learn, scikit-surprise, plotly and Streamlit.
        """
    )
    if (config.ASSETS_DIR / "model_comparison.png").exists():
        st.image(str(config.ASSETS_DIR / "model_comparison.png"))

st.markdown("---")
st.caption("MovieLens data © GroupLens. Built for learning / demo purposes.")