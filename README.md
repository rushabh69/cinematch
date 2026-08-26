# CineMatch

A movie recommender built on the MovieLens dataset (~100k ratings). I wanted to
do more than train one model in a notebook — this compares five different
recommendation methods, explains why it picks each movie, and handles new users
who haven't rated anything yet.

Live demo: (add your link here)

## What it does

- Recommends movies for a user from what similar users liked
- Compares five approaches: popularity baseline, user-based CF, item-based CF,
  SVD matrix factorisation, and a content + CF hybrid
- Explains each pick ("because you liked X, Y, Z")
- Cold start: unknown users get popular picks; or give it a few movies you like
  and it fds you in without retraining
- "Similar movies" search (by genre/tags, by co-rating, or both)
- A comparison page: RMSE, MAE, precision@10 and recall@10 for every model
- Movie posters via TMDB

## The part I found interesting

Best RMSE doesn't mean best recommendations. SVD won on RMSE (0.855) but the
hybrid built better top-10 lists. And the neighbourhood models kept surfacing
obscure movies that a single user had rated 5 stars — precision@10 was about
0.01 until I added a support-shrinkage term that pushes thinly-rated picks down,
which took it to ~0.13.

## Running it

```
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.train
streamlit run app.py
```

First run downloads the data and trains + caches the models (a few seconds).

There's a CLI too:

```
python cli.py recommend 42
python cli.py similar "The Matrix"
python cli.py compare
```

Posters need a free TMDB API key in `.streamlit/secrets.toml`:

```
TMDB_API_KEY = "your_key"
```

## Stack

Python, pandas, numpy, scikit-learn, scikit-surprise, Streamlit, Plotly. Data
from GroupLens (MovieLens ml-latest-small), posters from TMDB.

## Tests

```
python tests/test_pipeline.py
```

## AI usage

I used Chat gpt while building this — mainly for writing
chunks of the model code in `src/models.py`,wiring up the Streamlit app, and
debugging (for example the TMDB poster SSL/connection issue on Windows). The
design decisions, integration, testing and deployment were mine, and it also
helped draft this README.