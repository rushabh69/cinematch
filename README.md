# CineMatch

A movie recommender built on the MovieLens dataset (~100k ratings). I wanted to
compare a few recommendation approaches in one project instead of stopping at a
single notebook model, and to actually see where they disagree.

Live demo: https://sprightly-horse-f8df5b.netlify.app

## The part I found interesting

Best RMSE doesn't mean best recommendations. SVD had the lowest RMSE (0.855) but
the hybrid model produced better top-10 lists. And the user/item-based models kept
recommending obscure movies that a single person had rated 5 stars — precision@10
was around 0.01 until I added a support-shrinkage term that pushes thinly-rated
picks down, which took it to about 0.13.

## What it does

- Recommend movies using five models (popularity, user-based CF, item-based CF,
  SVD, and a content + CF hybrid)
- Explain why a movie was recommended ("because you liked X, Y, Z")
- Handle users with no rating history
- Find movies similar to a given one
- Compare the models on RMSE, MAE, precision@10 and recall@10
- Show posters pulled from TMDB

## Running it

```
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.train
streamlit run app.py
```

The first run downloads MovieLens and trains the models. It takes a few seconds.

CLI:

```
python cli.py recommend 42
python cli.py similar "The Matrix"
python cli.py compare
```

Posters need a free TMDB API key in `.streamlit/secrets.toml`:

```
TMDB_API_KEY = "your_key"
```

The live demo is a static build (in `docs/`) that precomputes the recommendations,
so it can be hosted for free on Netlify. Run it locally for the full interactive app.

## Stack

Python, pandas, numpy, scikit-learn, scikit-surprise, Streamlit, Plotly. Data from
GroupLens (MovieLens ml-latest-small), posters from TMDB.

## Tests

```
python tests/test_pipeline.py
```

## AI usage

I used an AI assistant while building this, mainly for parts of `src/models.py`,
wiring up the Streamlit app, and debugging the TMDB poster issue on Windows. The
design and integration decisions were mine, and I tested and deployed it myself.
