# CineMatch

This is a movie recommender project I made using the MovieLens dataset. It has around 100k ratings.

I wanted to try making a movie recommendation system and also compare some different ways of recommending movies. I also wanted to see which one works better.

Live demo: https://sprightly-horse-f8df5b.netlify.app

## What it does

* Recommend movies based on what a user likes
* Uses 5 different recommendation methods
* Shows why a movie was recommended
* Can recommend movies even if the user has not rated anything
* Find similar movies
* Compare how the different models perform
* Shows movie posters using TMDB

## Something I found interesting

I found that the model with the best score is not always the one that gives the best recommendations.

SVD got the best RMSE score, but the hybrid model was giving better movie lists.

At first some models were recommending very random/less popular movies because only a few people had rated them highly. I changed this and the recommendations became much better.

## Running it

First install the requirements:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then download the data and train the models:

```bash
python -m scripts.download_data
python -m scripts.train
```

Start the app:

```bash
streamlit run app.py
```

There is also a CLI:

```bash
python cli.py recommend 42
python cli.py similar "The Matrix"
python cli.py compare
```

For movie posters you need a TMDB API key in `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key"
```

The live demo is a static version of the project. The full app can be run locally.

## Made with

Python, pandas, numpy, scikit-learn, Streamlit and Plotly.

## Tests

```bash
python tests/test_pipeline.py
```

## AI usage

I used AI while making this project. It helped me with some of the code, setting up the Streamlit app and fixing some problems I had with TMDB on Windows.

I did the project and testing myself, but AI helped me when I got stuck.
