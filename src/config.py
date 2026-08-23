"""All the tunable settings in one place so everything imports from here."""
from __future__ import annotations

from pathlib import Path
# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "ml-latest-small"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"   # trained model cache + metrics
ASSETS_DIR = PROJECT_ROOT / "assets"         # generated charts

RATINGS_CSV = DATASET_DIR / "ratings.csv"
MOVIES_CSV = DATASET_DIR / "movies.csv"
TAGS_CSV = DATASET_DIR / "tags.csv"
LINKS_CSV = DATASET_DIR / "links.csv"

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# data pipeline
MIN_RATINGS_PER_USER = 5       # drop users with fewer ratings than this
MIN_RATINGS_PER_MOVIE = 5      # drop movies with fewer ratings than this
TEST_SIZE = 0.20               # per-user holdout for the test set
RANDOM_SEED = 42

# rating >= this counts as a "like" (used for precision/recall and explanations)
LIKE_THRESHOLD = 4.0

# models
TOP_K_NEIGHBOURS = 40          # neighbourhood size for user/item CF
DEFAULT_N_RECOMMENDATIONS = 10
RATING_SCALE = (0.5, 5.0)      # clip predictions to the MovieLens scale

# neighbourhood CF tends to top-rank obscure movies that one similar user loved.
# when ranking we shrink scores toward the mean by support/(support+beta) so
# those thinly-backed picks drop down. big jump in precision@k, and it only
# affects ranking, not the rating predictions used for RMSE.
RANK_SHRINKAGE_BETA = 10

# matrix factorisation (SVD)
MF_FACTORS = 100
MF_EPOCHS = 30
MF_LR = 0.005
MF_REG = 0.02

# hybrid: weight on collaborative score vs content (genres + tags)
HYBRID_CF_WEIGHT = 0.85

# popularity model: IMDB-style prior, min votes before we trust a movie's own mean
POPULARITY_MIN_VOTES = 20

EVAL_K = 10