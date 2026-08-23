"""Download and unzip the MovieLens ml-latest-small dataset into data/."""
import sys
import zipfile
import urllib.request
from pathlib import Path
# make `from src import config` work when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.RATINGS_CSV.exists():
        print(f"Already have the data at {config.DATASET_DIR}")
        return

    zip_path = config.DATA_DIR / "ml-latest-small.zip"
    print(f"Downloading {config.DATASET_URL}")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path)

    print("Unzipping...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_DIR)

    print(f"Done. Data is in {config.DATASET_DIR}")
    for name in ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]:
        p = config.DATASET_DIR / name
        print(f"  {'ok ' if p.exists() else 'missing'} {name}")


if __name__ == "__main__":
    main()