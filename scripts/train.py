"""Train + evaluate every model, then cache artifacts for the apps.

Run:  python -m scripts.train
Produces:
  artifacts/metrics.csv          - model comparison table
  artifacts/recommender.pkl      - fitted RecommenderSystem (loaded by the apps)
  assets/model_comparison.png    - RMSE/MAE + precision/recall bar charts
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data import load_dataset
from src.models import build_all_models
from src.evaluation import compare_models
from src.recommender import RecommenderSystem


def plot_comparison(metrics: pd.DataFrame, out_path: Path):
    df = metrics.reset_index()
    names = [n.split(" [")[0] for n in df["model"]]
    k = config.EVAL_K
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    # error metrics (lower = better)
    ax = axes[0]
    x = range(len(names))
    w = 0.38
    ax.bar([i - w / 2 for i in x], df["RMSE"], w, label="RMSE", color="#C44E52")
    ax.bar([i + w / 2 for i in x], df["MAE"], w, label="MAE", color="#4C72B0")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Rating accuracy (lower is better)")
    ax.set_ylabel("error"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df["RMSE"]):
        ax.text(i - w / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # ranking metrics (higher = better)
    ax = axes[1]
    ax.bar([i - w / 2 for i in x], df[f"Precision@{k}"], w,
           label=f"Precision@{k}", color="#55A868")
    ax.bar([i + w / 2 for i in x], df[f"Recall@{k}"], w,
           label=f"Recall@{k}", color="#DD8452")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title(f"Top-{k} ranking quality (higher is better)")
    ax.set_ylabel("score"); ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("MovieLens recommender — model comparison", fontsize=14, weight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"  saved chart -> {out_path}")


def main():
    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    ds = load_dataset(verbose=True)

    print("\n" + "=" * 70)
    print("Evaluating models")
    print("=" * 70)
    metrics = compare_models(build_all_models(), ds)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.ARTIFACTS_DIR / "metrics.csv"
    metrics.to_csv(metrics_path)
    print("\nModel comparison")
    print(metrics.round(4).to_string())
    print(f"\n  saved metrics -> {metrics_path}")

    plot_comparison(metrics, config.ASSETS_DIR / "model_comparison.png")

    print("\n" + "=" * 70)
    print("Fitting full recommender (all models + content) for the apps")
    print("=" * 70)
    rec = RecommenderSystem.build(dataset=ds, verbose=True)
    rec.metrics = metrics
    path = rec.save()
    print(f"  saved recommender -> {path}")
    print("\nDone. Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()