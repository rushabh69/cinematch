"""Fetch movie poster URLs from TMDB using the ids in links.csv.

Needs a free TMDB API key via the TMDB_API_KEY env var or Streamlit secrets.
If no key is set, everything returns None and the app falls back to text tables.
"""
import ssl
import os
import json
import urllib.request
from functools import lru_cache

import pandas as pd

from . import config

IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = None

def _api_key():
    key = os.environ.get("TMDB_API_KEY")
    if key:
        return key
    try:                                   # fall back to Streamlit secrets
        import streamlit as st
        return st.secrets.get("TMDB_API_KEY")
    except Exception:
        return None


def has_key() -> bool:
    return bool(_api_key())


@lru_cache(maxsize=1)
def _movie_to_tmdb():
    """map movieId -> tmdbId from links.csv"""
    if not config.LINKS_CSV.exists():
        return {}
    links = pd.read_csv(config.LINKS_CSV)
    return {int(m): int(t) for m, t in zip(links.movieId, links.tmdbId)
            if pd.notna(t)}


@lru_cache(maxsize=8192)
def poster_url(movie_id: int):
    """Poster image URL for a movieId, or None if unavailable."""
    key = _api_key()
    if not key:
        return None
    tmdb_id = _movie_to_tmdb().get(int(movie_id))
    if not tmdb_id:
        return None
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={key}"
    try:
        with urllib.request.urlopen(url, timeout=6, context=_SSL_CONTEXT) as resp:
            data = json.loads(resp.read())
        path = data.get("poster_path")
        return IMAGE_BASE + path if path else None
    except Exception as e:
        print("POSTER ERROR:", repr(e))
        return None