"""Fetch movie poster URLs from TMDB using the ids in links.csv.

Needs a free TMDB API key via the TMDB_API_KEY env var or Streamlit secrets.
If no key is set, everything returns None and the app falls back to text tables.
"""
import os
import ssl
import time
import shutil
import subprocess
import json
import urllib.request
from functools import lru_cache

import pandas as pd

from . import config

IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

# Use certifi's CA bundle so HTTPS works even when the OS/Python cert store is
# missing root certificates (common on fresh Windows Python installs).
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

_CURL = shutil.which("curl") or shutil.which("curl.exe")


def _fetch_json(url: str):
    """GET a JSON URL. Try urllib first; if it fails (some antivirus/firewalls
    reset Python's OpenSSL TLS while allowing native connections), fall back to
    the system curl, which uses the OS TLS stack."""
    try:
        with urllib.request.urlopen(url, timeout=10, context=_SSL_CONTEXT) as resp:
            return json.loads(resp.read())
    except Exception:
        pass
    if _CURL:
        try:
            r = subprocess.run([_CURL, "-sS", "--max-time", "15", url],
                               capture_output=True)
            if r.returncode == 0 and r.stdout:
                return json.loads(r.stdout)
        except Exception:
            pass
    return None


_poster_cache = {}


def poster_url(movie_id: int, retries: int = 3):
    """Poster image URL for a movieId, or None if unavailable."""
    mid = int(movie_id)
    if mid in _poster_cache:
        return _poster_cache[mid]
    key = _api_key()
    if not key:
        return None
    tmdb_id = _movie_to_tmdb().get(mid)
    if not tmdb_id:
        return None
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={key}"
    for attempt in range(retries):
        data = _fetch_json(url)
        if data is not None:
            path = data.get("poster_path")
            result = IMAGE_BASE + path if path else None
            _poster_cache[mid] = result
            return result
        time.sleep(0.4 * (attempt + 1))
    return None