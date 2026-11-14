"""Minimal settings: output directory and device, with an optional .env file.

Environment variables
---------------------
UNBED_DATA_DIR         Where jobs are written. Default: ./data (relative to the working directory)
UNBED_DEVICE           Optional Demucs device: cpu, cuda, or mps (default: auto)
UNBED_ENV_FILE         Optional path to a .env file (otherwise ./.env, then the repo root .env)
UNBED_ALT_DB           Fallback music attenuation in dB (default: -25). Read by the pipeline.
UNBED_DEMUCS_JOBS      CPU worker count passed to Demucs (default: 2)
UNBED_DEMUCS_SEGMENT   Demucs segment length in seconds (default: 7)
UNBED_YTDLP_COOKIES    Netscape cookies file for yt-dlp. UNBED_COOKIES is accepted as an alias.

Existing environment variables are never overwritten by .env values.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_env_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ[key] = val


def load_dotenv() -> None:
    candidates = []
    env_file = os.environ.get("UNBED_ENV_FILE")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    candidates.append(Path.cwd() / ".env")
    candidates.append(REPO_ROOT / ".env")
    seen = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except OSError:
            continue
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        _parse_env_file(p)


load_dotenv()

DATA_DIR = Path(os.environ.get("UNBED_DATA_DIR") or "./data").expanduser()
