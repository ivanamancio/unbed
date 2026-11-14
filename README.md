# unbed

Remove background music from a video and keep the spoken voice.

unbed takes a public video link (YouTube, Instagram, or X) or a local video file, removes or strongly attenuates the background music, and keeps the voice as faithful as possible. It writes a clean MP4 plus a credit package in the original language.

Run it on Linux x86_64 or Apple Silicon (CPU, with optional NVIDIA CUDA or MPS).

> GPU and MPS paths were **not tested** in the environment that packaged this repository. Device auto-detection is included and should work, but please treat the first GPU run as a trial and fall back to `--device cpu` if anything looks off.

## What it does

1. **Ingest** a URL with yt-dlp, preferring the original audio track (no auto-dubs) and **capping video at 1080p** so the picture can be stream-copied — or copy a local file you already have.
2. **Separate** vocals with Demucs `htdemucs_ft` (`--two-stems vocals --segment 7`).
3. **Check the vocal stem** against the original (level, band balance, dropouts). If the stem looks damaged, the tool keeps the music but turns it down by **−25 dB** (overridable) instead of removing it completely. Otherwise it uses full removal.
4. **Loudness**: two-pass `loudnorm` to **−16 LUFS**, kept linear (true peak is raised only if needed so the voice is not squeezed).
5. **Mux**: AAC audio at 256 kbps, **video stream copied** (`-c:v copy`). Music-only tails longer than 5 seconds are trimmed from the no-music version.
6. **Credits** (`credit.md`) are written in the original language (Portuguese, Spanish, or English), with title suggestions, captions, and a credit to the original creator and URL when that information is available.

Both mixes are always written: `… - sem musica.mp4` (full removal) and `… - musica atenuada.mp4` (attenuated). The “chosen” file is the full-removal mix unless the vocal stem looked damaged.

## Requirements

- Linux x86_64 or Apple Silicon. Intel Macs and Linux ARM are not supported: Demucs 4.1 depends on `sphn`, which has no wheel for those machines.
- Python 3.11 or newer (3.13 is known to work)
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (ideally built with the soxr resampler)
- Optional: an NVIDIA GPU with a recent driver, or Apple Silicon for MPS

## Install

```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg python3 python3-venv python3-pip

# macOS (Homebrew)
brew install ffmpeg python

git clone https://github.com/ivanamancio/unbed.git
cd unbed
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install **torch first**, from the index that matches your machine, so `requirements.txt` cannot replace it with a different wheel. Demucs 4.1 does not import torchaudio.

**CPU** (Linux x86_64):

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**NVIDIA CUDA** — PyTorch 2.15’s stable CUDA build is 13.2:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

An experimental CUDA 13.4 build lives at `https://download.pytorch.org/whl/cu134`. You only need a recent enough NVIDIA **driver**; you do not need a full local CUDA toolkit. Confirm with:

```bash
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

**Apple Silicon**: install `torch` from PyPI (the default index), then `pip install -r requirements.txt`. MPS is picked automatically when it is available. If Demucs fails on MPS, the pipeline retries on CPU.

The torch pin above is from a working **CPU** environment in November 2026. If a CUDA or Mac index rejects that exact pin, install `torch` unpinned from the **same** index. The current selector is at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Usage

```bash
source .venv/bin/activate

# A public video URL
python -m unbed https://www.youtube.com/watch?v=VIDEO_ID

# A file already on disk
python -m unbed ./talk.mp4 --out ./data

# Common flags
python -m unbed URL_OR_FILE \
  --out ./data \
  --device cpu \
  --attenuation-db -25 \
  --keep-temp \
  --cookies ./cookies.txt \
  --small
```

| Flag | Meaning |
| --- | --- |
| `--out DIR` | Parent directory for the job folder (default: `$UNBED_DATA_DIR` or `./data`) |
| `--device cpu\|cuda\|mps` | Force the Demucs device (default: auto-detect) |
| `--attenuation-db N` | Music level in the fallback mix, in dB (default: `-25`) |
| `--keep-temp` | Keep the `work/` folder (raw audio and stems) instead of deleting it |
| `--cookies FILE` | Netscape cookies file, for login-required source videos |
| `--small` | Also write a 1080p copy under 50 MB when the main file is larger |

You can also set `UNBED_DATA_DIR` and `UNBED_DEVICE` in the environment, or in an optional `.env` file (`cp .env.example .env`). Values already in the environment are not overwritten.

Other optional variables: `UNBED_ALT_DB` (same as `--attenuation-db`), `UNBED_DEMUCS_JOBS` (CPU workers, default 2), `UNBED_DEMUCS_SEGMENT` (default 7), and `UNBED_YTDLP_COOKIES` (same as `--cookies`).

## Outputs

Each run writes a job folder `$OUT/<id>/`:

```
final/<title> - sem musica.mp4        # voice only (or the chosen mix if the stem was damaged)
final/<title> - musica atenuada.mp4   # voice + music at --attenuation-db
final/credit.md                       # titles, captions, credits, in the source language
final/<title> (1080p small).mp4       # only with --small, and only if needed
result.json                           # paths, fidelity metrics, the chosen mix
<id>.info.json                        # source metadata
```

## Devices and tuning

The pipeline chooses a device automatically and **logs it**:

1. `cuda` if `torch.cuda.is_available()`
2. otherwise `mps` if Apple Silicon MPS is available
3. otherwise `cpu`

`--device` (or `UNBED_DEVICE`) overrides that. If you ask for CUDA or MPS and it is missing, the pipeline warns and uses CPU. If Demucs **fails on MPS**, it retries on CPU automatically.

On CUDA and MPS, Demucs is called with `-d <device>` and **without** `-j`. The segment size stays at **7** (the `htdemucs_ft` maximum is about 7.8). On CPU it also passes `-j` (default 2, overridable with `UNBED_DEMUCS_JOBS`). You can set `UNBED_DEMUCS_SEGMENT` if you really need to, but values above 7 are tight for this model.

Again: these GPU/MPS branches were not tested when this repo was packaged.

## Docker

The included `Dockerfile` is **CPU-only** (Python 3.13 slim + ffmpeg + the official CPU torch wheel).

```bash
docker build -t unbed .
docker run --rm -v "$PWD/data:/data" unbed \
  https://www.youtube.com/watch?v=VIDEO_ID --out /data
docker run --rm -v "$PWD/data:/data" -v "$PWD/talk.mp4:/talk.mp4" \
  unbed /talk.mp4 --out /data
```

For an NVIDIA GPU, build your own image from an official CUDA runtime base (for example `nvidia/cuda:13.2.2-cudnn-runtime-ubuntu24.04`, or whatever tag NVIDIA currently publishes), install Python 3.11+, ffmpeg, and:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

Then run it with GPU access:

```bash
docker run --gpus all --rm -v "$PWD/data:/data" your-gpu-image URL --out /data --device cuda
```

This GPU Docker path was not tested here.

## Troubleshooting

**Login-required / age-restricted / members-only source videos, or “sign in to confirm you’re not a bot”.** yt-dlp will refuse them without cookies. Export a Netscape `cookies.txt` from a browser session that can play the video, and pass `--cookies cookies.txt` (or set `UNBED_YTDLP_COOKIES`). This tool does not log into those sites for you.

**CUDA is installed but the tool still uses CPU.** Check `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"`. A CPU wheel (`+cpu` in the version) cannot see the GPU — reinstall torch from the `cu132` index. You can also pass `--device cuda` to make a missing GPU obvious in the log.

**ffmpeg / soxr errors.** Stem resampling uses ffmpeg's soxr resampler. Install a full ffmpeg build rather than a minimal one, and confirm with `ffmpeg -filters | grep soxr`.

**The job is slow on CPU.** That is expected: `htdemucs_ft` is heavy. A short clip is fine for a trial; longer videos are happier on a GPU (untested here).

## Credits, fair use, and your responsibility

The tool never claims ownership of the source. `credit.md` names the original creator and URL when the source provides them. The speech is left alone; only the soundtrack is removed or turned down, for informational and educational use.

**You** are responsible for how you use the output: copyright, fair use / fair dealing in your jurisdiction, and the original creator’s rights. This software is provided as-is under the MIT license (see `LICENSE`).

## Layout

```
unbed/           Python package (pipeline + CLI)
Dockerfile     CPU image
.env.example   Optional defaults (data dir, device, attenuation, Demucs, cookies)
```

Job folders stay in `UNBED_DATA_DIR` / `--out` and are gitignored.
