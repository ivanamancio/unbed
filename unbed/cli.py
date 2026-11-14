"""Command-line entry point.

Usage:
  python -m unbed <url_or_path> [--out DIR] [--device cpu|cuda|mps]
                              [--attenuation-db N] [--keep-temp]
                              [--cookies FILE] [--small]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from unbed import config, pipeline


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _fail(e):
    logging.getLogger("cli").error("%s", e)
    print("Error: %s" % (e.pt or e), file=sys.stderr)
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unbed",
        description="unbed — remove background music and keep the voice.",
    )
    parser.add_argument("source", help="a video URL (YouTube, Instagram, X) or a local file path")
    parser.add_argument("--out", default=None,
                        help="parent directory for the job folder (default: $UNBED_DATA_DIR or ./data)")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                        help="Demucs device (default: auto-detect, or $UNBED_DEVICE)")
    parser.add_argument("--attenuation-db", type=float, default=None, metavar="N",
                        help="music level in the fallback mix, in dB (default: -25)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="keep the work/ folder (raw audio, stems) instead of deleting it")
    parser.add_argument("--cookies", default=None, metavar="FILE",
                        help="Netscape cookies file for login-required source videos")
    parser.add_argument("--small", action="store_true",
                        help="also write a 1080p copy under 50 MB (only if the main file is larger)")
    args = parser.parse_args(argv)

    _setup_logging()
    log = logging.getLogger("cli")
    if args.device:
        os.environ["UNBED_DEVICE"] = args.device

    source = args.source
    if not pipeline.is_url(source):
        source = os.path.abspath(os.path.expanduser(source))
        if not os.path.isfile(source):
            print("Error: no such file or URL: %s" % args.source, file=sys.stderr)
            return 2

    try:
        if pipeline.is_url(source):
            info = pipeline.probe(source, cookies=args.cookies)
            key = pipeline.video_key(info)
        else:
            key = pipeline.video_key(pipeline.local_info(source, original_path=source))
    except pipeline.PipelineError as e:
        return _fail(e)

    parent = os.path.abspath(os.path.expanduser(args.out or str(config.DATA_DIR)))
    job_dir = os.path.join(parent, key)
    log.info("Processing %s -> %s", source, job_dir)
    try:
        res = pipeline.process(
            source, job_dir,
            keep_temp=args.keep_temp,
            cookies=args.cookies,
            device=args.device,
            alt_db=args.attenuation_db,
            small=args.small,
        )
    except pipeline.PipelineError as e:
        return _fail(e)

    title = res.get("metadata", {}).get("title") or key
    print("Title:   %s" % title)
    print("Chosen:  %s" % res["chosen"])
    print("Alt:     %s" % res.get("alt"))
    if res.get("small"):
        print("Small:   %s" % res["small"])
    print("Credits: %s" % res["credit"])
    print("Device:  %s" % res.get("metadata", {}).get("device"))
    if res.get("quality_notes"):
        print("Notes:   %s" % " | ".join(res["quality_notes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
