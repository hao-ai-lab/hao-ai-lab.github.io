#!/usr/bin/env python3
"""Download a gallery profile's references and write a UniServe ref2va request."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blog-url", default="https://haoailab.com/blogs/fastvideo-omniref-update/"
    )
    parser.add_argument("--composition", default="image-voice")
    parser.add_argument("--seconds", type=int, choices=(5, 10, 15), default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="FastH3-OmniRef")
    parser.add_argument("--media-directory", type=Path, default=Path("omniref-media"))
    parser.add_argument("--output", type=Path, default=Path("omniref-request.json"))
    args = parser.parse_args()
    blog_url = args.blog_url.rstrip("/") + "/"
    with urlopen(urljoin(blog_url, "data/benchmark.json"), timeout=30) as response:
        data = json.load(response)
    profile = next(
        (
            profile
            for profile in data["profiles"]
            if profile["composition"] == args.composition
            and profile["target_seconds"] == args.seconds
        ),
        None,
    )
    if profile is None:
        choices = ", ".join(sorted({p["composition"] for p in data["profiles"]}))
        parser.error(f"Unknown composition {args.composition!r}. Choose from: {choices}")
    media_directory = args.media_directory.resolve()
    media_directory.mkdir(parents=True, exist_ok=True)
    conditions = []
    for reference in profile["conditions"]:
        destination = media_directory / Path(reference["file"]).name
        content = destination.read_bytes() if destination.is_file() else b""
        if hashlib.sha256(content).hexdigest() != reference["sha256"]:
            with urlopen(urljoin(blog_url, reference["file"]), timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != reference["sha256"]:
                raise SystemExit(f"Reference checksum mismatch: {reference['file']}")
            destination.write_bytes(content)
        conditions.append(
            {"type": reference["type"], "role": "reference", "uri": destination.as_uri()}
        )
    request = {
        "model": args.model,
        "task": "ref2va",
        "prompt": profile["prompt"],
        "conditions": conditions,
        "target": {
            "short_edge": min(profile["width"], profile["height"]),
            "aspect_ratio": "16:9",
            "duration_seconds": args.seconds,
        },
        "seed": args.seed,
    }
    args.output.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    print(f"References: {media_directory}")
    print(f"Profile: {profile['label']}, {args.seconds} seconds, seed {args.seed}")


if __name__ == "__main__":
    main()
