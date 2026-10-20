#!/usr/bin/env python3
"""Export a valid matched FastH3 Omni cohort, preserving videos and hiding 36L timings.

The input is a frozen matrix API response. Raw deployment receipts are checked
locally, then only publication fields are exported. Absolute paths, service
URLs, job IDs, historical values and 36L latency metrics are never published.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from urllib.parse import parse_qs, urlparse

COMPOSITIONS = {"image": "1 image", "image-voice": "Image + voice", "images3": "3 images",
                "images9": "9 images", "clip": "1 short clip", "video": "1 long clip",
                "clips3": "3 clips", "mixed": "Image + video + voice", "full": "9 images + 3 clips"}
VARIANTS = {"base": ("sglang", "original", "Base"),
            "fast": ("uniserve", "original", "FastH3 Omni"),
            "decoder36": ("uniserve", "nvfp4_36l", "FastH3 Omni · 36L VAE")}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def artifact(url, root):
    relative = parse_qs(urlparse(url).query)["path"][0]
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or invalid artifact: {relative}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path)
    p.add_argument("--page", required=True, type=Path)
    p.add_argument("--ffmpeg", required=True, type=Path)
    p.add_argument("--ffprobe", required=True, type=Path)
    p.add_argument("--gpu-count", default=8, type=int)
    p.add_argument("--seed", default=0, type=int)
    args = p.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    selected = [c for c in snapshot["cells"] if c["gpu_count"] == args.gpu_count and c["hardware"] == "GB200"]
    if len(selected) != 81 or any(c["state"] != "completed" or c.get("measured_requests") != 3 for c in selected):
        raise ValueError("Export requires all 81 valid completed cells in the selected GPU cohort")
    output = args.page
    for directory in ["data", "img", "media/inputs", "media/outputs", "media/posters"]:
        (output / directory).mkdir(parents=True, exist_ok=True)
    assets = {}

    def copy_asset(source, relative, expected=None):
        checksum = sha(source)
        if expected and checksum != expected:
            raise ValueError(f"Source checksum mismatch: {source.name}")
        target = output / relative
        if not target.exists():
            shutil.copyfile(source, target)
        if sha(target) != checksum:
            raise ValueError(f"Destination checksum mismatch: {relative}")
        assets[relative] = {"file": relative, "bytes": target.stat().st_size, "sha256": checksum,
                            "treatment": "original bytes"}
        return relative

    profiles, poster_jobs = [], []
    base_request_policies = set()
    for composition, label in COMPOSITIONS.items():
        for seconds in [5, 10, 15]:
            key = f"{composition}:{seconds}"
            workload = snapshot["workloads"][key]
            profile = {"id": f"{composition}-{seconds}s", "composition": composition, "label": label,
                       "target_seconds": seconds, "seed": args.seed, "width": 1344, "height": 768,
                       "fps": 24, "gpu_count": args.gpu_count, "prompt": workload["prompt"],
                       "conditions": [], "outputs": {}, "performance": {}}
            counters = {"image": 0, "video": 0, "audio": 0}
            for condition in workload["conditions"]:
                source = artifact(condition["url"], args.artifacts)
                kind = condition["type"]
                family = "image" if kind == "image" else ("audio" if kind == "audio" else "video")
                counters[family] += 1
                caption = {"image": "Picture", "audio": "Audio", "video": "Video"}[family] + f" {counters[family]}"
                if family == "video":
                    caption += " · with original audio"
                relative = copy_asset(source, f"media/inputs/{source.name}")
                profile["conditions"].append({"type": family, "caption": caption, "file": relative})
            for variant, (backend, vae, variant_label) in VARIANTS.items():
                cell = next(c for c in selected if c["composition"] == composition and c["target_seconds"] == seconds
                            and c["backend"] == backend and c["vae_variant"] == vae)
                summary_path = Path(cell["result_path"])
                if not summary_path.resolve().is_relative_to(args.artifacts.resolve()):
                    raise ValueError("Summary is outside the artifact root")
                summary = json.loads(summary_path.read_text())
                if summary.get("status") != "completed" or not summary.get("validation", {}).get("valid"):
                    raise ValueError("Unvalidated source summary")
                if variant == "base":
                    extra_params = summary["workload"]["video"]["extra_params"]
                    base_request_policies.add((extra_params["quality"], extra_params["reference_resize_mode"]))
                rows = [json.loads(line) for line in (summary_path.parent / "requests.jsonl").read_text().splitlines()]
                rows.sort(key=lambda r: r["example"]["seed"])
                if [r["example"]["seed"] for r in rows] != [0, 1, 2] or any(not r["success"] for r in rows):
                    raise ValueError("Canonical measured seeds are missing or failed")
                if any(r["example"]["prompt"] != workload["prompt"] for r in rows):
                    raise ValueError("Prompt mismatch across the matched cohort")
                expected_conditions = [{k: v for k, v in c.items() if k != "url"} for c in workload["conditions"]]
                if any(r["example"]["conditions"] != expected_conditions for r in rows):
                    raise ValueError("Reference condition mismatch")
                row = rows[args.seed]
                video = next(v for v in cell["videos"] if v["seed"] == args.seed)
                source = artifact(video["url"], args.artifacts)
                relative = copy_asset(source, f"media/outputs/{variant}-{composition}-{seconds}s-seed{args.seed}.mp4",
                                      row["video_output"]["sha256"])
                shape = row["video_output"]
                if shape["width"] != 1344 or shape["height"] != 768 or shape["audio_channels"] != 2:
                    raise ValueError("Unexpected output geometry/audio")
                if profile.get("frames", shape["frame_count"]) != shape["frame_count"]:
                    raise ValueError("Frame count mismatch across variants")
                profile["frames"] = shape["frame_count"]
                profile["actual_video_seconds"] = shape["video_duration_s"]
                poster = f"media/posters/{variant}-{composition}-{seconds}s-seed{args.seed}.jpg"
                poster_jobs.append((source, poster))
                profile["outputs"][variant] = {"label": variant_label, "file": relative, "poster": poster}
                # The 36L lane is qualitative only. Never export any timing from it.
                if variant != "decoder36":
                    measured = [r["e2e_ms"] / 1000 for r in rows]
                    mean, std = statistics.mean(measured), statistics.stdev(measured)
                    if not all(math.isfinite(v) and v > 0 for v in measured):
                        raise ValueError("Invalid measured latency")
                    if not math.isclose(mean, cell["latency_mean_s"], rel_tol=1e-10):
                        raise ValueError("API mean does not match the original measured requests")
                    if not math.isclose(std, cell["latency_std_s"], rel_tol=1e-8):
                        raise ValueError("API sample standard deviation does not match")
                    profile["performance"][variant] = {"mean_s": mean, "sample_std_s": std,
                        "measured_seconds": measured, "measured_seeds": [0, 1, 2]}
            profile["performance"]["speedup"] = (profile["performance"]["base"]["mean_s"] /
                                                  profile["performance"]["fast"]["mean_s"])
            profiles.append(profile)

    if len(base_request_policies) != 1:
        raise ValueError("Base cohort must use one declared quality and reference-resize policy")
    base_quality, base_reference_resize = next(iter(base_request_policies))

    def poster(job):
        source, relative = job
        target = output / relative
        if not target.exists():
            subprocess.run([str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", "0.2",
                "-i", str(source), "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "5", str(target)], check=True)
        return relative, {"file": relative, "bytes": target.stat().st_size, "sha256": sha(target),
                          "treatment": "480px JPEG preview extracted at 0.2s;original MP4 unchanged"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for relative, item in pool.map(poster, poster_jobs):
            assets[relative] = item
    settings = {"hardware": "NVIDIA GB200", "gpu_count": args.gpu_count,
        "nodes": 2, "gpus_per_node": 4, "hardware_proxy": "B200 proxy; B200 was not separately measured",
        "canvas": "1344x768", "fps": 24, "target_seconds": [5, 10, 15], "measured_seeds": [0, 1, 2],
        "warmup_seed": 42, "warmup_requests_per_profile": 1, "concurrency": 1,
        "gallery_seed": args.seed, "latency": "client end-to-end to completed MP4;startup excluded",
        "aggregation": "arithmetic mean and sample standard deviation across three measured seeds",
        "speedup": "Base mean seconds / FastH3 Omni mean seconds; no normalization",
        "base_denoiser_evaluations": 50, "fast_denoiser_evaluations": 8,
        "reference_sizing": "enlarged reference image short edge 2048; native released preprocessing",
        "media": "H.264 video and AAC stereo 32kHz audio",
        "base": {"request_quality": base_quality,
                 "reference_resize_mode": base_reference_resize},
        "comparison_scope": "model-and-serving-system comparison; not an isolated distillation or kernel ablation",
        "quality_scope": "matching inputs and seeds do not imply numerical or perceptual output parity"}
    payload = {"schema_version": 1, "settings": settings, "profiles": profiles}
    (output / "data/benchmark.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (output / "data/latency.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["composition", "target_seconds", "gpu_count", "base_mean_s", "base_sample_std_s",
                         "fasth3_omni_mean_s", "fasth3_omni_sample_std_s", "base_divided_by_fasth3_omni"])
        for row in profiles:
            perf = row["performance"]
            writer.writerow([row["composition"], row["target_seconds"], args.gpu_count,
                perf["base"]["mean_s"], perf["base"]["sample_std_s"], perf["fast"]["mean_s"],
                perf["fast"]["sample_std_s"], perf["speedup"]])
    (output / "data/media-manifest.json").unlink(missing_ok=True)
    for seconds in [5, 10, 15]:
        (output / f"img/latency-{seconds}s.svg").unlink(missing_ok=True)

    from plot_omniref_results import render_charts
    render_charts(payload, output / "img")
    print(json.dumps({"profiles": len(profiles),
        "output_videos": 81, "copied_original_bytes": sum(x["bytes"] for x in assets.values() if x["treatment"] == "original bytes"),
        "speedup_min": min(x["performance"]["speedup"] for x in profiles),
        "speedup_max": max(x["performance"]["speedup"] for x in profiles)}, indent=2))


if __name__ == "__main__":
    main()
