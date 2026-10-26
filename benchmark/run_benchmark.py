"""Measure per-shot GPU cost for keyframes and each render type on this machine.

    python -m benchmark.run_benchmark --shots 3 --out results/bench.json --save-videos outputs/bench

Each render type is loaded, warmed up with one untimed shot, timed over --shots shots,
then unloaded before the next one, so peak VRAM is reported per type. Feed the output
to the planner with `python -m bvg.planner --bench results/bench.json`.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from bvg.config import DEFAULT_CONFIG, load_config
from bvg.models import apply_overrides, stage_settings
from bvg.renderers import make_shot_renderer
from bvg.video import write_mp4

# Hero's journey style prompts covering the kinds of shots a film needs.
PROMPTS = [
    "a young courier stands on a rooftop at dawn, wind in her coat, city of copper towers behind her, cinematic",
    "a hooded stranger offers an old map across a candlelit tavern table, close-up on the hands, cinematic",
    "two travelers cross a rope bridge over a misty canyon, wide shot, camera slowly tracking, cinematic",
    "a giant mechanical owl opens its eyes in a ruined library, dust in the light beams, cinematic",
    "the hero raises a glowing lantern against a storm on a cliff, dramatic low angle, cinematic",
]


def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def _sync_and_reset(torch) -> None:
    if torch:
        torch.cuda.synchronize()


def _timed(torch, fn):
    _sync_and_reset(torch)
    start = time.perf_counter()
    out = fn()
    _sync_and_reset(torch)
    return out, time.perf_counter() - start


def _placeholder_keyframe(width: int, height: int, seed: int) -> Image.Image:
    """Used for video-only runs (--types without keyframe)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(width, height)
    c = rng.random((3, 3))
    img = np.stack([c[i, 0] + c[i, 1] * x + c[i, 2] * y for i in range(3)], axis=-1)
    return Image.fromarray((np.clip(img / img.max(), 0, 1) * 255).astype(np.uint8))


def _summary(times: list[float], load_seconds: float, torch) -> dict:
    result = {
        "mean_seconds": statistics.mean(times),
        "max_seconds": max(times),
        "min_seconds": min(times),
        "samples": len(times),
        "load_seconds": load_seconds,
    }
    if torch:
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--types", default="keyframe,pan,standard,hero",
                        help="comma-separated: keyframe and/or render types from the config")
    parser.add_argument("--shots", type=int, default=3, help="timed samples per type (after one warmup)")
    parser.add_argument("--out", default="results/bench.json")
    parser.add_argument("--save-videos", default=None, help="directory to write the rendered shots to")
    parser.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL",
                        help="pick a catalog model for a stage, e.g. --model standard=wan2.2-ti2v-5b (repeatable)")
    parser.add_argument("--permissive-only", action="store_true", default=None,
                        help="refuse models whose license has usage restrictions")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.model, args.permissive_only)
    shot_seconds = config["film"]["shot_seconds"]
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    torch = _cuda()
    save_dir = Path(args.save_videos) if args.save_videos else None
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch else None,
        "shot_seconds": shot_seconds,
        "config": args.config,
        "models": {},
        "results": {},
    }

    def save():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))

    n = args.shots + 1  # first one is the warmup
    kf = stage_settings(config, "keyframe")
    keyframes = [_placeholder_keyframe(kf["width"], kf["height"], i) for i in range(n)]

    if "keyframe" in types:
        from bvg.renderers.diffusers_backend import KeyframeRenderer

        print(f"[keyframe] loading {kf['model_id']}")
        if torch:
            torch.cuda.reset_peak_memory_stats()
        renderer, load_s = _timed(torch, lambda: KeyframeRenderer(kf))
        times = []
        for i in range(n):
            img, t = _timed(torch, lambda i=i: renderer.render(PROMPTS[i % len(PROMPTS)], seed=i))
            keyframes[i] = img
            if i > 0:
                times.append(t)
            print(f"[keyframe] image {i}{' (warmup)' if i == 0 else ''}: {t:.1f}s")
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                img.save(save_dir / f"keyframe_{i}.png")
        report["results"]["keyframe"] = _summary(times, load_s, torch)
        report["models"]["keyframe"] = kf.get("model_name")
        renderer.unload()
        save()

    for name in types:
        if name == "keyframe":
            continue
        settings = stage_settings(config, name)
        print(f"[{name}] loading {settings.get('model_id', settings['backend'])}")
        if torch:
            torch.cuda.reset_peak_memory_stats()
        renderer, load_s = _timed(torch, lambda: make_shot_renderer(settings))
        times = []
        for i in range(n):
            frames, t = _timed(torch, lambda i=i: renderer.render(keyframes[i], PROMPTS[i % len(PROMPTS)],
                                                                    shot_seconds, seed=i))
            if i > 0:
                times.append(t)
            print(f"[{name}] shot {i}{' (warmup)' if i == 0 else ''}: {t:.1f}s, {len(frames)} frames")
            if save_dir:
                write_mp4(frames, save_dir / f"{name}_{i}.mp4", settings["fps"])
        report["results"][name] = _summary(times, load_s, torch)
        report["results"][name]["frames_per_shot"] = len(frames)
        report["models"][name] = settings.get("model_name")
        renderer.unload()
        save()

    print(json.dumps(report["results"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
