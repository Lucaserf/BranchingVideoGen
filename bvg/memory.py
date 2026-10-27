"""Memory on the render server: what is free now, and what a film's models will need.

`snapshot()` reads host RAM (from /proc/meminfo) and GPU memory (from torch, or
nvidia-smi when torch has not touched the GPU yet), plus any LLM that Ollama is
holding. `forecast()` predicts the peak RAM and GPU memory of a film from the
catalog's `weights_gb` (what a model takes in RAM) and `vram_gb` (its peak on the
GPU as configured), for the server's residency mode:

- sequential: one model at a time. Load it, render every job that needs it, free
  it, load the next. Peak = the largest single model.
- swap: every model the film uses stays in RAM and only the one rendering is on
  the GPU. RAM = all of them together.

    python -m bvg.memory                      # what is free now
    python -m bvg.memory --model hero=wan2.2-i2v-a14b
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import logging
import shutil
import subprocess
import threading
import time
import urllib.request

log = logging.getLogger("bvg.memory")

GB = 1024 ** 3


def _meminfo() -> dict[str, float]:
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                out[key] = int(value.split()[0]) * 1024 / GB
    except OSError:
        pass
    return out


def _process_rss() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / GB
    except OSError:
        pass
    return None


def _gpu() -> dict | None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            return {"name": torch.cuda.get_device_name(0), "total_gb": total / GB, "free_gb": free / GB,
                    "process_gb": torch.cuda.memory_reserved() / GB}
    except ImportError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            line = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits",
                                   "-i", "0"], capture_output=True, text=True, timeout=5).stdout.strip()
            name, total, free = (x.strip() for x in line.split(","))
            return {"name": name, "total_gb": float(total) / 1024, "free_gb": float(free) / 1024, "process_gb": None}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def _ollama(base_url: str | None) -> list[dict]:
    """Models Ollama has loaded right now, with how much of each sits on the GPU."""
    if not base_url or "11434" not in base_url:
        return []
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/ps", timeout=2) as res:
            models = json.load(res).get("models", [])
    except (OSError, ValueError):
        return []
    return [{"name": m.get("name"), "size_gb": m.get("size", 0) / GB, "vram_gb": m.get("size_vram", 0) / GB}
            for m in models]


def snapshot(llm_url: str | None = None) -> dict:
    info = _meminfo()
    ram = None
    if "MemTotal" in info:
        ram = {"total_gb": info["MemTotal"], "free_gb": info.get("MemAvailable", info.get("MemFree", 0)),
               "process_gb": _process_rss()}
    return {"time": time.time(), "ram": ram, "gpu": _gpu(), "llm": _ollama(llm_url)}


def release() -> None:
    """Return freed memory to the OS: Python garbage, the CUDA cache, and glibc's free lists."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# --- forecast ---------------------------------------------------------------

def footprint(settings: dict) -> dict:
    """RAM and GPU memory one stage's model takes, from its catalog entry."""
    weights = float(settings.get("weights_gb") or settings.get("vram_gb") or 0)
    vram = float(settings.get("vram_gb") or weights)
    offload = bool(settings.get("cpu_offload"))
    return {"model": settings.get("model_name"), "weights_gb": weights, "vram_gb": vram, "offload": offload,
            # Loading goes through host RAM; with CPU offload the whole model stays there while it renders.
            "ram_gb": weights}


def forecast(settings: dict[str, dict], residency: str, snap: dict, *, used_stages: list[str],
             reserve_gb: float = 4.0, llm_device: str = "gpu", held_ram_gb: float = 0.0) -> dict:
    """Predicted peak memory for a film, and whether it fits.

    `settings` maps stage -> stage settings (from bvg.models.stage_settings); only
    `used_stages` count. `held_ram_gb` is RAM the server already holds for models
    that will be freed first, so it counts as available.
    """
    models: dict[str, dict] = {}
    for stage in used_stages:
        s = settings.get(stage) or {}
        if s.get("model_name") and s.get("backend") != "llm":
            entry = models.setdefault(s["model_name"], {**footprint(s), "stages": []})
            entry["stages"].append(stage)
    items = list(models.values())

    llm = settings.get("story") or {}
    llm_gb = float(llm.get("vram_gb") or 0) if llm.get("writer", "llm") != "simple" else 0.0
    llm_vram = llm_gb if llm_device == "gpu" else 0.0
    llm_ram = llm_gb if llm_device == "cpu" else 0.0

    if residency == "sequential":
        peak_vram = max((m["vram_gb"] for m in items), default=0.0)
        peak_ram = max((m["ram_gb"] for m in items), default=0.0)
    else:  # swap: all in RAM, one on the GPU
        peak_vram = max((m["vram_gb"] for m in items), default=0.0)
        peak_ram = sum(m["ram_gb"] for m in items)
    peak_vram += llm_vram
    peak_ram += llm_ram

    gpu, ram = snap.get("gpu"), snap.get("ram")
    vram_budget = gpu["total_gb"] - reserve_gb if gpu else None
    ram_budget = ram["free_gb"] + held_ram_gb - reserve_gb if ram else None
    warnings = []
    for m in items:
        need = m["vram_gb"] + llm_vram
        if vram_budget is not None and need > vram_budget:
            warnings.append(f"{m['model']} needs about {need:.0f} GB on the GPU"
                            f"{' with the LLM' if llm_vram else ''}, more than the {vram_budget:.0f} GB available"
                            + ("" if m["offload"] else "; turn on cpu_offload for it or pick a smaller model"))
    if ram_budget is not None and peak_ram > ram_budget:
        what = "the largest model" if residency == "sequential" else "all the film's models together"
        warnings.append(f"{what} needs about {peak_ram:.0f} GB of RAM, more than the {ram_budget:.0f} GB free")
    return {"residency": residency, "models": items,
            "llm": {"model": llm.get("model_name"), "gb": llm_gb, "device": llm_device} if llm_gb else None,
            "peak_vram_gb": round(peak_vram, 1), "peak_ram_gb": round(peak_ram, 1),
            "vram_budget_gb": None if vram_budget is None else round(vram_budget, 1),
            "ram_budget_gb": None if ram_budget is None else round(ram_budget, 1),
            "fits": not warnings, "warnings": warnings}


class Monitor:
    """Samples memory every few seconds, keeps the latest and the low points, and warns when it runs low."""

    def __init__(self, interval: float = 5.0, warn_free_gb: float = 4.0, llm_url: str | None = None):
        self.interval, self.warn_free_gb, self.llm_url = interval, warn_free_gb, llm_url
        self.latest = snapshot(llm_url)
        self.low = {"ram_free_gb": None, "gpu_free_gb": None}
        self._warned = False

    def start(self) -> "Monitor":
        threading.Thread(target=self._run, daemon=True, name="memory-monitor").start()
        return self

    def sample(self) -> dict:
        snap = snapshot(self.llm_url)
        self.latest = snap
        for key, part in (("ram_free_gb", snap["ram"]), ("gpu_free_gb", snap["gpu"])):
            if part:
                low = self.low[key]
                self.low[key] = part["free_gb"] if low is None else min(low, part["free_gb"])
        ram = snap["ram"]
        if ram and ram["free_gb"] < self.warn_free_gb:
            if not self._warned:
                log.warning("host RAM is low: %.1f GB free of %.1f GB", ram["free_gb"], ram["total_gb"])
            self._warned = True
        else:
            self._warned = False
        return snap

    def _run(self) -> None:
        while True:
            try:
                self.sample()
            except Exception:
                log.exception("memory sample failed")
            time.sleep(self.interval)


def describe(snap: dict) -> str:
    parts = []
    if snap.get("gpu"):
        g = snap["gpu"]
        parts.append(f"GPU {g['total_gb'] - g['free_gb']:.1f}/{g['total_gb']:.0f} GB used")
    if snap.get("ram"):
        r = snap["ram"]
        parts.append(f"RAM {r['total_gb'] - r['free_gb']:.1f}/{r['total_gb']:.0f} GB used")
    for m in snap.get("llm") or []:
        parts.append(f"LLM {m['name']} {m['size_gb']:.1f} GB ({m['vram_gb']:.1f} on GPU)")
    return ", ".join(parts) or "no memory information"


def main() -> None:
    from bvg.config import load_config
    from bvg.models import apply_overrides, stage_settings

    parser = argparse.ArgumentParser(description="Show free memory and the forecast for the configured models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL")
    parser.add_argument("--photo", action="store_true", help="the film uses a photo of the protagonist")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config) if args.config else load_config(), args.model)
    server = config.get("server", {})
    snap = snapshot(config.get("story", {}).get("base_url"))
    print(describe(snap))
    stages = ["keyframe", "standard", "hero", "story"] + (["photo"] if args.photo else [])
    settings = {s: stage_settings(config, s) for s in stages}
    result = forecast(settings, server.get("residency", "sequential"), snap, used_stages=stages,
                      reserve_gb=server.get("memory_reserve_gb", 4), llm_device=config["story"].get("device", "gpu"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
