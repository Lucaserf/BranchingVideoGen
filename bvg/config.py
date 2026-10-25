"""Loading render settings, profiles and per-shot costs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "profiles.yaml"

# Render types from cheapest to most expensive. The planner upgrades shots along this order.
RENDER_ORDER = ("pan", "standard", "hero")


@dataclass(frozen=True)
class Profile:
    name: str
    target_wait_minutes: float
    max_hero_shots: int


@dataclass(frozen=True)
class Costs:
    """GPU seconds per unit of work."""

    keyframe: float
    shot: dict[str, float]  # render type -> seconds per shot
    finish: float


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_profile(config: dict, name: str, **overrides) -> Profile:
    if name not in config["profiles"]:
        raise KeyError(f"unknown profile {name!r}, expected one of {sorted(config['profiles'])}")
    values = {**config["profiles"][name], **{k: v for k, v in overrides.items() if v is not None}}
    return Profile(name=name, target_wait_minutes=float(values["target_wait_minutes"]),
                   max_hero_shots=int(values["max_hero_shots"]))


def load_costs(config: dict, bench_path: str | Path | None = None) -> Costs:
    """Estimated costs from the config, replaced by benchmark measurements where present."""
    keyframe = float(config["keyframe"]["est_seconds_per_image"])
    shot = {name: float(rt["est_seconds_per_shot"]) for name, rt in config["render_types"].items()}
    finish = float(config["finish"]["est_seconds"])

    if bench_path is not None:
        with open(bench_path) as f:
            bench = json.load(f)
        results = bench.get("results", {})
        if "keyframe" in results:
            keyframe = results["keyframe"]["mean_seconds"]
        for name in shot:
            if name in results:
                shot[name] = results[name]["mean_seconds"]

    return Costs(keyframe=keyframe, shot=shot, finish=finish)
