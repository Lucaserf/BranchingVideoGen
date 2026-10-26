"""Choose a render type for every shot so the film is ready within the target wait.

The GPU renders shots one at a time, in act order. Act N's shots can start once the
viewer locks act N, so earlier acts render while the viewer is still playing. The
film is done when the last shot and the finishing pass (stitch, mix, upscale) are
done, and that must happen within `target_wait_minutes` of the last choice.

The planner starts with every shot as a pan, then upgrades shots in priority order,
first to `standard` and then to `hero`, keeping each upgrade only if the schedule
still fits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from bvg.config import RENDER_ORDER, Costs, Profile, load_config, load_costs, load_profile


@dataclass(frozen=True)
class Shot:
    act: int  # 0-based
    index: int  # 0-based position within the act
    priority: float = 0.5  # 0..1, higher gets upgraded first (set by the story LLM)
    render_type: str = "pan"


@dataclass(frozen=True)
class Plan:
    shots: tuple[Shot, ...]
    finish_seconds_after_last_choice: float
    target_seconds: float

    @property
    def fits(self) -> bool:
        return self.finish_seconds_after_last_choice <= self.target_seconds

    def counts(self) -> dict[str, int]:
        return {rt: sum(s.render_type == rt for s in self.shots) for rt in RENDER_ORDER}


def lock_times(acts: int, seconds_per_choice: float) -> list[float]:
    """Seconds from the start of play at which each act is locked."""
    return [(a + 1) * seconds_per_choice for a in range(acts)]


def finish_after_last_choice(shots: list[Shot], costs: Costs, locks: list[float]) -> float:
    """Simulate the single-GPU queue and return when the film is done, relative to the last lock."""
    clock = 0.0
    for shot in sorted(shots, key=lambda s: (s.act, s.index)):
        clock = max(clock, locks[shot.act])
        clock += costs.keyframe + costs.shot[shot.render_type]
    clock += costs.finish
    return clock - locks[-1]


def plan_film(shots: list[Shot], costs: Costs, profile: Profile, seconds_per_choice: float) -> Plan:
    acts = max(s.act for s in shots) + 1
    locks = lock_times(acts, seconds_per_choice)
    target = profile.target_wait_minutes * 60
    current = [replace(s, render_type="pan") for s in shots]
    by_priority = sorted(range(len(current)), key=lambda i: (-current[i].priority, current[i].act, current[i].index))

    hero_used = 0
    for render_type in RENDER_ORDER[1:]:
        for i in by_priority:
            if render_type == "hero" and hero_used >= profile.max_hero_shots:
                break
            candidate = current.copy()
            candidate[i] = replace(current[i], render_type=render_type)
            if finish_after_last_choice(candidate, costs, locks) <= target:
                current = candidate
                hero_used += render_type == "hero"

    return Plan(shots=tuple(current), finish_seconds_after_last_choice=finish_after_last_choice(current, costs, locks),
                target_seconds=target)


def default_shots(config: dict) -> list[Shot]:
    """A placeholder shot list: the first and last shot of each act matter most."""
    film = config["film"]
    n = film["shots_per_act"]
    return [Shot(act=a, index=i, priority=0.8 if i in (0, n - 1) else 0.5)
            for a in range(film["acts"]) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show how a profile splits the film's shots across render types.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--bench", default=None, help="benchmark results JSON with measured costs")
    parser.add_argument("--target-wait-minutes", type=float, default=None)
    parser.add_argument("--max-hero-shots", type=int, default=None)
    parser.add_argument("--seconds-per-choice", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    profile = load_profile(config, args.profile, target_wait_minutes=args.target_wait_minutes,
                           max_hero_shots=args.max_hero_shots)
    costs = load_costs(config, args.bench)
    seconds_per_choice = args.seconds_per_choice or config["film"]["seconds_per_choice"]
    plan = plan_film(default_shots(config), costs, profile, seconds_per_choice)

    print(f"profile {profile.name}: target {profile.target_wait_minutes:g} min after the last choice")
    print(f"costs (s): keyframe {costs.keyframe:g}, " + ", ".join(f"{k} {v:g}" for k, v in costs.shot.items())
          + f", finish {costs.finish:g}")
    print("shots: " + ", ".join(f"{n} {rt}" for rt, n in plan.counts().items()))
    status = "fits" if plan.fits else "does NOT fit even with all pans"
    print(f"film ready {plan.finish_seconds_after_last_choice / 60:.1f} min after the last choice ({status})")
    for act in range(config["film"]["acts"]):
        row = [s.render_type[0].upper() for s in plan.shots if s.act == act]
        print(f"  act {act + 1}: {' '.join(row)}")


if __name__ == "__main__":
    main()
