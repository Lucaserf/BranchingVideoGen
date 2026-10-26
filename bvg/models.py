"""The model catalog: which model each stage uses, and whether its license is allowed.

    python -m bvg.models                     # list the catalog
    python -m bvg.models --permissive-only   # only models with no usage restrictions
"""

from __future__ import annotations

import argparse

from bvg.config import load_config

STAGE_KIND = {"keyframe": "t2i", "photo": "edit"}  # every render type with a model is i2v
TOP_LEVEL_STAGES = ("keyframe", "photo")
BACKEND = {"t2i": "diffusers_t2i", "i2v": "diffusers_i2v", "edit": "diffusers_edit"}


class ModelPolicyError(ValueError):
    pass


def apply_overrides(config: dict, overrides: list[str] | None, permissive_only: bool | None = None) -> dict:
    """Apply `stage=model` selections (e.g. from --model) and the permissive-only switch to the config."""
    for item in overrides or []:
        stage, _, model = item.partition("=")
        if not model:
            raise ValueError(f"expected stage=model, got {item!r}")
        select_model(config, stage, model)
    if permissive_only is not None:
        config.setdefault("model_policy", {})["permissive_only"] = permissive_only
    return config


def _section(config: dict, stage: str) -> dict:
    section = config.get(stage) if stage in TOP_LEVEL_STAGES else config["render_types"].get(stage)
    if section is None:
        raise KeyError(f"unknown stage {stage!r}")
    return section


def select_model(config: dict, stage: str, model: str) -> None:
    """Point a stage at a catalog model, taking the model's own time estimate for planning."""
    if model not in config.get("models", {}):
        raise KeyError(f"{stage}: unknown model {model!r}, expected one of {sorted(config.get('models', {}))}")
    section = _section(config, stage)
    if section.get("model") == model:
        return
    section["model"] = model
    est = config["models"][model].get("est_seconds")
    if est is not None:
        section["est_seconds_per_image" if stage == "keyframe" else "est_seconds_per_shot"] = est


def models_by_kind(config: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name, entry in config.get("models", {}).items():
        out.setdefault(entry["kind"], []).append({"name": name, "license": entry.get("license"),
                                                   "permissive": bool(entry.get("permissive")),
                                                   "est_seconds": entry.get("est_seconds")})
    return out


def stage_settings(config: dict, stage: str) -> dict:
    """Catalog defaults for the stage's model, with the stage's own keys layered on top."""
    section = _section(config, stage)
    name = section.get("model")
    if name is None:
        return dict(section)  # e.g. pan, which needs no model

    catalog = config.get("models", {})
    if name not in catalog:
        raise KeyError(f"{stage}: unknown model {name!r}, expected one of {sorted(catalog)}")
    entry = catalog[name]
    kind = STAGE_KIND.get(stage, "i2v")
    if entry["kind"] != kind:
        raise ValueError(f"{stage} needs a {kind} model, but {name!r} is {entry['kind']}")
    if config.get("model_policy", {}).get("permissive_only") and not entry.get("permissive"):
        raise ModelPolicyError(f"{stage}: {name!r} is not permissively licensed ({entry.get('license')}); "
                               "pick another model or turn off model_policy.permissive_only")

    return {"backend": BACKEND[kind], **entry, **section, "model_name": name}


def main() -> None:
    parser = argparse.ArgumentParser(description="List the model catalog and what each stage uses.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--permissive-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    in_use = {s: config["render_types"][s].get("model") for s in config["render_types"]}
    in_use["keyframe"] = config["keyframe"].get("model")
    for name, entry in config["models"].items():
        if args.permissive_only and not entry.get("permissive"):
            continue
        stages = ", ".join(s for s, m in in_use.items() if m == name)
        mark = "permissive" if entry.get("permissive") else "restricted"
        print(f"{name:24} {entry['kind']}  {mark:10}  {entry['license']}" + (f"  [used by {stages}]" if stages else ""))


if __name__ == "__main__":
    main()
