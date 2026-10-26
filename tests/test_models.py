import pytest

from bvg.config import load_config
from bvg.models import ModelPolicyError, apply_overrides, stage_settings


def test_stage_settings_merge_catalog_and_stage_keys():
    s = stage_settings(load_config(), "hero")
    assert s["model_name"] == "wan2.2-i2v-a14b"
    assert s["pipeline"] == "WanImageToVideoPipeline"
    assert s["backend"] == "diffusers_i2v"
    assert s["cpu_offload"] is True  # set on the stage, not in the catalog


def test_pan_needs_no_model():
    assert stage_settings(load_config(), "pan")["backend"] == "pan"


def test_override_selects_another_model():
    config = apply_overrides(load_config(), ["standard=wan2.2-ti2v-5b", "keyframe=qwen-image"])
    assert stage_settings(config, "standard")["model_id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert stage_settings(config, "keyframe")["backend"] == "diffusers_t2i"


def test_permissive_only_rejects_restricted_models():
    config = apply_overrides(load_config(), [], permissive_only=True)
    with pytest.raises(ModelPolicyError):
        stage_settings(config, "standard")  # LTX-Video default
    assert stage_settings(config, "hero")["permissive"]


def test_wrong_kind_is_rejected():
    with pytest.raises(ValueError):
        stage_settings(apply_overrides(load_config(), ["keyframe=wan2.2-ti2v-5b"]), "keyframe")


def test_every_catalog_entry_is_complete():
    for name, entry in load_config()["models"].items():
        for key in ("kind", "license", "permissive", "pipeline", "model_id", "width", "height", "steps", "guidance"):
            assert key in entry, (name, key)
        if entry["kind"] == "i2v":
            assert {"fps", "frame_multiple", "call_style"} <= entry.keys(), name
