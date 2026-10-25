from __future__ import annotations


def make_shot_renderer(settings: dict):
    if settings["backend"] == "pan":
        from bvg.renderers.pan import PanRenderer

        return PanRenderer(settings)
    if settings["backend"] == "diffusers_i2v":
        from bvg.renderers.diffusers_backend import ImageToVideoRenderer

        return ImageToVideoRenderer(settings)
    raise ValueError(f"unknown shot backend {settings['backend']!r}")
