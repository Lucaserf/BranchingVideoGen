"""Pan shots: a slow zoom and drift over a still keyframe."""

from __future__ import annotations

import math

from PIL import Image


class PanRenderer:
    def __init__(self, settings: dict):
        self.width = settings["width"]
        self.height = settings["height"]
        self.fps = settings["fps"]
        self.zoom = settings.get("zoom", 1.15)

    def render(self, image: Image.Image, prompt: str, seconds: float, seed: int = 0) -> list[Image.Image]:
        # Cover the output frame, then crop a window that shrinks (zooms in) over the shot.
        scale = max(self.width / image.width, self.height / image.height) * self.zoom
        base = image.convert("RGB").resize((round(image.width * scale), round(image.height * scale)),
                                            Image.Resampling.LANCZOS)
        # Drift direction depends on the seed so consecutive pans don't all move the same way.
        angle = (seed % 8) * math.pi / 4
        n = max(1, round(seconds * self.fps))
        frames = []
        for f in range(n):
            t = f / max(1, n - 1)
            t = t * t * (3 - 2 * t)  # ease in and out
            z = 1 + (self.zoom - 1) * t
            w, h = base.width / z, base.height / z
            # Keep the window inside the image while drifting from the center.
            slack_x, slack_y = (base.width - w) / 2, (base.height - h) / 2
            cx = base.width / 2 + math.cos(angle) * slack_x * t
            cy = base.height / 2 + math.sin(angle) * slack_y * t
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            frames.append(base.resize((self.width, self.height), Image.Resampling.BILINEAR, box=box))
        return frames

    def unload(self) -> None:
        pass
