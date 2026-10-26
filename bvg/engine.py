"""The render engine the server drives: keyframes, shots and the finished film.

`mock=True` swaps the GPU models for placeholders (a drawn keyframe and a pan), so
the whole app can be tried on a laptop.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bvg.models import stage_settings
from bvg.renderers import make_shot_renderer
from bvg.renderers.pan import PanRenderer
from bvg.video import ffmpeg_exe, write_mp4


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize to cover width x height, then center-crop."""
    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def resample(frames: list, src_fps: int, dst_fps: int, seconds: float) -> list:
    """Pick frames so the shot plays `seconds` long at dst_fps."""
    n = round(seconds * dst_fps)
    return [frames[min(len(frames) - 1, int(i * src_fps / dst_fps))] for i in range(n)]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def caption(frame: Image.Image, text: str) -> Image.Image:
    """Subtitle at the bottom of the frame, on a translucent band."""
    frame = frame.convert("RGBA")
    size = max(14, frame.height // 22)
    font = _font(size)
    lines = textwrap.wrap(text, width=max(20, frame.width // (size // 2 + 1)))
    band_h = (size + 6) * len(lines) + 16
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, frame.height - band_h, frame.width, frame.height), fill=(0, 0, 0, 140))
    y = frame.height - band_h + 8
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((frame.width - w) / 2, y), line, font=font, fill=(255, 255, 255, 255))
        y += size + 6
    return Image.alpha_composite(frame, overlay).convert("RGB")


def placeholder_keyframe(prompt: str, width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(width, height)
    c = rng.random((3, 3))
    arr = np.stack([c[i, 0] + c[i, 1] * x + c[i, 2] * y for i in range(3)], axis=-1)
    img = Image.fromarray((np.clip(arr / arr.max(), 0, 1) * 220).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((24, 24), "\n".join(textwrap.wrap(prompt, 48)), font=_font(height // 24), fill="white")
    return img


class Engine:
    def __init__(self, config: dict, mock: bool = False):
        self.config = config
        self.mock = mock
        film = config["film"]
        self.width, self.height, self.fps = film["width"], film["height"], film["fps"]
        self.shot_seconds = film["shot_seconds"]
        self.offload_all = config.get("server", {}).get("offload_all", True)
        self._keyframe = None
        self._shot_renderers: dict = {}

    def _settings(self, stage: str) -> dict:
        settings = stage_settings(self.config, stage)
        if self.offload_all and settings.get("backend") != "pan":
            settings["cpu_offload"] = True
        return settings

    def load(self, render_types: set[str]) -> None:
        """Load the keyframe model and the shot models a session needs, once."""
        if self.mock:
            return
        if self._keyframe is None:
            from bvg.renderers.diffusers_backend import KeyframeRenderer

            self._keyframe = KeyframeRenderer(self._settings("keyframe"))
        for rt in render_types:
            if rt not in self._shot_renderers:
                self._shot_renderers[rt] = make_shot_renderer(self._settings(rt))

    def keyframe(self, prompt: str, seed: int) -> Image.Image:
        if self.mock:
            return placeholder_keyframe(prompt, self.width, self.height, seed)
        return fit(self._keyframe.render(prompt, seed=seed), self.width, self.height)

    def shot(self, render_type: str, image: Image.Image, prompt: str, narration: str, seed: int, out: Path) -> None:
        if self.mock or render_type == "pan":
            settings = {**self.config["render_types"]["pan"], "width": self.width, "height": self.height,
                        "fps": self.fps}
            renderer, src_fps = PanRenderer(settings), self.fps
        else:
            renderer = self._shot_renderers[render_type]
            src_fps = renderer.fps
        frames = renderer.render(image, prompt, self.shot_seconds, seed=seed)
        frames = [f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f)) for f in frames]
        frames = resample(frames, src_fps, self.fps, self.shot_seconds)
        frames = [caption(fit(f, self.width, self.height), narration) for f in frames]
        write_mp4(frames, out, self.fps)

    def stitch(self, shot_paths: list[Path], out: Path) -> None:
        """Every shot was written with the same size, fps and encoder, so the streams can be copied."""
        listing = out.with_suffix(".txt")
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in shot_paths))
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
               "-c", "copy", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
