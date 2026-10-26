"""The render engine the server drives: keyframes, shots and the finished film.

`mock=True` swaps the GPU models for placeholders (a drawn keyframe and a pan), so
the whole app can be tried on a laptop.
"""

from __future__ import annotations

import subprocess
import textwrap
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def trim_letterbox(frames: list[Image.Image], threshold: int = 16) -> list[Image.Image]:
    """Crop black bars a model drew into a shot, judged on its middle frame."""
    rows = np.asarray(frames[len(frames) // 2].convert("L")).mean(axis=1)
    dark = rows < threshold
    top = int(np.argmin(dark)) if not dark.all() else 0
    bottom = int(np.argmin(dark[::-1])) if not dark.all() else 0
    h = len(rows)
    if top + bottom < 0.03 * h or top + bottom > 0.4 * h:
        return frames
    return [f.crop((0, top, f.width, f.height - bottom)) for f in frames]


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


def placeholder_keyframe(prompt: str, width: int, height: int, seed: int,
                         photo: Image.Image | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(width, height)
    c = rng.random((3, 3))
    arr = np.stack([c[i, 0] + c[i, 1] * x + c[i, 2] * y for i in range(3)], axis=-1)
    img = Image.fromarray((np.clip(arr / arr.max(), 0, 1) * 220).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((24, 24), "\n".join(textwrap.wrap(prompt, 48)), font=_font(height // 24), fill="white")
    if photo is not None:
        thumb = fit(photo, height // 3, height // 3)
        img.paste(thumb, (width - thumb.width - 24, height - thumb.height - 24))
    return img


class Engine:
    def __init__(self, config: dict, mock: bool = False):
        self.config = config
        self.mock = mock
        film = config["film"]
        self.width, self.height, self.fps = film["width"], film["height"], film["fps"]
        self.shot_seconds = film["shot_seconds"]
        server = config.get("server", {})
        self.offload_all = server.get("offload_all", True)
        self.swap = server.get("swap_models", False)
        # Pipelines kept loaded (in host RAM with swap_models), least recently used first.
        self.max_loaded = server.get("max_loaded_models", 3)
        self._loaded: OrderedDict[tuple, object] = OrderedDict()
        self._on_gpu = None

    def _prepare(self, settings: dict) -> dict:
        settings = dict(settings)
        if settings.get("cpu_offload"):
            return settings
        if self.swap:
            settings["device"] = "cpu"
        elif self.offload_all:
            settings["cpu_offload"] = True
        return settings

    def _renderer(self, settings: dict):
        """The loaded pipeline for these settings, loading it (and evicting the oldest) if needed."""
        key = (settings["backend"], settings["model_name"])
        if key in self._loaded:
            self._loaded.move_to_end(key)
            return self._loaded[key]
        while len(self._loaded) >= self.max_loaded:
            _, old = self._loaded.popitem(last=False)
            if old is self._on_gpu:
                self._on_gpu = None
            old.unload()
        from bvg.renderers.diffusers_backend import EditRenderer, KeyframeRenderer

        backend = settings["backend"]
        prepared = self._prepare(settings)
        if backend == "diffusers_t2i":
            renderer = KeyframeRenderer(prepared)
        elif backend == "diffusers_edit":
            renderer = EditRenderer(prepared)
        else:
            renderer = make_shot_renderer(prepared)
        self._loaded[key] = renderer
        return renderer

    def _to_gpu(self, renderer) -> None:
        """With swap_models on, keep one pipeline on the GPU and park the others in host RAM."""
        if not self.swap or renderer is self._on_gpu:
            return
        import torch

        if self._on_gpu is not None:
            self._on_gpu.pipe.to("cpu")
            self._on_gpu = None
            torch.cuda.empty_cache()
        if not renderer.settings.get("cpu_offload"):  # offloaded pipelines move their own weights
            renderer.pipe.to("cuda")
            self._on_gpu = renderer

    def preload(self, settings: list[dict]) -> None:
        if not self.mock:
            for s in settings:
                self._renderer(s)

    def keyframe(self, settings: dict, prompt: str, seed: int) -> Image.Image:
        if self.mock:
            return placeholder_keyframe(prompt, self.width, self.height, seed)
        renderer = self._renderer(settings)
        self._to_gpu(renderer)
        return fit(renderer.render(prompt, seed=seed), self.width, self.height)

    def keyframe_from_photo(self, settings: dict, photo: Image.Image, prompt: str, seed: int) -> Image.Image:
        if self.mock:
            return placeholder_keyframe(prompt, self.width, self.height, seed, photo=photo)
        renderer = self._renderer(settings)
        self._to_gpu(renderer)
        return fit(renderer.render(photo, prompt, seed=seed), self.width, self.height)

    def shot(self, settings: dict, image: Image.Image, prompt: str, narration: str | None, seed: int,
             out: Path) -> None:
        if self.mock or settings["backend"] == "pan":
            pan = {**self.config["render_types"]["pan"], "width": self.width, "height": self.height, "fps": self.fps}
            renderer, src_fps = PanRenderer(pan), self.fps
        else:
            renderer = self._renderer(settings)
            self._to_gpu(renderer)
            src_fps = renderer.fps
        frames = renderer.render(image, prompt, self.shot_seconds, seed=seed)
        frames = [f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f)) for f in frames]
        frames = trim_letterbox(resample(frames, src_fps, self.fps, self.shot_seconds))
        frames = [fit(f, self.width, self.height) for f in frames]
        if narration:
            frames = [caption(f, narration) for f in frames]
        write_mp4(frames, out, self.fps)

    def stitch(self, shot_paths: list[Path], out: Path) -> None:
        """Every shot was written with the same size, fps and encoder, so the streams can be copied."""
        listing = out.with_suffix(".txt")
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in shot_paths))
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
               "-c", "copy", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)

    def add_audio(self, video: Path, wav: Path, out: Path) -> None:
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-map", "0:v", "-map",
               "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
