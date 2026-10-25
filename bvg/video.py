"""Writing frames to mp4 through ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def write_mp4(frames: list[Image.Image] | list[np.ndarray], path: str | Path, fps: int) -> None:
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        arr = np.asarray(frame.convert("RGB") if isinstance(frame, Image.Image) else frame)
        if arr.dtype != np.uint8:
            # diffusers returns floats in [0, 1] when output_type is np
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
