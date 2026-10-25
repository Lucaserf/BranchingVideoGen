import shutil

import pytest
from PIL import Image

from bvg.renderers.pan import PanRenderer
from bvg.renderers.diffusers_backend import frames_for
from bvg.video import write_mp4


def test_pan_frame_count_and_size():
    frames = PanRenderer({"width": 320, "height": 192, "fps": 24}).render(Image.new("RGB", (640, 480), "red"), "", 2)
    assert len(frames) == 48
    assert all(f.size == (320, 192) for f in frames)


def test_frames_for_matches_model_constraints():
    assert frames_for(5, 24, 8) == 121  # LTX: 8k + 1
    assert frames_for(5, 16, 4) == 81  # Wan: 4k + 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_write_mp4(tmp_path):
    frames = PanRenderer({"width": 64, "height": 64, "fps": 8}).render(Image.new("RGB", (100, 80), "blue"), "", 1)
    out = tmp_path / "pan.mp4"
    write_mp4(frames, out, 8)
    assert out.stat().st_size > 0
