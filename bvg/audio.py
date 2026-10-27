"""Narration voice and music bed for the film.

Narration uses Kokoro (an 82M-parameter TTS that runs on the CPU, so it never
competes with the video models for the GPU). Music comes from audio files in
`audio.music_dir`, or a soft synthesized pad when there are none.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from bvg import MODEL_LOAD_LOCK
from bvg.video import ffmpeg_exe

SAMPLE_RATE = 48000
KOKORO_RATE = 24000

# Kokoro voices; the first letter of the id is its language (a: US English, b: UK English).
VOICES = {
    "af_heart": "Heart (US, female)",
    "af_bella": "Bella (US, female)",
    "am_michael": "Michael (US, male)",
    "am_fenrir": "Fenrir (US, male)",
    "bf_emma": "Emma (UK, female)",
    "bm_george": "George (UK, male)",
}
MUSIC_EXTENSIONS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(audio) == 0:
        return audio.astype(np.float32)
    n = round(len(audio) * dst / src)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


class Narrator:
    """Text to speech at SAMPLE_RATE. `mock=True` returns soft tones instead of a voice."""

    def __init__(self, backend: str = "kokoro", mock: bool = False):
        self.backend = backend
        self.mock = mock
        self._pipelines: dict = {}

    def speak(self, text: str, voice: str) -> np.ndarray:
        if self.mock or self.backend == "none":
            return self._tones(text)
        lang = voice[0]
        if lang not in self._pipelines:
            with MODEL_LOAD_LOCK:
                from kokoro import KPipeline

                self._pipelines[lang] = KPipeline(lang_code=lang, device="cpu")
        parts = [np.asarray(r.audio, dtype=np.float32) for r in self._pipelines[lang](text, voice=voice)
                 if r.audio is not None]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return _resample(audio, KOKORO_RATE, SAMPLE_RATE)

    @staticmethod
    def _tones(text: str) -> np.ndarray:
        """Placeholder speech: one short soft tone per word."""
        out = []
        for i, _ in enumerate(text.split()):
            t = np.arange(int(0.22 * SAMPLE_RATE)) / SAMPLE_RATE
            tone = 0.2 * np.sin(2 * np.pi * (220 + 30 * (i % 4)) * t) * np.hanning(len(t))
            out += [tone, np.zeros(int(0.05 * SAMPLE_RATE))]
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def decode(path: Path) -> np.ndarray:
    """Any audio file to mono float32 at SAMPLE_RATE, through ffmpeg."""
    cmd = [ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype=np.float32)


def music_files(music_dir: str | Path) -> list[Path]:
    d = Path(music_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in MUSIC_EXTENSIONS) if d.is_dir() else []


def synth_pad(seconds: float, seed: int = 0) -> np.ndarray:
    """A slow, quiet chord progression, used when there is no music file."""
    rng = np.random.default_rng(seed)
    roots = [57, 53, 60, 55]  # A minor, F, C, G (MIDI)
    rng.shuffle(roots[1:])
    chord_len = 6.0
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    out = np.zeros(n, np.float32)
    for k in range(int(np.ceil(seconds / chord_len))):
        root = roots[k % len(roots)]
        start, end = int(k * chord_len * SAMPLE_RATE), min(n, int((k + 1) * chord_len * SAMPLE_RATE + SAMPLE_RATE))
        seg = t[start:end] - t[start]
        env = np.minimum(1, seg / 1.5) * np.minimum(1, (seg[-1] - seg) / 1.5 + 1e-3)
        minor = k % len(roots) == 0
        for interval in (0, 3 if minor else 4, 7, 12):
            f = 440 * 2 ** ((root + interval - 69) / 12)
            out[start:end] += (0.06 * np.sin(2 * np.pi * f * seg) * env).astype(np.float32)
    return out


def music_bed(seconds: float, music_dir: str | Path, seed: int = 0) -> np.ndarray:
    files = music_files(music_dir)
    if not files:
        track = synth_pad(seconds, seed)
    else:
        track = decode(files[seed % len(files)])
        n = int(seconds * SAMPLE_RATE)
        track = np.tile(track, int(np.ceil(n / max(1, len(track)))))[:n]
    fade = min(len(track), 2 * SAMPLE_RATE)
    track = track.copy()
    track[:fade] *= np.linspace(0, 1, fade)
    track[-fade:] *= np.linspace(1, 0, fade)
    peak = np.abs(track).max() or 1
    return track / peak


def mix(seconds: float, lines: list[tuple[float, np.ndarray]], music: np.ndarray | None,
        music_volume: float = 0.25) -> np.ndarray:
    """Narration lines placed at their start times over the music, which dips while someone speaks."""
    n = int(seconds * SAMPLE_RATE)
    voice = np.zeros(n, np.float32)
    for start, audio in lines:
        i = int(start * SAMPLE_RATE)
        chunk = audio[: max(0, n - i)]
        voice[i:i + len(chunk)] += chunk
    peak = np.abs(voice).max()
    if peak > 0:
        voice *= 0.9 / peak
    out = voice
    if music is not None:
        music = np.resize(music, n) if len(music) != n else music
        speaking = np.convolve((np.abs(voice) > 0.01).astype(np.float32), np.ones(SAMPLE_RATE // 4) / (SAMPLE_RATE // 4),
                               mode="same")
        duck = 1 - 0.6 * np.clip(speaking * 4, 0, 1)
        out = voice + music * music_volume * duck
    return np.clip(out, -1, 1)


def write_wav(audio: np.ndarray, path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
