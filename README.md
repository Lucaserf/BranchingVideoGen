# BranchingVideoGen

A web app where a viewer plays a branching hero's journey in near real time and receives a personalized, AI-generated ~3 minute film at the end, rendered on a single NVIDIA L40.

The design and the options considered are in the project doc "Branching story video app: options".

## Status

This first piece measures what the L40 can actually render and plans each film around a target waiting time. The story engine, GPU server and web app come next.

## Demo

A playable end-to-end demo: pick a look, make six choices, and watch the film develop while you play.

```bash
pip install -e ".[server]"
python -m bvg.server --mock                          # any machine: placeholder stills and pans
python -m bvg.server --host 0.0.0.0 --preload        # on the L40, real models
```

Open http://localhost:8000 (on the server, forward the port: `ssh -L 8000:localhost:8000 <server>`). To play without a browser and get a film at the end: `python -m bvg.demo_play --length short`.

Before starting, the viewer can also:

- **Upload a photo of the protagonist.** Every keyframe that shows the hero is then drawn by an image-edit model (FLUX.1 Kontext by default, Qwen-Image-Edit as an Apache-2.0 alternative) from that photo, so the hero keeps that face. The app asks the viewer to confirm the photo is of them or that they have the person's permission.
- **Pick the sound.** A narration voice (Kokoro, a small TTS model that runs on the CPU while the GPU renders video), music, and captions, each optional. Music comes from files in `assets/music/` (mp3, wav, ogg, flac, m4a), or a soft synthesized pad when the folder is empty. It dips while the narrator speaks.
- **Pick the models** for stills, video, quality shots and the photo edit, from the catalog in `configs/profiles.yaml`. Planning uses each model's `est_seconds`; a model loads the first time a film uses it, and up to `server.max_loaded_models` (default 3) stay in host RAM.

Install the narration voice with `pip install -e ".[audio]"`.

How it works:

- **Story** (`bvg/story.py`): six fixed acts of the hero's journey, three options per act plus "write your own". Each act has six shot templates that the choices fill in, with a narration line per shot. Each option also moves a four-axis value fingerprint (courage, empathy, curiosity, order).
- **Server** (`bvg/server.py`): FastAPI plus one GPU worker thread. One viewer holds the GPU at a time (released when the film is done or after 15 idle minutes). When an act is locked its keyframes render first, so stills appear within seconds, then its video shots. The planner decides at the start which shots are video and which are pans.
- **Film** (`bvg/engine.py`): every shot is cropped to 768x512 at 24 fps with its narration burned in as a caption, then the shots are joined with ffmpeg.

Choices made for the demo (all easy to change):

| Decision | Choice | Why |
| --- | --- | --- |
| Keyframes | FLUX.1-schnell, 768x512 | 1.3 s per image on the L40 |
| Video | LTX-Video 0.9.7 distilled | 20 s per 5 s shot, the only measured model fast enough for every profile |
| Hero shots | Effectively off | Wan 2.2 A14B took 1455 s per shot; the planner only fits 2 in `quality` |
| GPU sharing | Both models loaded with CPU offload | FLUX (32 GB) and LTX (38.5 GB) don't fit in 48 GB together, and reloading per act would take minutes |
| Story text | Templates, no LLM | Works offline and keeps the GPU for video; an LLM can rewrite shot prompts later |
| Sound | Kokoro narration on the CPU, music files or a synthesized pad | Keeps the GPU free for video; no music model yet |
| Character consistency | Only with an uploaded photo | Without a photo each shot is drawn from the prompt alone, so the hero changes look between shots |
| Film length | Full (36 shots, 3 min) or short (18 shots, 90 s) | Short makes testing faster |

## Layout

- `configs/profiles.yaml`: models, resolutions and steps for each render type, plus the `fast`, `balanced` and `quality` waiting-time profiles.
- `benchmark/run_benchmark.py`: renders a few shots per render type on the GPU and records seconds per shot and peak VRAM.
- `bvg/planner.py`: picks a render type (`pan`, `standard`, `hero`) for each of the 36 shots so the film is ready within `target_wait_minutes` of the viewer's last choice.
- `bvg/renderers/`: the pan renderer (camera move over a still) and diffusers-based keyframe and image-to-video renderers.

## Render types

| Type | Default model | What it is |
| --- | --- | --- |
| `pan` | none | Slow zoom and drift over the keyframe, about a second per shot |
| `standard` | LTX-Video 0.9.7 distilled | Fast image-to-video, used for most shots |
| `hero` | Wan 2.2 I2V A14B | Higher quality image-to-video for the most important shots |

Every shot also needs a keyframe from FLUX.1-schnell.

## Choosing models

`configs/profiles.yaml` has a `models` catalog. Each stage (`keyframe`, `standard`, `hero`) names a catalog model with `model:`, and any other key on the stage overrides the catalog defaults. Each entry records its license and whether it is `permissive` (commercial use with no field-of-use, revenue or territory limits).

```bash
python -m bvg.models                       # list the catalog and which stage uses what
python -m bvg.models --permissive-only
python -m benchmark.run_benchmark --model standard=wan2.2-ti2v-5b --model keyframe=qwen-image
python -m benchmark.run_benchmark --permissive-only   # refuses restricted models
```

Set `model_policy.permissive_only: true` in the config to enforce this everywhere. Adding a model is a new catalog entry: any diffusers pipeline class with its model id and call arguments. The license notes are a summary; check each model card before shipping.

### Uncensored and community models

The pipeline adds no content filter, safety checker or prompt rewriting: whatever the selected model generates is what ends up in the film. The open video models in the catalog (Wan, LTX-Video, HunyuanVideo) are not refusal-tuned; how far they go depends on their training data. To use a community fine-tune instead, add a catalog entry whose `model_id` is its Hugging Face repo or a local folder in diffusers format, or keep a base entry and set `lora` on the stage:

```yaml
models:
  my-finetune:
    kind: i2v
    license: see model card
    permissive: false
    pipeline: WanImageToVideoPipeline
    call_style: image
    model_id: /models/my-wan-finetune   # or an HF repo id
    # ...same fields as the base model it was trained from
```

Since viewers type free text into the story, the story layer (not yet built) should still block a small set of things no model choice changes: sexual content involving minors and realistic depictions of real, identifiable people.

## Running the benchmark on the L40

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
pip install -e ".[gpu,dev]"
huggingface-cli login   # FLUX.1-schnell and some video models need an accepted license

python -m benchmark.run_benchmark --shots 3 --out results/bench.json --save-videos outputs/bench
```

Notes from the first L40 run:

- The T5 tokenizer in recent transformers needs `sentencepiece` and `tiktoken` (both in the `gpu` extra).
- Without a system ffmpeg, the bundled `imageio-ffmpeg` binary is used.
- FLUX.1-schnell is gated: accept its license on Hugging Face and log in first.
- Wan 2.2 I2V A14B is about 126 GB of weights and needs about 70 GB of host RAM to load, so check free disk and RAM before running `hero`. `--model hero=wan2.2-ti2v-5b` is a smaller stand-in.
- LTX-Video distilled peaks at 38.5 GB, so only one video model fits in VRAM at a time; the benchmark unloads each type before the next.

Use `--types keyframe,standard` to run a subset. Results are written after each type, so a crash in one model keeps the earlier numbers. The hero model defaults to 40 steps with CPU offload, which measured 1455 s per shot on the L40. To try a 4-step distillation LoRA, fill in `lora` under `hero` in the config and set `steps: 4`, `guidance: 1.0`.

## Planning a film

```bash
python -m bvg.planner --profile balanced                        # with the estimates in the config
python -m bvg.planner --profile balanced --bench results/bench.json
python -m bvg.planner --profile fast --target-wait-minutes 5 --seconds-per-choice 60
```

The planner simulates the single-GPU queue: act N's shots start rendering once the viewer locks act N (every `seconds_per_choice` seconds), so most of the film renders during play. It starts from all pans and upgrades shots in priority order to `standard`, then to `hero` up to `max_hero_shots`, keeping each upgrade only if the film still finishes in time.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
