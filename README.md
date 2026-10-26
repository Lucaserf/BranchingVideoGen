# BranchingVideoGen

A web app where a viewer plays a branching hero's journey in near real time and receives a personalized, AI-generated ~3 minute film at the end, rendered on a single NVIDIA L40.

The design and the options considered are in the project doc "Branching story video app: options".

## Status

This first piece measures what the L40 can actually render and plans each film around a target waiting time. The story engine, GPU server and web app come next.

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
- Wan 2.2 I2V A14B is about 126 GB of weights, so check free disk before running `hero`. `--model hero=wan2.2-ti2v-5b` is a smaller stand-in.
- LTX-Video distilled peaks at 38.5 GB, so only one video model fits in VRAM at a time; the benchmark unloads each type before the next.

Use `--types keyframe,standard` to run a subset. Results are written after each type, so a crash in one model keeps the earlier numbers. The hero model defaults to 40 steps with CPU offload, which is slow. To try a 4-step distillation LoRA, fill in `lora` under `hero` in the config and set `steps: 4`, `guidance: 1.0`.

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
