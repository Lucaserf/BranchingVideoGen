# BranchingVideoGen

A web app where a viewer plays a branching hero's journey in near real time and receives a personalized, AI-generated ~3 minute film at the end, rendered on a single NVIDIA L40.

The design and the options considered are in the project doc "Branching story video app: options".

## Status

A playable demo: an LLM writes each viewer's story, the L40 renders it while they play, with narration, music and an optional photo of the protagonist. Tested end to end on the L40 with templates; the LLM writer is tested with a fake endpoint so far.

## Demo

A playable end-to-end demo: pick a look, make six choices, and watch the film develop while you play.

```bash
pip install -e ".[server]"
python -m bvg.server --mock                          # any machine: placeholder stills and pans
python -m bvg.server --host 0.0.0.0 --preload        # on the L40, real models
```

Open http://localhost:8000 (on the server, forward the port: `ssh -L 8000:localhost:8000 <server>`). To play without a browser and get a film at the end: `python -m bvg.demo_play --shots 18`.

Before starting, the viewer can also:

- **Upload a photo of the protagonist.** Every keyframe that shows the hero is then drawn by an image-edit model (FLUX.1 Kontext by default, Qwen-Image-Edit as an Apache-2.0 alternative) from that photo, so the hero keeps that face. The app asks the viewer to confirm the photo is of them or that they have the person's permission.
- **Pick the sound.** A narration voice (Kokoro, a small TTS model that runs on the CPU while the GPU renders video), music, and captions, each optional. Music comes from files in `assets/music/` (mp3, wav, ogg, flac, m4a), or a soft synthesized pad when the folder is empty. It dips while the narrator speaks.
- **Pick the models** for stills, video, quality shots and the photo edit, from the catalog in `configs/profiles.yaml`. Planning uses each model's `est_seconds`; the app shows a memory forecast for the picks (see Memory below).

Install the narration voice with `pip install -e ".[audio]"`.

How it works:

- **Story** (`bvg/story.py`, `bvg/writer.py`): the six beats of the hero's journey are the fixed skeleton, the same for every viewer so runs can be compared. Everything inside them is written as the viewer plays: an LLM writes each act's scene, question and three choices from the answers so far (the viewer can also write their own), then the act's shot prompts and narration. It keeps a short cast list (visual descriptions of the hero, places and recurring characters) and repeats it in every image prompt so they stay recognizable. Each choice also moves a four-axis value fingerprint (courage, empathy, curiosity, order); the LLM scores the options it writes and any free-text answer.
- **Server** (`bvg/server.py`): FastAPI plus one GPU worker thread. One viewer holds the GPU at a time (released when the film is done or after 15 idle minutes). When an act is locked its keyframes render first, so stills appear within seconds, then its video shots. The planner decides at the start which shots are video and which are pans.
- **Film** (`bvg/engine.py`): every shot is cropped to 768x512 at 24 fps with its narration burned in as a caption, then the shots are joined with ffmpeg.

Choices made for the demo (all easy to change):

| Decision | Choice | Why |
| --- | --- | --- |
| Keyframes | FLUX.1-schnell, 768x512 | 1.3 s per image on the L40 |
| Video | LTX-Video 0.9.7 distilled | 20 s per 5 s shot, the only measured model fast enough for every profile |
| Hero shots | Effectively off | Wan 2.2 A14B took 1455 s per shot; the planner only fits 2 in `quality` |
| GPU sharing | One model in memory at a time, jobs grouped by model | FLUX (34 GB) and LTX (36 GB) together don't fit in 62 GB of RAM next to anything else |
| Story text | A local LLM (Qwen3 8B through Ollama) writes text, choices and prompts | Every film differs in more than the six answers; a plain fallback writer takes over if the LLM fails |
| Sound | Kokoro narration on the CPU, music files or a synthesized pad | Keeps the GPU free for video; no music model yet |
| Character consistency | Only with an uploaded photo | Without a photo each shot is drawn from the prompt alone, so the hero changes look between shots |
| Film length | The viewer picks the number of shots (6 to 72, default 36), each one image prompt and 5 s of film | Short films make testing faster; shots are split evenly across the six beats, extras go to the climax first |

### The story writer

The writer talks to any OpenAI-compatible chat endpoint, set in the `story:` section of `configs/profiles.yaml`. The default is Ollama on the same machine:

```bash
curl -fsSL https://ollama.com/install.sh | sh   # or a user-level install, see ollama.com
ollama pull qwen3:8b
python -m bvg.server --host 0.0.0.0             # uses story.base_url, http://127.0.0.1:11434/v1
```

- **Any model or server** works: change `model` and `base_url` (vLLM, llama.cpp server, LM Studio, or a hosted API with its key in `BVG_LLM_API_KEY`). Uncensored community fine-tunes from Ollama or Hugging Face fit here too.
- **GPU memory**: an 8B model takes about 6 GB of VRAM next to the video models. To keep the GPU entirely for video, run it on the CPU with `extra_body: {options: {num_gpu: 0}}` (slower: tens of seconds per act).
- **Timing**: after each choice the writer makes two calls on its own thread, the next act's text (what the viewer waits for) and the shot prompts for the act just chosen (what the GPU waits for).
- **Without an LLM**: `--writer simple` uses plain placeholder text and is the default with `--mock`; `python -m bvg.server --mock --writer llm` tests the real writer with placeholder images. If an LLM call fails, the simple writer covers that step and the premiere shows it.

## Memory

With `server.residency: sequential` (the default) the server holds one model at a time: it loads a model, renders every ready job that needs it, frees it from the GPU and RAM, then loads the next. The worker picks jobs to keep the loaded model busy (a video shot whose keyframe is ready runs before a keyframe that would need a switch), so a film usually costs one switch per act. Models are freed when a film finishes. `residency: swap` keeps all of a film's models in RAM instead, for faster switches when RAM allows.

- **Live monitor**: the app's header shows GPU and RAM in use, the loaded model, and any LLM Ollama is holding (`GET /api/memory`). The server logs a warning when free RAM drops under `server.memory_reserve_gb`, and logs memory after every model load.
- **Forecast**: as soon as models, length or profile change, the app shows the peak GPU memory and RAM the film will need (`POST /api/memory/forecast`), from each catalog entry's `weights_gb` (RAM to hold it) and `vram_gb` (GPU peak). If a model won't fit, it says which and why, and with `server.block_on_overflow` the server refuses to start that film. `python -m bvg.memory --model hero=wan2.2-i2v-a14b` prints the same from the command line.
- The LLM writer counts against the GPU when `story.device: gpu`, or against RAM with `cpu`. Qwen3.8 27B (18 GB) next to LTX (38.5 GB) doesn't fit on the 48 GB GPU, for example; run it on the CPU or pick a smaller writer.

Footprints marked `measured` in the catalog come from the L40 benchmark; the rest are estimates from the weight sizes, so treat a close call as a maybe.

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

The story writer can be abliterated too: these LLMs have the "refusal direction" removed from their weights, so they write dark or violent turns the viewer asks for instead of refusing. The catalog has the strongest ones that run on the L40 (from [huihui_ai on Ollama](https://ollama.com/huihui_ai)); pick one per film in the app or set `story.model`:

| Model | Size | Notes |
| --- | --- | --- |
| `qwen3-8b-abliterated` (default) | ~5 GB | Fits on the GPU next to the video models |
| `gemma4-12b-abliterated` | ~9 GB | Better prose, tight next to the video models |
| `qwen3.6-35b-a3b-abliterated` | ~22 GB | Mixture of experts, 3B active: usable on the CPU |
| `gemma4-26b-a4b-abliterated` | ~18 GB | Mixture of experts, 4B active: usable on the CPU |
| `qwen3.8-27b-abliterated` | ~18 GB | The strongest writer here; dense, so it needs GPU room |
| `qwen3-8b` | ~5 GB | The stock model, for comparison |

For images, `qwen-image-abliterated` and `qwen-image-edit-2509-abliterated` are Qwen-Image and Qwen-Image-Edit with an abliterated copy of their text encoder (Qwen2.5-VL 7B), set through the entry's `text_encoder` key. FLUX and the video models use T5 text encoders, which are not refusal-tuned, so there is nothing to abliterate there.

Two limits stay whatever the model: the writer's instructions forbid sexual content involving minors and sexual or nude depictions of a viewer's uploaded photo. With an abliterated writer that instruction is the only guard, so add a classifier on prompts before opening the app to the public.

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
