"""Keyframe (text-to-image, image edit) and shot (image-to-video) renderers built on diffusers pipelines.

Model ids, pipeline classes and call arguments all come from configs/profiles.yaml, so
swapping a model is a config change. torch and diffusers are imported lazily so the
planner and pan renderer work without the GPU stack.
"""

from __future__ import annotations

from PIL import Image

from bvg import MODEL_LOAD_LOCK

# Weight-only quantization (bitsandbytes) to fit big models on the GPU: about a quarter
# (nf4) or half (int8) of the bf16 size, somewhat slower per step.
QUANT_BACKENDS = {"nf4": "bitsandbytes_4bit", "int8": "bitsandbytes_8bit"}


def _quant_kwargs(mode: str, dtype) -> dict:
    if mode == "nf4":
        return {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype}
    if mode == "int8":
        return {"load_in_8bit": True}
    raise ValueError(f"unknown quantize mode {mode!r}, expected one of {sorted(QUANT_BACKENDS)}")


def _load_pipeline(settings: dict):
    with MODEL_LOAD_LOCK:
        return _load_pipeline_unlocked(settings)


def _load_pipeline_unlocked(settings: dict):
    import diffusers
    import torch

    pipeline_cls = getattr(diffusers, settings["pipeline"])
    dtype = getattr(torch, settings.get("dtype", "bfloat16"))
    quant = settings.get("quantize")  # {"mode": "nf4", "components": ["transformer", "text_encoder"]}
    quantized = set(quant.get("components", ["transformer"])) if quant else set()
    components, kwargs = {}, {}
    encoder = settings.get("text_encoder")
    if encoder:  # e.g. an abliterated copy of the model's own text encoder
        import transformers

        encoder_cls = getattr(transformers, encoder.get("class", "AutoModel"))
        extra = {}
        if "text_encoder" in quantized:
            extra["quantization_config"] = transformers.BitsAndBytesConfig(**_quant_kwargs(quant["mode"], dtype))
            quantized.discard("text_encoder")
        components["text_encoder"] = encoder_cls.from_pretrained(encoder["model_id"], torch_dtype=dtype, **extra)
    if quantized:
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=QUANT_BACKENDS[quant["mode"]], quant_kwargs=_quant_kwargs(quant["mode"], dtype),
            components_to_quantize=sorted(quantized))
    pipe = pipeline_cls.from_pretrained(settings["model_id"], torch_dtype=dtype, **components, **kwargs)

    lora = settings.get("lora")
    if lora:
        pipe.load_lora_weights(lora["repo"], weight_name=lora.get("weight_name"), adapter_name="distill")
        pipe.set_adapters(["distill"], adapter_weights=[lora.get("scale", 1.0)])

    if settings.get("vae_tiling") and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if settings.get("cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(settings.get("device", "cuda"))
    return pipe


def _generator(seed: int):
    import torch

    return torch.Generator(device="cuda").manual_seed(seed)


def _unload(renderer) -> None:
    """Free a renderer's pipeline from GPU and host memory."""
    from bvg import memory

    pipe, renderer.pipe = renderer.pipe, None
    if pipe is None:
        return
    if hasattr(pipe, "remove_all_hooks"):  # CPU offload hooks hold references to the weights
        pipe.remove_all_hooks()
    for name in list(getattr(pipe, "components", {})):
        try:
            setattr(pipe, name, None)
        except Exception:  # a pipeline that won't drop a component still frees it when collected
            pass
    del pipe
    memory.release()


def frames_for(seconds: float, fps: int, multiple: int) -> int:
    """Video models want num_frames = k * multiple + 1."""
    k = max(1, round(seconds * fps / multiple))
    return k * multiple + 1


class KeyframeRenderer:
    def __init__(self, settings: dict):
        self.settings = settings
        self.pipe = _load_pipeline(settings)

    def render(self, prompt: str, seed: int = 0) -> Image.Image:
        s = self.settings
        guidance = {s.get("guidance_arg", "guidance_scale"): s["guidance"]}
        return self.pipe(prompt=prompt, width=s["width"], height=s["height"], num_inference_steps=s["steps"],
                         generator=_generator(seed), **guidance, **s.get("extra_kwargs", {})).images[0]

    def unload(self) -> None:
        _unload(self)


class ImageToVideoRenderer:
    def __init__(self, settings: dict):
        self.settings = settings
        self.fps = settings["fps"]
        self.pipe = _load_pipeline(settings)

    def render(self, image: Image.Image, prompt: str, seconds: float, seed: int = 0) -> list[Image.Image]:
        s = self.settings
        image = image.convert("RGB").resize((s["width"], s["height"]), Image.Resampling.LANCZOS)
        kwargs = dict(prompt=prompt, negative_prompt=s.get("negative_prompt"), width=s["width"],
                      height=s["height"], num_frames=frames_for(seconds, s["fps"], s["frame_multiple"]),
                      num_inference_steps=s["steps"], guidance_scale=s["guidance"], generator=_generator(seed),
                      output_type="pil",  # Wan defaults to float numpy frames
                      **s.get("extra_kwargs", {}))
        if s.get("call_style") == "ltx_condition":
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

            kwargs["conditions"] = [LTXVideoCondition(image=image, frame_index=0)]
        else:
            kwargs["image"] = image
        return self.pipe(**kwargs).frames[0]

    def unload(self) -> None:
        _unload(self)


class EditRenderer:
    """Image edit (e.g. FLUX.1 Kontext, Qwen-Image-Edit): draws a scene around a reference photo."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.pipe = _load_pipeline(settings)

    def render(self, reference: Image.Image, prompt: str, seed: int = 0) -> Image.Image:
        s = self.settings
        guidance = {s.get("guidance_arg", "guidance_scale"): s["guidance"]}
        return self.pipe(image=reference.convert("RGB"), prompt=prompt, width=s["width"], height=s["height"],
                         num_inference_steps=s["steps"], generator=_generator(seed), **guidance,
                         **s.get("extra_kwargs", {})).images[0]

    def unload(self) -> None:
        _unload(self)
