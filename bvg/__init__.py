import threading

# transformers resolves its classes lazily and that isn't thread-safe: the render
# worker loading a diffusers pipeline while the writer thread loads Kokoro failed with
# "cannot import name 'Qwen2_5_VLForConditionalGeneration'". Hold this while loading either.
MODEL_LOAD_LOCK = threading.Lock()
