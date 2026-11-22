"""Memory-bounded Qwen prefill for long inputs on Apple silicon, using the same weights."""
import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from tracing.qwen import MODEL_ID, REVISION


def load_weights(local, rope):
    from pathlib import Path
    folder = local or Path(snapshot_download(MODEL_ID, revision=REVISION,
                                             allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"]))
    model, _ = load_model(folder, lazy=True, model_config={"rope_scaling": dict(rope), "max_position_embeddings": 65536})
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    mx.clear_cache()
    return model


def prefill(model, tokens):
    cache = make_prompt_cache(model)
    # Only the last prompt position needs vocabulary logits. Materialize each KV block
    # before moving on, so earlier computation graphs do not accumulate in unified memory.
    for start in range(0, len(tokens) - 1, 512):
        block = tokens[start:min(start + 512, len(tokens) - 1)]
        model(mx.array([block]), cache=cache)
        mx.eval([layer.state for layer in cache])
        mx.clear_cache()

    def advance(token):
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(logits)
        return logits

    logits = advance(tokens[-1])
    return logits, advance, int(cache[0].offset)
