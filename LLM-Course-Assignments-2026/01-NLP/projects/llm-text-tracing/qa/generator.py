"""Autoregressive, evidence-constrained generation with a shared KV cache."""
import math
from importlib.util import find_spec
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from tracing.data import ROOT
from tracing.qwen import MODEL_ID, REVISION

CONTEXT_LIMIT = 49152


class EvidenceGenerator:
    def __init__(self, model_dir=None, device=None):
        local = Path(model_dir) if model_dir else ROOT / "models/Qwen2.5-0.5B"
        self.local = local if local.is_dir() else None
        source = str(local) if self.local else MODEL_ID
        if model_dir and not local.is_dir():
            raise FileNotFoundError(local)
        options = {"local_files_only": True} if local.is_dir() else {"revision": REVISION}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **options)
        config = AutoConfig.from_pretrained(source, **options)
        original = config.max_position_embeddings
        config.rope_parameters = {**config.rope_parameters, "rope_type": "yarn", "factor": 2.0,
                                  "original_max_position_embeddings": original}
        config.max_position_embeddings = original * 2
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.config, self.source, self.options, self.dtype = config, source, options, dtype
        self.model, self.mlx_model = None, None
        self.lock = Lock()

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False, verbose=False)

    def _forward(self, tokens, cache):
        output = self.model(input_ids=torch.tensor([tokens], device=self.device),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
        return output.logits[0, -1]

    def backend_for(self, length):
        if length > 8192 and self.device.type == "mps" and find_spec("mlx_lm") is not None:
            return "mlx-metal"
        return "torch-" + self.device.type

    def _prefill(self, tokens):
        if self.backend_for(len(tokens)) == "mlx-metal":
            from .mlx_context import load_weights, prefill
            if self.mlx_model is None:
                self.mlx_model = load_weights(self.local, self.config.rope_parameters)
            return prefill(self.mlx_model, tokens)
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.source, config=self.config, dtype=self.dtype,
                                                             attn_implementation="sdpa", **self.options)
            self.model.requires_grad_(False).to(self.device).eval()
        cache = DynamicCache(config=self.config)
        for start in range(0, len(tokens), 256):
            logits = self._forward(tokens[start:start + 256], cache)
            if self.device.type == "mps" and len(tokens) > 8192 and start % 1024 == 0:
                torch.mps.synchronize()
                torch.mps.empty_cache()
        return logits, lambda token: self._forward([token], cache), int(cache.get_seq_length())

    @torch.inference_mode()
    def choose(self, prompt, passages):
        """Decode a quote. At branch points Qwen scores valid next tokens with a retrieval prior."""
        tokens = self.tokens(prompt)
        candidates = [(i, self.tokens(p["answer"])) for i, p in enumerate(passages)]
        if not candidates or any(not sequence for _, sequence in candidates):
            raise ValueError("候选证据不能为空")
        if len(tokens) + max(len(seq) for _, seq in candidates) + 1 > CONTEXT_LIMIT:
            raise ValueError(f"问题、对话、材料与答案合计超过 {CONTEXT_LIMIT:,} 个 token，请缩短补充材料。")
        if not tokens:
            raise ValueError("问题上下文不能为空")
        # A fresh cache per call keeps different users and conversations independent.
        with self.lock:
            logits, advance, seen = self._prefill(tokens)
            if seen != len(tokens):
                raise RuntimeError("模型缓存未覆盖完整提示")
            eos = self.tokenizer.eos_token_id
            prefix = []
            while len(candidates) > 1:
                branches = {}
                for i, seq in candidates:
                    token = seq[len(prefix)] if len(prefix) < len(seq) else eos
                    branches.setdefault(token, []).append((i, seq))
                if len(branches) == 1:
                    token = next(iter(branches))
                else:
                    choices = list(branches)
                    values = logits[choices].tolist()
                    def prior(t):
                        return max(math.log(max(passages[i]["score"], 1e-6)) for i, _ in branches[t])
                    token = max(zip(choices, values), key=lambda pair: pair[1] + 8 * prior(pair[0]))[0]
                candidates = branches[token]
                if token == eos:
                    break
                prefix.append(token)
                if len(candidates) > 1:
                    logits = advance(token)
        index, sequence = candidates[0]
        # Once only one branch remains, its suffix is forced by the decoding constraint.
        quote = self.tokenizer.decode(prefix + sequence[len(prefix):], clean_up_tokenization_spaces=False)
        if quote != passages[index]["answer"]:
            raise RuntimeError("模型输出与候选原文不一致")
        return index, quote, seen
