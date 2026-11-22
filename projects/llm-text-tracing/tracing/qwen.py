"""Frozen Qwen2.5 representations; model loading is isolated from the CPU baseline."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
WINDOW = 64


class QwenEncoder:
    def __init__(self, model_dir=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        default_dir = Path(__file__).resolve().parents[1] / "models/Qwen2.5-0.5B"
        if model_dir is None and default_dir.is_dir():
            model_dir = default_dir
        source = str(model_dir) if model_dir is not None else MODEL_ID
        options = {"local_files_only": True} if model_dir is not None else {"revision": REVISION}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **options)
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(source, dtype=self.dtype, **options)
        self.model.requires_grad_(False).to(self.device).eval()

    @torch.inference_mode()
    def pool(self, windows):
        """Mean hidden states over tokens. An absent window has a zero vector."""
        indices = [i for i, tokens in enumerate(windows) if tokens]
        pooled = np.zeros((len(windows), self.model.config.hidden_size), dtype=np.float32)
        if indices:
            batch = self.tokenizer.pad({"input_ids": [windows[i] for i in indices]},
                                       padding=True, return_tensors="pt").to(self.device)
            hidden = self.model(**batch, use_cache=False).last_hidden_state.float()
            weights = batch["attention_mask"].unsqueeze(-1)
            pooled[indices] = ((hidden * weights).sum(1) / weights.sum(1)).cpu().numpy()
        return pooled

    def encode(self, texts, batch_size=12):
        if batch_size < 1:
            raise ValueError("batch_size 必须为正数")
        batches = []
        for start in range(0, len(texts), batch_size):
            ids = self.tokenizer(texts[start:start + batch_size], add_special_tokens=False,
                                 verbose=False)["input_ids"]
            windows = []
            for seq in ids:
                if not seq:
                    raise ValueError("分词后文本为空")
                windows.extend([seq[:WINDOW], seq[-WINDOW:]])
            batches.append(self.pool(windows).reshape(len(ids), 2, -1))
        return np.concatenate(batches)
