import os

from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        # 先根据原始版本字符串判断是否为 CLIP
        is_clip = version.startswith("openai/clip-vit-large-patch14") or version.startswith("openai")

        # 在本机上优先复用已下载的本地模型目录，避免访问 huggingface.co
        local_override = None
        if version.startswith("google/t5-v1_1-xxl"):
            local_override = os.getenv("FLUX_T5_LOCAL_PATH")
        elif version.startswith("openai/clip-vit-large-patch14"):
            local_override = os.getenv("FLUX_CLIP_LOCAL_PATH")

        if local_override is not None and os.path.isdir(local_override):
            model_id = local_override
            hf_kwargs.setdefault("local_files_only", True)
        else:
            model_id = version

        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_id, max_length=max_length, **hf_kwargs)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(model_id, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_id, max_length=max_length, **hf_kwargs)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(model_id, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
