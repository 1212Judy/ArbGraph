# local_llm/hf_adapter.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class HFAdapter:
    """
    Minimal HuggingFace-based LLM adapter.

    Provides a unified generate() interface for the ArbGraph pipeline.
    """

    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 400,
        temperature: float = 0.0,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
