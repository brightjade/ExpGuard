import re
from abc import ABC

import torch
from transformers import AutoTokenizer
from vllm import LLM

from src.classifier_models.base import (
    SafetyClassifierOutput,
    SafetyClassifierBase,
    PromptHarmfulness,
    ResponseHarmfulness,
)
from src.generation_utils import subprocess_inference_with_vllm, inference_with_vllm


class ExpGuardBase(SafetyClassifierBase, ABC):
    def get_required_input_fields(self) -> list[str]:
        return ["prompt"]

    def get_optional_input_fields(self) -> list[str]:
        return ["response"]

    def get_output_fields(self) -> list[str]:
        return ["prompt_harmfulness", "response_harmfulness"]
    
    def parse_model_generation_to_output(self, output: str) -> SafetyClassifierOutput:
        # First try to match both prompt and response assessments
        m = re.match("User Safety Assessment: (.*)(?:\nAgent Safety Assessment: (.*))?", output)
        if m is None:
            return SafetyClassifierOutput(is_parsing_error=True)
            
        prompt_harmfulness = m.group(1)
        response_harmfulness = m.group(2) if m.group(2) is not None else None
        
        is_parse_error = False
        is_prompt_harmful = None
        is_response_harmful = None

        if prompt_harmfulness.strip().lower() == "safe":
            is_prompt_harmful = PromptHarmfulness.UNHARMFUL
        elif prompt_harmfulness.strip().lower() == "unsafe":
            is_prompt_harmful = PromptHarmfulness.HARMFUL
        else:
            is_parse_error = True
        
        if response_harmfulness is not None:
            if response_harmfulness.strip().lower() == "safe":
                is_response_harmful = ResponseHarmfulness.UNHARMFUL
            elif response_harmfulness.strip().lower() == "unsafe":
                is_response_harmful = ResponseHarmfulness.HARMFUL
            else:
                is_parse_error = True

        return SafetyClassifierOutput(
            is_parsing_error=is_parse_error,
            prompt_harmfulness=is_prompt_harmful,
            response_harmfulness=is_response_harmful,
        )

    def build_input_prompts(self, batch: list[dict[str, str]]) -> list[str]:
        inputs = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response"]}] if item["response"] else \
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            ) for item in batch
        ]
        return inputs


class ExpGuard(ExpGuardBase):
    DEFAULT_MODEL_ID = "6rightjade/expguard-7b"

    def __init__(self, batch_size: int = -1, ephemeral_model: bool = False, model_id: str = None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.lora_path = None
        if ephemeral_model:
            self.model = None
        else:
            self.model = LLM(model=self.model_id, enable_lora=self.lora_path is not None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @torch.inference_mode()
    def _classify_batch(self, batch: list[dict[str, str]]) -> list[SafetyClassifierOutput]:
        formatted_prompts = self.build_input_prompts(batch)
        if self.model is None:
            decoded_outputs = subprocess_inference_with_vllm(
                prompts=formatted_prompts,
                model_name_or_path=self.model_id,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_tqdm=True,
                lora_path=self.lora_path
            )
        else:
            decoded_outputs = inference_with_vllm(
                prompts=formatted_prompts,
                model=self.model,
                model_name_or_path=self.model_id,
                max_tokens=128,
                temperature=0.0,
                top_p=1.0,
                use_tqdm=True,
                lora_path=self.lora_path
            )
        outputs = [self.parse_model_generation_to_output(output) for output in decoded_outputs]
        return outputs
