"""
This module contains utility constants and configurations for the TinyVLM project.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipVisionModel,
    SiglipProcessor,
)


## Token information
NUM_IMAGE_TOKENS = 49
TOKENS = {
    "EOT": "<|endoftext|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}
SPECIAL_TOKENS = {
    "img_start": "<|img_start|>",
    "img_end": "<|img_end|>",
    "img_token": "<|img_token|>",
    "pad_token": "<|pad|>",
}

## Prompt Template
IMAGE_PROMPT_TEMPLATE = f"{SPECIAL_TOKENS['img_start']}{SPECIAL_TOKENS['img_token'] * NUM_IMAGE_TOKENS}{SPECIAL_TOKENS['img_end']}"

## Dataset local and remote paths
LOCAL_DATASET_PATH = "data/LLaVA-CC3M-Pretrain-595K"

## Model paths
VISION_ENCODER_PATH = "models/siglip-base-patch16-224"
LANGUAGE_MODEL_PATH = "models/SmolLM2-135M-Instruct"
PROJECTOR_PATH = "models/projection_checkpoint.pth"

## Example Prompt
EXAMPLE_PROMPT = {
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe the image."},
    ],
    "image": "example.jpg"
}


def get_language_model_and_tokenizer(path: str, special_tokens: dict):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(special_tokens.values())}
    )
    tokenizer.pad_token = special_tokens['pad_token']
    tokenizer.image_token = special_tokens['img_token']
    tokenizer.img_start_token = special_tokens['img_start']
    tokenizer.img_end_token = special_tokens['img_end']

    language_model = AutoModelForCausalLM.from_pretrained(path)
    language_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, language_model

def get_vision_processor_and_model(path):
    vision_processor = SiglipProcessor.from_pretrained(path, use_fast=True)
    vision_model = SiglipVisionModel.from_pretrained(path)
    return vision_processor, vision_model