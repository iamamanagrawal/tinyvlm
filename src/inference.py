"""
    Inference script for the visual language model.
"""

import torch
from transformers import AutoTokenizer, SiglipProcessor

from src.model import VisionLanguageModel
from src.utils import TOKENS, IMAGE_PROMPT_TEMPLATE

def inference(
        model: VisionLanguageModel,
        conversations: dict,
        vision_processor: SiglipProcessor,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 50,
    ) -> str:
    
    ## preprocess the image
    image_path = conversations["image"]
    pixel_values = vision_processor(images=image_path, return_tensors="pt").pixel_values

    ## prepare the prompt
    prompt = ""
    for turn in conversations["conversations"]:
        prompt += f"{TOKENS['im_start']}{turn['role']}\n{turn['content']}\n{TOKENS['im_end']}\n"
    prompt += f"{TOKENS['im_start']}assistant\n"
    prompt = prompt.replace("<image>", IMAGE_PROMPT_TEMPLATE)

    ## tokenize the prompt
    tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    ## generate text
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text