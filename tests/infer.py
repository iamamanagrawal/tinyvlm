"""
    Test inference script for the visual language model.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import inference
from src.model import VisionLanguageModel
from src.utils import (
    PROJECTOR_PATH,
    get_vision_processor_and_model,
    get_language_model_and_tokenizer,
    LANGUAGE_MODEL_PATH,
    VISION_ENCODER_PATH,
    SPECIAL_TOKENS,
)

def main():
    ## load the vision processor and model
    vision_processor, vision_encoder = get_vision_processor_and_model(VISION_ENCODER_PATH)

    ## load the language model and tokenizer
    tokenizer, language_model = get_language_model_and_tokenizer(
        LANGUAGE_MODEL_PATH,
        SPECIAL_TOKENS
    )

    ## initialize the vision-language model
    model = VisionLanguageModel(
        vision_encoder=vision_encoder,
        language_model=language_model,
        image_token_id=tokenizer(SPECIAL_TOKENS['img_token']).input_ids[0],
    )

    model = model.from_pretrained_projector(PROJECTOR_PATH)
    
    for image_path in os.listdir("examples/"):
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        conversation = {
            "conversations": [
                {"role": "human", "content": "<image>\n\n\nDescribe the image."},
            ],
            "image": os.path.join("examples/", image_path)
        }

        ## perform inference
        generated_text = inference(
            model=model,
            conversations=conversation,
            vision_processor=vision_processor,
            tokenizer=tokenizer,
            max_new_tokens=50,
        )

        print(f"=== Inference Example {image_path} ===")
        print(f"Image Path: {image_path}")
        print(f"Generated Text: {generated_text}")
        print("=========================")

    return None

if __name__ == "__main__":
    main()