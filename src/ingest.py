"""
This script is for
   - creating a custom Dataset class for the llava-cc3m-pretrain dataset
   - building the text that model can interpret by building the chat_template function
   - a dataloader with collate function to handle variable seq lengths
"""

import json
import random
from transformers import AutoTokenizer, SiglipProcessor

from src.utils import TOKENS

def chat_template(data: dict, path: str, image_prompt_template: str) -> dict:
    """Apply chat template to the conversation with image tokens."""
    text = ""
    for message in data['conversations']:
        role = message["from"] if "from" in message else message['role']
        content = message["value"] if "value" in message else message['content']
        if role == "human":
            text += f"{TOKENS['im_start']}user\n{content}{TOKENS['im_end']}\n"
        elif role == "gpt":
            text += f"{TOKENS['im_start']}assistant\n{content}{TOKENS['im_end']}\n"
    text = text.replace("<image>", image_prompt_template)

    return {
        "text": text,
        "image": f"{path}/images/{data['image']}"
    }

def load_dataset(path: str, image_prompt_template: str) -> list:
    """Load dataset from a JSON file and apply chat template."""
    with open(f"{path}/chat.json", "r") as f:
        data = json.load(f)
    data = [chat_template(sample, path, image_prompt_template) for sample in data]
    return data


class DataLoaderLite:
    def __init__(
            self, 
            data: list, 
            tokenizer: AutoTokenizer,
            vision_processor: SiglipProcessor,
            batch_size: int = 16
        ) -> None:

        ## shuffle the data in-place
        self.data = data
        random.shuffle(self.data)

        ## init entities
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.batch_size = batch_size

        ## iterator
        self.index = 0
        self.num_batches = (len(data) + batch_size - 1) // batch_size
        self.response_token = f"{TOKENS['im_start']}assistant"
            

    def next_batch(self):
        if self.index >= len(self.data):
            self.index = 0  # reset for next epoch
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch
    
    def collate_fn(self, batch: list) -> dict:
        texts = [item['text'] for item in batch]
        images = [item['image'] for item in batch]

        tokenzied_texts = self.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenzied_texts.input_ids
        attention_mask = tokenzied_texts.attention_mask

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        parts = [text.split(self.response_token)[0] for text in texts]
        for i, part in enumerate(parts):
            part_ids = self.tokenizer(
                part,
                return_tensors='pt'
            ).input_ids
            labels[i, :part_ids.size(1)] = -100  # Mask out the prompt part

        pixel_values = self.vision_processor(images=images, return_tensors="pt").pixel_values
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }