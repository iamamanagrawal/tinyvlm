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

def chat_template(data: dict, path: str) -> dict:
    """Apply chat template to the conversation with image tokens."""

    assert len(data['conversations']) == 2, "Each sample must have exactly two conversation turns."
    assert data['conversations'][0]['from'] == 'human', "First turn must be from human."
    assert data['conversations'][1]['from'] == 'gpt', "Second turn must be from assistant."

    text = f"{TOKENS['im_start']}user\n{data['conversations'][0]['value']}\n{TOKENS['im_end']}\n{TOKENS['im_start']}assistant\n"
    label = f"{data['conversations'][1]['value']}\n{TOKENS['im_end']}"

    return {
        "text": text,
        "labels": label,
        "image": f"{path}/images/{data['image']}"
    }

def load_dataset(path: str) -> list:
    """Load dataset from a JSON file and apply chat template."""
    with open(f"{path}/chat.json", "r") as f:
        data = json.load(f)
    data = [chat_template(sample, path) for sample in data]
    return data


class DataLoaderLite:
    def __init__(
            self, 
            data: list, 
            tokenizer: AutoTokenizer,
            vision_processor: SiglipProcessor,
            batch_size: int
        ) -> None:
        """A lightweight dataloader with collate function for variable sequence lengths."""

        ## shuffle the data in-place
        self.data = data
        random.shuffle(self.data)

        ## init entities
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.batch_size = batch_size
        self.num_batches = (len(data) + batch_size - 1) // batch_size

        ## iterator
        self.index = 0            

    def next_batch(self):
        """Get the next batch of data."""
        if self.index + self.batch_size >= len(self.data):
            index = self.index
            self.index = 0
            return self.collate_fn(self.data[index:])
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return self.collate_fn(batch)
    
    def collate_fn(self, batch: list) -> dict:
        """Collate function to handle variable sequence lengths."""
        texts = [item['text'] for item in batch]
        labels = [item['labels'] for item in batch]
        images = [item['image'] for item in batch]

        prompts = [t + l for t, l in zip(texts, labels)]
        tokenized_prompts = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding="longest",
        )

        input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        labels = input_ids.clone()
        for idx, text in enumerate(texts):
            prompt_ids = self.tokenizer(text).input_ids
            labels[idx, :len(prompt_ids)] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        pixel_values = self.vision_processor(images=images, return_tensors="pt").pixel_values

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }