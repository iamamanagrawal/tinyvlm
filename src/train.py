"""
    The purpose of this file is to handle training part of the tinyvlm model.
"""

import os
import torch
import wandb
import logging
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from src.inference import inference
from src.model import VisionLanguageModel
from src.ingest import DataLoaderLite
from src.utils import PROJECTOR_PATH

## init the weight & biases monitoring service
wandb.init(project="TinyVLM", name="TinyVLM-Training")

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class Trainer:
    def __init__(
        self,
        model: VisionLanguageModel,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoaderLite,
        val_dataloader: DataLoaderLite,
        num_epochs: int,
        learning_rate: float,
        gradient_accumulation_steps: int,
        max_norm: float,
        device: str,
    ) -> None:
        
        self.model = model        
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_norm = max_norm

        ## optimizer and scheduler
        self.optimizer = AdamW(self.model.projector.parameters(), lr=learning_rate)
        self.total_steps = (self.train_dataloader.num_batches // self.gradient_accumulation_steps + 1) * self.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * 0.05),
            num_training_steps=self.total_steps
        )

        ## move model to device
        self.model.to(self.device)

    @torch.inference_mode()
    def run_example(self, example_directory: str) -> dict:
        """Run inference on example images in the specified directory containing image files."""

        for image_path in os.listdir(example_directory):
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            conversation = {
                "conversations": [
                    {"role": "human", "content": "<image>\nDescribe the image."},
                ],
                "image": os.path.join(example_directory, image_path)
            }

            ## perform inference
            generated_text = inference(
                model=self.model,
                conversations=conversation,
                vision_processor=self.val_dataloader.vision_processor,
                tokenizer=self.tokenizer,
                max_new_tokens=20,
            )

            print(f"=== Inference Example {image_path} ===")
            print(f"Image Path: {image_path}")
            print(f"Generated Text: {generated_text}")
            print("=========================")
        pass

    def train(self) -> None:

        for idx in range(self.total_steps):

            ## at every iteration, update the parameters
            self.model.train()
            gradient_accum_loss = 0.0
            self.optimizer.zero_grad()
            for _ in range(self.gradient_accumulation_steps):
                batch = self.train_dataloader.next_batch()
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                ## added a try and except condition to mitigate OOM issue
                try:
                    with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                        _, loss = self.model(**inputs)
                        loss /= self.gradient_accumulation_steps
                    loss.backward()
                    gradient_accum_loss += loss.item()
                except RuntimeError as e:
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    logging.error(f"RuntimeError at index {idx} and seq len {inputs['input_ids'].size(1)}: {e}")
                    continue

            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            

            ## get batch from val dataloader and evaluate
            self.model.eval()
            batch = self.val_dataloader.next_batch()
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                try:
                    with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                        _, val_loss = self.model(**inputs)
                except RuntimeError as e:
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    logging.error(f"RuntimeError during validation at batch with seq len of {inputs['input_ids'].size(1)}: {e}")
                    val_loss = torch.tensor(float('inf'))

            ## every 20 steps, run an example inference
            if (idx + 1) % 20 == 0:
                logging.info("Running inference on example images...")
                self.run_example("examples/")
                logging.info("Completed inference on example images.")


            logging.info(
                f"Step {idx + 1}/{self.total_steps}, "
                f"Train Loss: {gradient_accum_loss:.4f}, "
                f"Val Loss: {val_loss.item():.4f}, "
                f"Norm: {norm:.4f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )
            wandb.log({
                "train/loss": gradient_accum_loss,
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/grad_norm": norm,
                "val/loss": val_loss.item()
            })

        torch.save(self.model.projector.state_dict(), PROJECTOR_PATH)
        logging.info("Epoch training complete.")

        return None