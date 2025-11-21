"""
    The purpose of this file is to handle training part of the tinyvlm model.
"""

import torch
import wandb
import logging
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer

from src.model import VisionLanguageModel
from src.ingest import DataLoaderLite, chat_template
from src.utils import EXAMPLE_PROMPT, IMAGE_PROMPT_TEMPLATE, PROJECTOR_PATH

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
        learning_rate: float,
        gradient_accumulation_steps: int,
        max_norm: float,
        device: str,
    ) -> None:
        
        self.model = model        
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_norm = max_norm

        ## optimizer and scheduler
        self.optimizer = AdamW(self.model.projector.parameters(), lr=learning_rate)
        self.total_steps = self.train_dataloader.num_batches // self.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * 0.03),
            num_training_steps=self.total_steps
        )

        ## move model to device
        self.model.to(self.device)

    def train_epoch(self) -> None:

        self.model.train()
        for idx in range(self.total_steps):

            ## at every iteration, update the parameters
            gradient_accum_loss = 0.0
            self.optimizer.zero_grad()
            for _ in range(self.gradient_accumulation_steps):
                batch = self.train_dataloader.next_batch()
                collated = self.train_dataloader.collate_fn(batch)
                inputs = {k: v.to(self.device) for k, v in collated.items()}

                ## added a try and except condition to mitigate OOM issue
                try:
                    _, loss = self.model(**inputs)
                    loss /= self.gradient_accumulation_steps
                    loss.backward()
                    gradient_accum_loss += loss.item()
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    logging.error(f"RuntimeError at index {idx} and seq len {inputs['input_ids'].size(1)}: {e}")
                    continue

            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()
            self.scheduler.step()
            

            ## get batch from val dataloader and evaluate
            self.model.eval()
            batch = self.val_dataloader.next_batch()
            collated = self.val_dataloader.collate_fn(batch)
            inputs = {k: v.to(self.device) for k, v in collated.items()}
            with torch.no_grad():
                try:
                    _, val_loss = self.model(**inputs)
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    logging.error(f"RuntimeError during validation at batch with seq len of {inputs['input_ids'].size(1)}: {e}")
                    val_loss = torch.tensor(float('inf'))

            ## use model to generate text for first sample in validation batch
            try:
                if idx % 20 == 0:
                    prompt = chat_template(EXAMPLE_PROMPT, path="dummy", image_prompt_template=IMAGE_PROMPT_TEMPLATE)['text'] + "<|im_start|>assistant\n"
                    pixel_values = inputs['pixel_values'][0]
                    label = inputs['input_ids'][0]
                    tokenized = self.tokenizer(prompt, return_tensors='pt')
                    input_ids = tokenized.input_ids.to(self.device)
                    attention_mask = tokenized.attention_mask.to(self.device)
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            pixel_values=pixel_values.unsqueeze(0),
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=50
                        )
                    generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    label_text = self.tokenizer.batch_decode(label.unsqueeze(0), skip_special_tokens=True)[0]

                    logging.info("=== Generation Example ===")
                    logging.info(f"Generated Text: {generated_text}")
                    logging.info(f"Label Text: {label_text}")
                    logging.info("=========================")

            except RuntimeError as e:
                torch.cuda.empty_cache()
                logging.error(f"RuntimeError during generation: {e}")

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