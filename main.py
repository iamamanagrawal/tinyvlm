"""
    Main script to train a Vision-Language Model (VLM) using a specified dataset.
"""
import torch
import logging
from dataclasses import dataclass, field

from src.model import VisionLanguageModel
from src.ingest import DataLoaderLite, load_dataset
from src.train import Trainer
from src.utils import (
    LANGUAGE_MODEL_PATH, SPECIAL_TOKENS, VISION_ENCODER_PATH, LOCAL_DATASET_PATH,
    get_language_model_and_tokenizer, get_vision_processor_and_model
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@dataclass
class VLMConfig:

    ## model configuration
    vision_encoder_path: str = VISION_ENCODER_PATH
    language_model_path: str = LANGUAGE_MODEL_PATH
    special_tokens: dict = field(default_factory=lambda: SPECIAL_TOKENS)

    ## dataset configuration
    dataset_path: str = LOCAL_DATASET_PATH

    ## training configurations
    test_size: float = 0.1
    batch_size: int = 96
    learning_rate: float = 1e-3
    gradient_accumulation_steps: int = 4
    max_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    num_epochs: int = 1

def main():

    config = VLMConfig()

    ## initialize models and tokenizer
    tokenizer, language_model = get_language_model_and_tokenizer(config.language_model_path, config.special_tokens)
    vision_processor, vision_model = get_vision_processor_and_model(config.vision_encoder_path)

    model = VisionLanguageModel(
        language_model=language_model,
        vision_encoder=vision_model,
        image_token_id=tokenizer.convert_tokens_to_ids(tokenizer.image_token),
        freeze_vision=True,
        freeze_language=True,
    )

    ## load dataset, split and create dataloaders
    dataset = load_dataset(config.dataset_path)

    test_size = int(config.test_size * len(dataset))
    train_dataset = dataset[:-test_size]
    val_dataset = dataset[-test_size:]

    train_dataloader = DataLoaderLite(train_dataset, batch_size=config.batch_size, tokenizer=tokenizer, vision_processor=vision_processor)
    val_dataloader = DataLoaderLite(val_dataset, batch_size=config.batch_size, tokenizer=tokenizer, vision_processor=vision_processor)

    logging.info("Using device: {}".format(config.device))
    logging.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    ## initialize trainer and start training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_norm=config.max_norm,
        device=config.device,
        num_epochs=config.num_epochs,
    )
    trainer.train()

    return None

if __name__ == "__main__":
    main()