"""
This script defines
    - Configuration for the Vision-Language Model (VLMConfig)
    - Vision-Language Model architecture (VisionLanguageModel) with a forward method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import *

class VisionProjector(nn.Module):
    def __init__(self, vision_dim, proj_dim):
        super().__init__()
        self.vision_dim = vision_dim
        self.proj_dim = proj_dim
        self.gate_proj = nn.Linear(self.vision_dim, 4 * self.vision_dim, bias=False)
        self.up_proj = nn.Linear(self.vision_dim, 4 * self.vision_dim, bias=False)
        self.down_proj = nn.Linear(4 * self.vision_dim, self.proj_dim, bias=False)
        self.act_fn = nn.SiLU()

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        x = self.act_fn(x1) * x2
        x = self.down_proj(x)
        return x


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        language_model,
        vision_encoder,
        image_token_id: int,
        freeze_vision: bool = True,
        freeze_language: bool = True,
    ) -> None:
        super(VisionLanguageModel, self).__init__()

        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.image_token_id = image_token_id

        # Enhanced projection layer with residual connections and dropout
        vision_dim = self.vision_encoder.config.hidden_size
        hidden_dim = self.language_model.config.hidden_size

        ## freeze vision encoder
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        ## freeze language model
        if freeze_language:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        self.projector = VisionProjector(4 * vision_dim, hidden_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        # Process images through vision encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values).last_hidden_state  ## (batch_size, 196, 768)
        vision_embeds = vision_outputs.view(vision_outputs.size(0), vision_outputs.size(1) // 4, -1)  ## (batch_size, 196/4 = 49, 4*768)

        # Project vision embeddings to language model hidden size
        projected_embeds = self.projector(vision_embeds)  ## (batch_size, 49, 576)

        # Prepare inputs for language model
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # # Replace image token embeddings with projected vision embeddings
        mask = (input_ids == self.image_token_id)
        inputs_embeds[mask] = projected_embeds.view(-1, projected_embeds.size(-1)).to(inputs_embeds.dtype)
        

        # Forward pass through language model
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()      # drop last token
        shift_labels = labels[..., 1:].contiguous()  # drop first token

        loss = None
        if labels is not None:
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        
        return logits, loss
    
    def from_pretrained_projector(self, projector_path: str) -> "VisionLanguageModel":
        state_dict = torch.load(projector_path, map_location='cpu')
        self.projector.load_state_dict(state_dict)
        return self

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values).last_hidden_state  ## (batch_size, 196, 768)
        vision_embeds = vision_outputs.view(vision_outputs.size(0), vision_outputs.size(1) // 4, -1)  ## (batch_size, 196/4 = 49, 4*768)

        # Project vision embeddings to language model hidden size
        projected_embeds = self.projector(vision_embeds)  ## (batch_size, 49, 576)

        # Project text embeddings from the embedding table
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Replace image token embeddings with projected vision embeddings
        mask = (input_ids == self.image_token_id)
        inputs_embeds[mask] = projected_embeds.view(-1, projected_embeds.size(-1)).to(inputs_embeds.dtype)

        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return generated_ids
    

