"""
This script defines
    - Configuration for the Vision-Language Model (VLMConfig)
    - Vision-Language Model architecture (VisionLanguageModel) with a forward method
    - A method to load pretrained weights for the projector module
    - A generate method for text generation given image and prompts.
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
        """Initialize the weights of the VisionProjector."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        language_model,
        vision_encoder,
        image_token_id: int,
        freeze_vision: bool = True,
        freeze_language: bool = True,
        downscale_factor: int = 2,
    ) -> None:
        super(VisionLanguageModel, self).__init__()

        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.image_token_id = image_token_id
        self.downscale_factor = downscale_factor

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
        
        self.projector = VisionProjector((self.downscale_factor ** 2) * vision_dim, hidden_dim)

    def _create_inputs_embeds(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create input embeddings by replacing image token embeddings with projected vision embeddings."""

        # Process images through vision encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values).last_hidden_state  ## (batch_size, 196, 768)
        
        ## apply pixel unshuffle to reduce spatial dimensions
        batch_size, seq_len, vision_dim = vision_outputs.size()
        vision_embeds = vision_outputs.view(batch_size, int(seq_len**0.5), int(seq_len**0.5), vision_dim).permute(0, 3, 1, 2)  ## (batch_size, 768, 14, 14)
        vision_embeds = F.pixel_unshuffle(vision_embeds, downscale_factor=self.downscale_factor)  ## (batch_size, 768*4, 7, 7)
        vision_embeds = vision_embeds.permute(0, 2, 3, 1).contiguous().view(batch_size, seq_len // (self.downscale_factor ** 2), vision_dim * (self.downscale_factor ** 2))  ## (batch_size, 49, 3072)

        # Project vision embeddings to language model hidden size
        projected_embeds = self.projector(vision_embeds)  ## (batch_size, 49, 576)

        # Project text embeddings from the embedding table
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Replace image token embeddings with projected vision embeddings
        inputs_embeds = []
        new_labels = []
        new_attention_mask = []
        for batch_idx in range(input_ids.size(0)):
            embds = text_embeds[batch_idx]
            img_token_pos = (input_ids[batch_idx] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_token_pos) > 0:
                img_token_pos = img_token_pos[0].item()
                prefix_embeds = embds[:img_token_pos, :]  # Embeddings before image token
                suffix_embeds = embds[img_token_pos + 1:, :]  # Embeddings after image token
                new_embeds = torch.cat([prefix_embeds, projected_embeds[batch_idx].to(dtype=embds.dtype), suffix_embeds], dim=0)
                inputs_embeds.append(new_embeds)

                new_mask = torch.cat([attention_mask[batch_idx, :img_token_pos], torch.ones(projected_embeds.size(1), device=attention_mask.device, dtype=attention_mask.dtype), attention_mask[batch_idx, img_token_pos + 1:]], dim=0)
                new_attention_mask.append(new_mask)

                if labels is not None:
                    lbls = labels[batch_idx]
                    prefix_labels = lbls[:img_token_pos]
                    suffix_labels = lbls[img_token_pos + 1:]
                    new_lbls = torch.cat([prefix_labels, torch.tensor([-100]*projected_embeds.size(1), device=lbls.device, dtype=lbls.dtype), suffix_labels], dim=0)
                    new_labels.append(new_lbls)
            else:
                inputs_embeds.append(embds)
                new_attention_mask.append(attention_mask[batch_idx])
                if labels is not None:
                    new_labels.append(labels[batch_idx])
        
        # Stack the list of embeddings into a single tensor
        inputs_embeds = torch.stack(inputs_embeds)
        attention_mask = torch.stack(new_attention_mask)
        if labels is not None:
            labels = torch.stack(new_labels)
            del new_labels  # free up memory

        del vision_outputs, vision_embeds, projected_embeds, text_embeds, new_attention_mask # free up memory
        return inputs_embeds, attention_mask, labels

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the Vision-Language Model."""

        # Create inputs_embeds by replacing image token embeddings with projected vision embeddings
        inputs_embeds, attention_mask, labels = self._create_inputs_embeds(pixel_values, attention_mask, input_ids, labels)

        # Forward pass through language model
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), 
                ignore_index=ignore_index
            )
        
        return logits, loss
    
    def from_pretrained_projector(self, projector_path: str) -> "VisionLanguageModel":
        """Load pretrained weights for the projector module."""
        state_dict = torch.load(projector_path, map_location='cpu')
        self.projector.load_state_dict(state_dict)
        return self
    
    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self.vision_encoder.device

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 2.0,
    ) -> torch.Tensor:
        """Generate text given image and input prompt."""
        inputs_embeds, attention_mask, _ = self._create_inputs_embeds(pixel_values, attention_mask, input_ids)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )