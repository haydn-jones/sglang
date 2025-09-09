from collections.abc import Iterable

import torch
from sglang.srt.configs.dots_ocr import DotsOCRConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

from .dots_ocr_vit import DotsVisionTransformer


class DotsOCRForCausalLM(nn.Module):
    """DotsVLM model for sglang inference"""

    def __init__(self, config: DotsOCRConfig, quant_config: QuantizationConfig | None = None) -> None:
        super().__init__()

        self.image_token_id = config.im_span_id
        self.video_token_id = config.video_span_id

        self.language_model = Qwen2ForCausalLM(config.language_config, quant_config)

        # Initialize vision tower (matching transformers naming for weight compatibility)
        self.vision_tower = DotsVisionTransformer(config.vision_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights"""
        weights = list(weights)

        # Separate vision tower weights and language model weights
        vision_weights = []
        language_weights = []

        for name, loaded_weight in weights:
            if name.startswith("vision_tower."):
                # Remove "vision_tower." prefix for vision tower weights
                vision_name = name[len("vision_tower.") :]
                vision_weights.append((vision_name, loaded_weight))
            else:
                # All other weights go to language model
                language_weights.append((name, loaded_weight))

        # Load vision tower weights
        vision_state_dict = dict(vision_weights)
        self.vision_tower.load_state_dict(vision_state_dict, strict=True)

        # Load language model weights
        if language_weights:
            self.language_model.load_weights(language_weights)

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Extract image features from multimodal data items"""
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat(
            [torch.as_tensor(item.feature, dtype=self.vision_tower.dtype) for item in items],
            dim=0,
        )
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)
        image_embeds = self.vision_tower(pixel_values, image_grid_thw)

        return image_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.language_model,
        )
        return hidden_states


EntryClass = [DotsOCRForCausalLM]
