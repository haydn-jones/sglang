import re

from PIL import Image
from sglang.srt.configs.dots_ocr import DotsOCRConfig, DotsOCRProcessor
from sglang.srt.managers.io_struct import (
    ImageDataInputItem,
)
from sglang.srt.models.dots_ocr import DotsOCRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import smart_resize
from sglang.srt.server_args import ServerArgs
from torch import nn


class DotsOCRImageProcessor(BaseMultimodalProcessor):
    models: list[type[nn.Module]] = [DotsOCRForCausalLM]

    _processor: DotsOCRProcessor

    def __init__(
        self,
        hf_config: DotsOCRConfig,
        server_args: ServerArgs,
        _processor: DotsOCRProcessor,
        transport_mode,
        *args,
        **kwargs,
    ):
        super().__init__(hf_config, server_args, _processor, transport_mode, *args, **kwargs)
        self.IMAGE_TOKEN = "<|img|><|imgpad|><|endofimg|>"
        self.IMAGE_TOKEN_REGEX = re.compile(r"<\|img\|>(?:<\|imgpad\|>)+<\|endofimg\|>")

        self.im_start_id = _processor.tokenizer.encode("<|img|>")[0]
        self.im_end_id = _processor.tokenizer.encode("<|endofimg|>")[0]
        self.im_token_id = _processor.tokenizer.encode("<|imgpad|>")[0]

        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size

        self.resize_kwargs = {
            "size_factor": patch_size * merge_size,
            "min_pixels": _processor.image_processor.min_pixels,  # type: ignore
            "max_pixels": _processor.image_processor.max_pixels,  # type: ignore
        }

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=hf_config.image_token_id,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: ImageDataInputItem | list[ImageDataInputItem] | None,
        input_text: str | list[int],
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,  # type: ignore
            image_data=image_data,  # type: ignore
            multimodal_tokens=self.mm_tokens,
        )

        if base_output.images and isinstance(base_output.images[0], Image.Image):
            base_output.images = [resize_image_async(image, **self.resize_kwargs) for image in base_output.images]  # type: ignore

        mm_items, input_ids, _ = self.process_and_combine_mm_data(base_output, self.mm_tokens)

        input_ids = input_ids.flatten()

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.im_token_id,
        }


def resize_image_async(
    image: Image.Image,
    size_factor: int,
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (resized_height, resized_width) != (height, width):
        image = image.resize((resized_width, resized_height))

    return image
