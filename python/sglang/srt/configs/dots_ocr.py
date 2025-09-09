from transformers import (
    AutoProcessor,
    PretrainedConfig,
    ProcessorMixin,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)
from transformers.processing_utils import ProcessingKwargs, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class DotsOCRVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_size: int = 1536,
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation: str = "flash_attention_2",
        initializer_range: float = 0.02,
        init_merger_std: float = 0.02,
        is_causal: bool = False,
        post_norm: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing


class DotsOCRConfig(PretrainedConfig):
    model_type = "dots_ocr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.im_span_id = kwargs.get("image_token_id", 128815)
        self.video_span_id = kwargs.get("video_token_id", 128836)
        self.vision_config = DotsOCRVisionConfig(**vision_config)
        self.language_config = Qwen2Config(**kwargs)
        self.architectures = ["DotsOCRForCausalLM"]


class DotsOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
    }  # type: ignore


class DotsOCRProcessor(ProcessorMixin):
    r"""
    Constructs a DotsOCR processor which wraps a Qwen2OCR image processor and a Llama tokenizer into a single processor.
    [`DotsOCRProcessor`] offers all the functionalities of [`Qwen2OCRImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~DotsOCRProcessor.__call__`] and [`~DotsOCRProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]

    image_processor_class = "Qwen2VLImageProcessorFast"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    tokenizer: Qwen2TokenizerFast
    image_processor: Qwen2VLImageProcessorFast

    def __init__(
        self,
        image_processor: Qwen2VLImageProcessorFast,
        tokenizer: Qwen2TokenizerFast,
        chat_template: str | None = None,
        **kwargs,
    ):
        self.image_token = "<|imgpad|>"
        self.video_token = "<|video_pad|>"
        self.img_token = "<|img|>"
        self.endofimg_token = "<|endofimg|>"
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | None = None,
        **kwargs: Unpack[DotsOCRProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            DotsOCRProcessorKwargs,  # type: ignore
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if not isinstance(text, list) and text is not None:
            text = [text]

        if image_grid_thw is not None and text is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})


AutoProcessor.register(DotsOCRConfig, DotsOCRProcessor)
