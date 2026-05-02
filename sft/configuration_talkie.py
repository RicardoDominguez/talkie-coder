"""Talkie model configuration for HuggingFace Transformers."""

from transformers import PretrainedConfig


class TalkieConfig(PretrainedConfig):
    """Configuration class for the Talkie 13B decoder-only transformer.

    This is a 40-layer, 40-head GPT with RoPE, SwiGLU, RMS normalisation,
    embedding skip connections, and per-head / per-layer gain parameters.
    """

    model_type = "talkie"

    def __init__(
        self,
        vocab_size: int = 65540,
        hidden_size: int = 5120,
        intermediate_size: int = 13696,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 40,
        head_dim: int = 128,
        max_position_embeddings: int = 2048,
        rope_theta: float = 1_000_000.0,
        torch_dtype: str = "bfloat16",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )
