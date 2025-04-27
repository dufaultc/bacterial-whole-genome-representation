# Adapted from https://huggingface.co/yairschiff/bimamba-template
# Some changes to allow for multi task pretraining
"""Caduceus config for Hugging Face.
"""

from typing import Optional, Union

from transformers import PretrainedConfig


class BiMambaConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality."""

    model_type = "bimamba"

    def __init__(
        self,
        # From original MambaConfig
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
        norm_epsilon: float = 1e-5,
        # Used in init_weights
        initializer_cfg: Optional[dict] = None,
        # Caduceus-specific params
        bidirectional: bool = True,
        bidirectional_strategy: Union[str, None] = "add",
        bidirectional_weight_tie: bool = True,
        mean_pool: bool = True,
        mlm_loss_share: float = 1.0,
        presence_loss_share: float = 1.0,
        random_truncation_level: bool = True,
        dropout_level: float = 0.4,
        simple_head: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.mean_pool = mean_pool
        self.mlm_loss_share = mlm_loss_share
        self.presence_loss_share = presence_loss_share
        self.random_truncation_level = random_truncation_level
        self.dropout_level = dropout_level
        self.simple_head = simple_head
