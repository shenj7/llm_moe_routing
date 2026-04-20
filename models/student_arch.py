"""
Student model architecture for LLaDA progressive distillation.

Creates a smaller bidirectional transformer using the *same* model class as
LLaDA (loaded via trust_remote_code) but with a reduced layer count /
hidden dimension.  The student is always initialised with random weights —
no knowledge of the teacher parameters is assumed up-front.

Three convenience presets are provided:
    "400m"  –  ~400 M parameters  (safe on A30 alongside 8-bit teacher)
    "1b"    –  ~1 B  parameters   (default, good distillation target)
    "3b"    –  ~3 B  parameters   (requires memory optimisation; see config)

You can also point to a YAML config file in config/student_configs/ for
full control over every architecture hyperparameter.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import yaml
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

# HuggingFace model ID used as the architecture template.
# We download the config from here, shrink it, then instantiate fresh weights.
_TEMPLATE_MODEL_ID = "GSAI-ML/LLaDA-8B-Base"

# Keys in the HuggingFace config that control model size.
_ARCH_KEYS = {
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
}

# ---------------------------------------------------------------------------
# Built-in size presets
# ---------------------------------------------------------------------------
# LLaDA-8B baseline for reference:
#   num_hidden_layers : 32    hidden_size : 4096
#   num_attention_heads: 32   num_kv_heads: 8    intermediate: 14336

STUDENT_PRESETS: Dict[str, Dict[str, int]] = {
    "400m": dict(
        num_hidden_layers=12,
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=2816,
    ),
    "1b": dict(
        num_hidden_layers=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=5632,
    ),
    "3b": dict(
        num_hidden_layers=28,
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,
        intermediate_size=8192,
    ),
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_student_model(
    student_config_path: Optional[str] = None,
    preset: Optional[str] = None,
    template_model_id: str = _TEMPLATE_MODEL_ID,
    overrides: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Instantiate a randomly-initialised student model.

    Exactly one of *student_config_path* or *preset* must be provided.

    Args:
        student_config_path: Path to a YAML file in config/student_configs/.
                             Takes precedence over *preset*.
        preset:              One of "400m" | "1b" | "3b".
        template_model_id:   HuggingFace repo to derive the base config from.
        overrides:           Extra key-value pairs applied on top (useful for
                             quick CLI experiments).

    Returns:
        student: Randomly-initialised nn.Module with the LLaDA model class
                 and the requested smaller dimensions.
    """
    if student_config_path is None and preset is None:
        raise ValueError("Provide either student_config_path or preset.")

    # ---- Resolve size parameters -------------------------------------------
    if student_config_path is not None:
        with open(student_config_path) as fh:
            yaml_cfg: Dict = yaml.safe_load(fh)
        arch = yaml_cfg.get("architecture", {})
        # Filter to known architecture keys only.
        size_params: Dict[str, Any] = {k: v for k, v in arch.items() if k in _ARCH_KEYS}
        # Allow the YAML to override the template model ID too.
        template_model_id = arch.get("base_model_id", template_model_id)
    else:
        if preset not in STUDENT_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Valid options: {list(STUDENT_PRESETS)}"
            )
        size_params = dict(STUDENT_PRESETS[preset])

    if overrides:
        size_params.update(overrides)

    # ---- Download & patch the config ---------------------------------------
    logger.info("Fetching base config from %s …", template_model_id)
    config = AutoConfig.from_pretrained(template_model_id, trust_remote_code=True)

    # Build a dict of only the settable (non-read-only-property) attributes.
    # The LLaDA config exposes some attributes (e.g. num_attention_heads) as
    # computed read-only properties; passing them into from_dict/from_pretrained
    # causes an AttributeError because PretrainedConfig.__init__ tries setattr.
    # We drop those keys — the properties will recompute automatically from the
    # backing attributes (e.g. hidden_size) that we do set.
    config_cls = type(config)
    settable: dict = {}
    for key, value in config.to_dict().items():
        is_readonly_prop = any(
            key in klass.__dict__
            and isinstance(klass.__dict__[key], property)
            and klass.__dict__[key].fset is None
            for klass in config_cls.__mro__
        )
        if not is_readonly_prop:
            settable[key] = value

    for key, value in size_params.items():
        if key in settable:
            settable[key] = value
        else:
            logger.warning(
                "'%s' is read-only or absent in the LLaDA config — skipping "
                "(it will be recomputed from underlying attributes).", key
            )

    config = type(config).from_dict(settable)

    # ---- Build model with random weights -----------------------------------
    logger.info("Instantiating student model …")
    student = AutoModel.from_config(config, trust_remote_code=True)

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("Student ready: %.2f B parameters", n_params / 1e9)

    return student
