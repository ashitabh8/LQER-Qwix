"""LQER (Low-rank Quantization Error Reconstruction) for Qwix.

This package provides an extension to Qwix that adds low-rank error correction
to post-training quantization (PTQ).
"""

from .lqer_core import (
    LqerRule,
    LqerWeight,
    LqerProvider,
    lqer_quantize_params,
)

__all__ = [
    'LqerRule',
    'LqerWeight',
    'LqerProvider',
    'lqer_quantize_params',
]

