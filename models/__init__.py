"""Radar models.

.. [N1] RadarHD: High resolution point clouds from mmWave Radar
    https://akarsh-prabhakara.github.io/research/radarhd/
.. [N2] ConvNeXt: A ConvNet for the 2020s
    https://arxiv.org/abs/2201.03545
.. [N3] Vision Transformers for Dense Prediction
    https://arxiv.org/abs/2103.13413
"""

from .axial import AxialEncoder
from .grt import (
    AzimuthFFTTransformerEncoder,
    QueryMixedTransformerDecoder,
    ResidualRefinedTransformerDecoder,
    TransformerDecoder,
    TransformerEncoder,
    VectorDecoder,
)
from .grtf import TransformerFourierEncoder
from .swin import SwinDPT2DDecoder, SwinTransformerEncoder
from .t_fftradnet import TFFTRadNet2DDecoder, TFFTRadNetEncoder
from .unet import (
    AzimuthFFTUNet3DEncoder,
    AzimuthFFTUNetEncoder,
    UNet3DDecoder,
    UNet3DEncoder,
    UNetBEVDecoder,
    UNetEncoder,
)
from .unext import UNeXTBEVDecoder, UNeXTEncoder

__all__ = [
    "AxialEncoder",
    "TransformerEncoder", "AzimuthFFTTransformerEncoder",
    "TransformerDecoder", "QueryMixedTransformerDecoder",
    "ResidualRefinedTransformerDecoder", "VectorDecoder",
    "TransformerFourierEncoder",
    "SwinDPT2DDecoder", "SwinTransformerEncoder",
    "UNetEncoder", "AzimuthFFTUNetEncoder", "UNetBEVDecoder",
    "UNet3DEncoder", "AzimuthFFTUNet3DEncoder", "UNet3DDecoder",
    "UNeXTEncoder", "UNeXTBEVDecoder",
    "TFFTRadNetEncoder", "TFFTRadNet2DDecoder"
]
