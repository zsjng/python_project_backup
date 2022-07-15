from enum import Enum
from typing import Optional

import ptwt
import torch
from einops import rearrange
from loguru import logger


class DwtCoeffs(Enum):
    LL = 0
    LH = 1
    HL = 2
    HH = 3


def image_dwt_loo(
    image: torch.Tensor, loo: Optional[DwtCoeffs] = DwtCoeffs.LH
) -> torch.Tensor:
    """
    Performs DWT on X, set coefficients other than approx (LL) of X as zeros
    (leave-one-out), and tries to reconstruct X with inverse DWT.

    This is a 'backward' technique for measuring the weight of a specific coefficient.

    X -(dwt)--> ll, lh, hl, hh
      -(zero)-> set one of lh, hl, or hh to [[0, ..., 0]]
      -(idwt)-> X'

    """
    wavelet = "haar"

    # change image shape from [b, c, w, h] to [c, b, w, h] to prepare for dwt
    X = rearrange(image, "b c w h -> c b w h")
    ll, (lh, hl, hh) = ptwt.wavedec2(X, wavelet, level=1, mode="constant")

    # a zeroed-out coeff
    zeros = torch.zeros_like(lh)  # ll, lh, hl, and hh are of the same shape

    # idwt with one of the coeffs as zeros
    if loo == DwtCoeffs.LH:
        X_prime = ptwt.waverec2([ll, (zeros, hl, hh)], wavelet)
    elif loo == DwtCoeffs.HL:
        X_prime = ptwt.waverec2([ll, (lh, zeros, hh)], wavelet)
    elif loo == DwtCoeffs.HH:
        X_prime = ptwt.waverec2([ll, (lh, hl, zeros)], wavelet)
    elif loo == DwtCoeffs.LL:
        X_prime = ptwt.waverec2([zeros, (lh, hl, hh)], wavelet)
        logger.warning("Coefficient LL set to zeros, IDWT may not succeed.")
    else:
        X_prime = ptwt.waverec2([ll, (lh, hl, hh)], wavelet)
        logger.debug("Coefficient left as default without leaving-one-out.")

    # revert image shape from [c, b, w, h] to [b, c, w, h]
    return rearrange(X_prime, "c b w h -> b c w h")


def image_dwt_leo(
    image: torch.Tensor, kept: Optional[DwtCoeffs] = DwtCoeffs.LL
) -> torch.Tensor:
    """
    Performs DWT on X, set all other coefficients other than selected one (kept) as
    zeros (leave-everything-out), and tries to reconstruct X with inverse DWT.

    This is a 'forward' technique for measuring the weight of a specific coefficient.

    X -(dwt)--> ll, lh, hl, hh
      -(zero)-> set lh, hl and hh to [[0, ..., 0]]
      -(idwt)-> X'

    """
    wavelet = "haar"

    # change image shape from [b, c, w, h] to [c, b, w, h] to prepare for dwt
    X = rearrange(image, "b c w h -> c b w h")
    ll, (lh, hl, hh) = ptwt.wavedec2(X, wavelet, level=1, mode="constant")

    # a zeroed-out coeff
    zeros = torch.zeros_like(lh)  # ll, lh, hl, and hh are of the same shape

    # idwt with coeffs other than kept as zeros
    if kept == DwtCoeffs.LL:
        X_prime = ptwt.waverec2([ll, (zeros, zeros, zeros)], wavelet)
    elif kept == DwtCoeffs.LH:
        X_prime = ptwt.waverec2([zeros, (lh, zeros, zeros)], wavelet)
    elif kept == DwtCoeffs.HL:
        X_prime = ptwt.waverec2([zeros, (zeros, hl, zeros)], wavelet)
    elif kept == DwtCoeffs.HH:
        X_prime = ptwt.waverec2([zeros, (zeros, zeros, hh)], wavelet)
    else:
        X_prime = ptwt.waverec2([ll, (lh, hl, hh)], wavelet)

    # revert image shape from [c, b, w, h] to [b, c, w, h]
    return rearrange(X_prime, "c b w h -> b c w h")
