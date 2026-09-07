"""Numerical bounds derived from the precision of reported values."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from numbers import Real

from .values import MeasuredValue


def reported_numeric_error(
    value: MeasuredValue,
    *,
    sigma_multiplier: float = 3.0,
) -> float:
    """Return an absolute error bound implied by uncertainty or written digits."""

    if not isinstance(value, MeasuredValue):
        raise TypeError("value must be MeasuredValue")
    if isinstance(sigma_multiplier, bool) or not isinstance(sigma_multiplier, Real):
        raise TypeError("sigma_multiplier must be a real number")
    multiplier = float(sigma_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("sigma_multiplier must be positive and finite")
    if value.uncertainty is not None:
        uncertainty = float(value.uncertainty)
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError("reported uncertainty must be finite and non-negative")
        return multiplier * uncertainty
    if value.raw is None:
        return 0.0
    numeric = value.raw.partition("(")[0]
    try:
        exponent = Decimal(numeric).as_tuple().exponent
    except InvalidOperation:
        return 0.0
    if not isinstance(exponent, int) or exponent >= 0:
        return 0.0
    return 0.5 * float(Decimal(10) ** exponent)


__all__ = ["reported_numeric_error"]
