from __future__ import annotations

import pytest

from cristma.core import MeasuredValue
from cristma.core.precision import reported_numeric_error


def test_reported_numeric_error_uses_three_sigma_when_available() -> None:
    value = MeasuredValue(9.10888, 0.00003, "9.10888(3)")

    assert reported_numeric_error(value) == pytest.approx(0.00009)


def test_reported_numeric_error_uses_half_last_decimal_without_uncertainty() -> None:
    value = MeasuredValue(9.108879, None, "9.108879")

    assert reported_numeric_error(value) == pytest.approx(0.0000005)


def test_reported_numeric_error_preserves_exponent_scale() -> None:
    value = MeasuredValue(1.25e-3, None, "1.25e-3")

    assert reported_numeric_error(value) == pytest.approx(0.005e-3)


@pytest.mark.parametrize("sigma_multiplier", (True, 0.0, -1.0, float("inf")))
def test_reported_numeric_error_rejects_invalid_sigma_multiplier(
    sigma_multiplier: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        reported_numeric_error(
            MeasuredValue(1.0, 0.1, "1.0(1)"),
            sigma_multiplier=sigma_multiplier,  # type: ignore[arg-type]
        )
