from __future__ import annotations

from fractions import Fraction

import pytest

from cristma.core import MeasuredValue, UnitCell
from cristma.crystallography import (
    DirectBasisConvention,
    SpaceGroupCatalog,
    SymmetryContext,
    SymmetryContextInvariantError,
    SymmetrySourceKind,
)
from cristma.symmetry import AffineOperation


def _value(value: float) -> MeasuredValue:
    return MeasuredValue(value, None, str(value))


def _orthogonal_cell(a: float = 5.0, b: float = 5.0, c: float = 7.0) -> UnitCell:
    return UnitCell(
        _value(a),
        _value(b),
        _value(c),
        _value(90.0),
        _value(90.0),
        _value(90.0),
    )


def _operation(
    rotation: tuple[tuple[int, int, int], ...],
    translation: tuple[Fraction | int, Fraction | int, Fraction | int] = (0, 0, 0),
    *,
    operation_id: str | None = None,
) -> AffineOperation:
    return AffineOperation(
        rotation=tuple(
            tuple(Fraction(value) for value in row)
            for row in rotation
        ),
        translation=tuple(Fraction(value) for value in translation),
        id=operation_id,
    )


IDENTITY = _operation(((1, 0, 0), (0, 1, 0), (0, 0, 1)), operation_id="reported-7")
INVERSION = _operation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)), operation_id="reported-2")


def test_explicit_operation_order_does_not_change_context_identity() -> None:
    cell = _orthogonal_cell()

    first = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)
    second = SymmetryContext.from_operations((INVERSION, IDENTITY), cell)

    assert first.operation_keys == second.operation_keys
    assert first.symmetry_action_fingerprint == second.symmetry_action_fingerprint
    assert first.operations == second.operations


def test_context_identity_ignores_reported_operation_ids() -> None:
    cell = _orthogonal_cell()
    renamed_identity = _operation(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        operation_id="anything",
    )
    renamed_inversion = _operation(
        ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
        operation_id="else",
    )

    first = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)
    second = SymmetryContext.from_operations((renamed_inversion, renamed_identity), cell)

    assert first.symmetry_action_fingerprint == second.symmetry_action_fingerprint


def test_valid_explicit_operations_are_complete_without_setting_id() -> None:
    context = SymmetryContext.from_operations((IDENTITY, INVERSION), _orthogonal_cell())

    assert context.source_kind is SymmetrySourceKind.VALID_EXPLICIT_OPERATIONS
    assert context.setting_id is None
    assert context.basis_convention is DirectBasisConvention.FRACTIONAL_DIRECT


def test_non_closed_operation_set_is_rejected() -> None:
    quarter_turn = _operation(((0, -1, 0), (1, 0, 0), (0, 0, 1)))

    with pytest.raises(SymmetryContextInvariantError) as caught:
        SymmetryContext.from_operations((IDENTITY, quarter_turn), _orthogonal_cell())

    assert caught.value.code == "symmetry.context.group_invalid"


def test_rotation_incompatible_with_cell_metric_is_rejected() -> None:
    swap_xy = _operation(((0, 1, 0), (1, 0, 0), (0, 0, 1)))

    with pytest.raises(SymmetryContextInvariantError) as caught:
        SymmetryContext.from_operations((IDENTITY, swap_xy), _orthogonal_cell(4.0, 5.0))

    assert caught.value.code == "symmetry.context.metric_incompatible"


def _rounded_fd3m_cell(*, b: float = 9.108879) -> UnitCell:
    return UnitCell(
        MeasuredValue(9.10888, 0.00003, "9.10888(3)"),
        MeasuredValue(b, None, str(b)),
        MeasuredValue(9.108879, None, "9.108879"),
        MeasuredValue(90.0, None, "90.0"),
        MeasuredValue(90.0, None, "90.0"),
        MeasuredValue(90.0, None, "90.0"),
    )


def test_metric_rounding_within_reported_cell_precision_is_recoverable() -> None:
    setting = SpaceGroupCatalog.default().by_setting(526)

    context = SymmetryContext.from_setting(setting, _rounded_fd3m_cell())

    assert context.metric_tolerance > 1e-8
    assert any(
        item.code == "symmetry.context.metric_within_reported_precision"
        for item in context.diagnostics
    )


def test_metric_mismatch_outside_reported_cell_precision_is_rejected() -> None:
    setting = SpaceGroupCatalog.default().by_setting(526)

    with pytest.raises(SymmetryContextInvariantError) as caught:
        SymmetryContext.from_setting(setting, _rounded_fd3m_cell(b=9.107879))

    assert caught.value.code == "symmetry.context.metric_incompatible"


def test_duplicate_normalized_operations_are_rejected() -> None:
    translated_identity = _operation(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        (1, 0, 0),
    )

    with pytest.raises(SymmetryContextInvariantError) as caught:
        SymmetryContext.from_operations((IDENTITY, translated_identity), _orthogonal_cell())

    assert caught.value.code == "symmetry.context.duplicate_operation"
