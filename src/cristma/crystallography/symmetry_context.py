"""Validated, deterministic direct-space symmetry contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from cristma.core import UnitCell
from cristma.core.precision import reported_numeric_error
from cristma.diagnostics import Diagnostic, Severity
from cristma.symmetry.affine import AffineOperation, Matrix3, Vector3
from cristma.symmetry.orbit import SpaceGroupDefinition

from .space_group import SpaceGroupSetting


DEFAULT_METRIC_TOLERANCE = 1e-8

_IDENTITY_ROTATION: Matrix3 = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)
_ZERO_TRANSLATION: Vector3 = (Fraction(0), Fraction(0), Fraction(0))


class SymmetrySourceKind(StrEnum):
    """How a validated direct-space symmetry action was supplied."""

    CATALOG_SETTING = "catalog_setting"
    VALID_EXPLICIT_OPERATIONS = "valid_explicit_operations"
    EXPLICIT_IDENTITY_FALLBACK = "explicit_identity_fallback"


class DirectBasisConvention(StrEnum):
    """Coordinate convention used by the direct-space affine action."""

    FRACTIONAL_DIRECT = "fractional_direct"


class SymmetryContextInvariantError(ValueError):
    """A raw symmetry source cannot form a valid working context."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        evidence: tuple[tuple[str, object], ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.evidence = evidence


def _fraction_descriptor(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _operation_descriptor(operation: AffineOperation) -> str:
    normalized = operation.normalized()
    values = (
        *(value for row in normalized.rotation for value in row),
        *normalized.translation,
    )
    return ",".join(_fraction_descriptor(value) for value in values)


def canonical_operation_key(operation: AffineOperation) -> str:
    """Return an exact key independent of source text and input ordering."""

    if not isinstance(operation, AffineOperation):
        raise TypeError("operation must be AffineOperation")
    descriptor = _operation_descriptor(operation).encode("ascii")
    return "operation:" + hashlib.sha256(descriptor).hexdigest()


def _canonical_operation(operation: AffineOperation) -> AffineOperation:
    if not isinstance(operation, AffineOperation):
        raise TypeError("operations must contain only AffineOperation values")
    rotation: list[tuple[Fraction, Fraction, Fraction]] = []
    for row in operation.rotation:
        if len(row) != 3:
            raise SymmetryContextInvariantError(
                "symmetry.context.group_invalid",
                "symmetry rotations must be 3x3 matrices",
            )
        converted = tuple(Fraction(value) for value in row)
        if any(value.denominator != 1 for value in converted):
            raise SymmetryContextInvariantError(
                "symmetry.context.group_invalid",
                "symmetry rotations must contain exact integers",
            )
        rotation.append(converted)
    if len(rotation) != 3 or len(operation.translation) != 3:
        raise SymmetryContextInvariantError(
            "symmetry.context.group_invalid",
            "symmetry operations must act in three dimensions",
        )
    translation = tuple(Fraction(value) % 1 for value in operation.translation)
    return AffineOperation(tuple(rotation), translation)


def _determinant(matrix: Matrix3) -> Fraction:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _matrix_product(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column] for inner in range(3)), Fraction(0))
            for column in range(3)
        )
        for row in range(3)
    )


def _matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum((matrix[row][column] * vector[column] for column in range(3)), Fraction(0))
        for row in range(3)
    )


def _compose(left: AffineOperation, right: AffineOperation) -> AffineOperation:
    return _compose_with_carry(left, right)[0]


def _compose_with_carry(
    left: AffineOperation,
    right: AffineOperation,
) -> tuple[AffineOperation, tuple[int, int, int]]:
    rotated_translation = _matrix_vector(left.rotation, right.translation)
    raw_translation = tuple(
        rotated_translation[index] + left.translation[index]
        for index in range(3)
    )
    normalized_translation = tuple(value % 1 for value in raw_translation)
    operation = AffineOperation(
        _matrix_product(left.rotation, right.rotation),
        normalized_translation,
    )
    carry = tuple(
        int(raw_translation[index] - normalized_translation[index])
        for index in range(3)
    )
    return operation, carry


def _cell_fingerprint(cell: UnitCell) -> str:
    measured = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    values = tuple(float(item.value) for item in measured)
    return hashlib.sha256("|".join(value.hex() for value in values).encode("ascii")).hexdigest()


def _stable_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("symmetry provenance floats must be finite")
        return {"float": value.hex()}
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    raise TypeError(f"unsupported symmetry provenance value: {type(value).__name__}")


def _digest(value: object) -> str:
    serialized = json.dumps(
        _stable_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _interval_product(*intervals: tuple[float, float]) -> tuple[float, float]:
    lower, upper = intervals[0]
    for next_lower, next_upper in intervals[1:]:
        products = (
            lower * next_lower,
            lower * next_upper,
            upper * next_lower,
            upper * next_upper,
        )
        lower, upper = min(products), max(products)
    return lower, upper


def _reported_metric_error(cell: UnitCell) -> np.ndarray:
    edges = (cell.a, cell.b, cell.c)
    angles = (cell.alpha, cell.beta, cell.gamma)
    edge_intervals = tuple(
        (
            max(0.0, float(value.value) - reported_numeric_error(value)),
            float(value.value) + reported_numeric_error(value),
        )
        for value in edges
    )
    cosine_intervals = []
    for value in angles:
        center = float(value.value)
        error = reported_numeric_error(value)
        lower_angle = max(0.0, center - error)
        upper_angle = min(180.0, center + error)
        cosine_intervals.append(
            (
                math.cos(math.radians(upper_angle)),
                math.cos(math.radians(lower_angle)),
            )
        )

    bounds: dict[tuple[int, int], tuple[float, float]] = {}
    for index, interval in enumerate(edge_intervals):
        bounds[(index, index)] = _interval_product(interval, interval)
    for first, second, angle_index in ((1, 2, 0), (0, 2, 1), (0, 1, 2)):
        bounds[(first, second)] = _interval_product(
            edge_intervals[first],
            edge_intervals[second],
            cosine_intervals[angle_index],
        )

    metric = np.asarray(cell.metric, dtype=float)
    error = np.zeros((3, 3), dtype=float)
    for (first, second), (lower, upper) in bounds.items():
        deviation = max(
            abs(metric[first, second] - lower),
            abs(upper - metric[first, second]),
        )
        error[first, second] = deviation
        error[second, first] = deviation
    return error


def _validate_metric(
    operations: tuple[AffineOperation, ...],
    cell: UnitCell,
    tolerance: float,
) -> tuple[float, float, bool]:
    metric = np.asarray(cell.metric, dtype=float)
    metric_error = _reported_metric_error(cell)
    scale = max(1.0, float(np.max(np.abs(metric))))
    maximum_residual = 0.0
    maximum_reported_bound = 0.0
    recovered = False
    for operation in operations:
        rotation = np.asarray(operation.rotation, dtype=float)
        transformed = rotation.T @ metric @ rotation
        residual = np.abs(transformed - metric)
        absolute_rotation = np.abs(rotation)
        reported_bound = (
            absolute_rotation.T @ metric_error @ absolute_rotation
            + metric_error
        )
        allowed = reported_bound + tolerance * scale
        maximum_residual = max(
            maximum_residual,
            float(np.max(residual)) / scale,
        )
        maximum_reported_bound = max(
            maximum_reported_bound,
            float(np.max(reported_bound)) / scale,
        )
        if np.any(residual > allowed):
            raise SymmetryContextInvariantError(
                "symmetry.context.metric_incompatible",
                "symmetry rotation is incompatible with the supplied cell metric",
                evidence=(
                    ("operation_key", canonical_operation_key(operation)),
                    ("maximum_normalized_residual", float(np.max(residual)) / scale),
                    ("configured_tolerance", tolerance),
                    (
                        "maximum_normalized_reported_bound",
                        float(np.max(reported_bound)) / scale,
                    ),
                ),
            )
        if np.any(residual > tolerance * scale):
            recovered = True
    return maximum_residual, maximum_reported_bound, recovered


def _validate_and_canonicalize_operations(
    operations: Iterable[AffineOperation],
    cell: UnitCell,
    metric_tolerance: float,
) -> tuple[
    tuple[AffineOperation, ...],
    dict[tuple[str, str], tuple[str, tuple[int, int, int]]],
    dict[str, str],
    str,
    float,
    float,
    bool,
]:
    if not isinstance(cell, UnitCell):
        raise TypeError("cell must be UnitCell")
    if isinstance(metric_tolerance, bool) or not isinstance(metric_tolerance, (int, float)):
        raise TypeError("metric_tolerance must be a real number")
    metric_tolerance = float(metric_tolerance)
    if not math.isfinite(metric_tolerance) or metric_tolerance <= 0:
        raise ValueError("metric_tolerance must be positive and finite")

    canonical = tuple(_canonical_operation(operation) for operation in operations)
    if not canonical:
        raise SymmetryContextInvariantError(
            "symmetry.context.group_invalid",
            "symmetry context requires at least one operation",
        )
    descriptors = tuple(_operation_descriptor(operation) for operation in canonical)
    if len(set(descriptors)) != len(descriptors):
        raise SymmetryContextInvariantError(
            "symmetry.context.duplicate_operation",
            "symmetry context contains duplicate normalized operations",
        )
    ordered = tuple(
        operation
        for _, operation in sorted(zip(descriptors, canonical, strict=True), key=lambda item: item[0])
    )
    for operation in ordered:
        if _determinant(operation.rotation) not in {Fraction(-1), Fraction(1)}:
            raise SymmetryContextInvariantError(
                "symmetry.context.group_invalid",
                "symmetry rotations must have determinant +1 or -1",
                evidence=(("operation_key", canonical_operation_key(operation)),),
            )

    ordered_descriptors = tuple(_operation_descriptor(operation) for operation in ordered)
    descriptor_set = set(ordered_descriptors)
    identity = AffineOperation(_IDENTITY_ROTATION, _ZERO_TRANSLATION)
    identity_descriptor = _operation_descriptor(identity)
    if identity_descriptor not in descriptor_set:
        raise SymmetryContextInvariantError(
            "symmetry.context.group_invalid",
            "symmetry operation set has no identity",
        )
    products: dict[tuple[str, str], tuple[str, tuple[int, int, int]]] = {}
    for left_descriptor, left in zip(ordered_descriptors, ordered, strict=True):
        for right_descriptor, right in zip(ordered_descriptors, ordered, strict=True):
            product, carry = _compose_with_carry(left, right)
            product_descriptor = _operation_descriptor(product)
            if product_descriptor not in descriptor_set:
                raise SymmetryContextInvariantError(
                    "symmetry.context.group_invalid",
                    "symmetry operation set is not closed",
                    evidence=(
                        ("left_operation_key", canonical_operation_key(left)),
                        ("right_operation_key", canonical_operation_key(right)),
                    ),
                )
            products[(left_descriptor, right_descriptor)] = (
                product_descriptor,
                carry,
            )
    inverses: dict[str, str] = {}
    for operation_descriptor in ordered_descriptors:
        inverse_descriptor = next(
            (
                candidate_descriptor
                for candidate_descriptor in ordered_descriptors
                if products[(operation_descriptor, candidate_descriptor)][0]
                == identity_descriptor
                and products[(candidate_descriptor, operation_descriptor)][0]
                == identity_descriptor
            ),
            None,
        )
        if inverse_descriptor is None:
            raise SymmetryContextInvariantError(
                "symmetry.context.group_invalid",
                "symmetry operation has no two-sided inverse",
                evidence=(("operation_descriptor", operation_descriptor),),
            )
        inverses[operation_descriptor] = inverse_descriptor

    maximum_residual, maximum_reported_bound, recovered = _validate_metric(
        ordered,
        cell,
        metric_tolerance,
    )
    return (
        ordered,
        products,
        inverses,
        identity_descriptor,
        maximum_residual,
        maximum_reported_bound,
        recovered,
    )


@dataclass(frozen=True, slots=True)
class SymmetryContext:
    """A validated exact symmetry action in one numerical direct basis."""

    operations: tuple[AffineOperation, ...]
    operation_keys: tuple[str, ...]
    basis_convention: DirectBasisConvention
    cell_fingerprint: str
    symmetry_action_fingerprint: str
    fingerprint: str
    setting_id: str | None
    source_kind: SymmetrySourceKind
    metric_tolerance: float
    diagnostics: tuple[Diagnostic, ...]
    provenance: tuple[tuple[str, object], ...]
    identity_operation_key: str
    _operation_lookup: Mapping[str, AffineOperation] = field(
        repr=False,
        compare=False,
        hash=False,
    )
    _operation_product_lookup: Mapping[
        tuple[str, str],
        tuple[str, tuple[int, int, int]],
    ] = field(repr=False, compare=False, hash=False)
    _inverse_operation_lookup: Mapping[str, str] = field(
        repr=False,
        compare=False,
        hash=False,
    )

    @classmethod
    def _build(
        cls,
        operations: Iterable[AffineOperation],
        cell: UnitCell,
        *,
        setting_id: str | None,
        source_kind: SymmetrySourceKind,
        diagnostics: tuple[Diagnostic, ...],
        provenance: tuple[tuple[str, object], ...],
        metric_tolerance: float,
    ) -> "SymmetryContext":
        (
            canonical,
            descriptor_products,
            descriptor_inverses,
            identity_descriptor,
            maximum_metric_residual,
            maximum_reported_metric_bound,
            metric_recovered,
        ) = (
            _validate_and_canonicalize_operations(
                operations,
                cell,
                metric_tolerance,
            )
        )
        effective_metric_tolerance = max(
            metric_tolerance,
            maximum_reported_metric_bound,
        )
        if metric_recovered:
            diagnostics = (
                *diagnostics,
                Diagnostic(
                    Severity.WARNING,
                    "symmetry.context.metric_within_reported_precision",
                    "Symmetry metric compatibility was accepted within the "
                    "reported cell-parameter precision.",
                    recovery=(
                        f"configured tolerance={metric_tolerance:.8g}; "
                        f"reported bound={maximum_reported_metric_bound:.8g}; "
                        f"maximum residual={maximum_metric_residual:.8g}"
                    ),
                ),
            )
        provenance = (
            *provenance,
            ("configured_metric_tolerance", metric_tolerance),
            ("reported_metric_tolerance", maximum_reported_metric_bound),
            ("maximum_metric_residual", maximum_metric_residual),
        )
        keys = tuple(canonical_operation_key(operation) for operation in canonical)
        descriptors = tuple(_operation_descriptor(operation) for operation in canonical)
        key_by_descriptor = dict(zip(descriptors, keys, strict=True))
        product_lookup = MappingProxyType(
            {
                (key_by_descriptor[left], key_by_descriptor[right]): (
                    key_by_descriptor[product],
                    carry,
                )
                for (left, right), (product, carry) in descriptor_products.items()
            }
        )
        inverse_lookup = MappingProxyType(
            {
                key_by_descriptor[operation]: key_by_descriptor[inverse]
                for operation, inverse in descriptor_inverses.items()
            }
        )
        basis = DirectBasisConvention.FRACTIONAL_DIRECT
        action_fingerprint = _digest(
            {
                "basis_convention": basis.value,
                "operations": descriptors,
            }
        )
        cell_digest = _cell_fingerprint(cell)
        fingerprint = _digest(
            {
                "symmetry_action_fingerprint": action_fingerprint,
                "cell_fingerprint": cell_digest,
                "setting_id": setting_id,
                "source_kind": source_kind.value,
                "metric_tolerance": effective_metric_tolerance,
                "diagnostics": tuple(
                    (item.severity.value, item.code, item.message, item.recovery)
                    for item in diagnostics
                ),
                "provenance": provenance,
            }
        )
        return cls(
            operations=canonical,
            operation_keys=keys,
            basis_convention=basis,
            cell_fingerprint=cell_digest,
            symmetry_action_fingerprint=action_fingerprint,
            fingerprint=fingerprint,
            setting_id=setting_id,
            source_kind=source_kind,
            metric_tolerance=effective_metric_tolerance,
            diagnostics=diagnostics,
            provenance=provenance,
            identity_operation_key=key_by_descriptor[identity_descriptor],
            _operation_lookup=MappingProxyType(dict(zip(keys, canonical, strict=True))),
            _operation_product_lookup=product_lookup,
            _inverse_operation_lookup=inverse_lookup,
        )

    @classmethod
    def from_operations(
        cls,
        operations: Iterable[AffineOperation],
        cell: UnitCell,
        *,
        provenance: tuple[tuple[str, object], ...] = (),
        metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    ) -> "SymmetryContext":
        diagnostics = (
            Diagnostic(
                Severity.INFO,
                "symmetry.setting_unresolved",
                "valid explicit symmetry operations are not tied to a catalog setting",
            ),
        )
        return cls._build(
            operations,
            cell,
            setting_id=None,
            source_kind=SymmetrySourceKind.VALID_EXPLICIT_OPERATIONS,
            diagnostics=diagnostics,
            provenance=tuple(provenance),
            metric_tolerance=metric_tolerance,
        )

    @classmethod
    def from_definition(
        cls,
        definition: SpaceGroupDefinition,
        cell: UnitCell,
        *,
        metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    ) -> "SymmetryContext":
        if not isinstance(definition, SpaceGroupDefinition):
            raise TypeError("definition must be SpaceGroupDefinition")
        fallback = definition.provenance in {"identity_fallback", "unreported_identity"}
        source_kind = (
            SymmetrySourceKind.EXPLICIT_IDENTITY_FALLBACK
            if fallback
            else SymmetrySourceKind.VALID_EXPLICIT_OPERATIONS
        )
        diagnostics = (
            Diagnostic(
                Severity.WARNING if fallback else Severity.INFO,
                "symmetry.identity_fallback" if fallback else "symmetry.setting_unresolved",
                (
                    "an explicitly requested identity symmetry fallback is in use"
                    if fallback
                    else "valid explicit symmetry operations are not tied to a catalog setting"
                ),
            ),
        )
        provenance: tuple[tuple[str, object], ...] = (
            ("source", "space_group_definition"),
            ("reported_provenance", definition.provenance),
            ("number", definition.number),
            ("hall_symbol", definition.hall_symbol),
            ("setting", definition.setting),
            ("origin_choice", definition.origin_choice),
        )
        return cls._build(
            definition.operations,
            cell,
            setting_id=None,
            source_kind=source_kind,
            diagnostics=diagnostics,
            provenance=provenance,
            metric_tolerance=metric_tolerance,
        )

    @classmethod
    def from_setting(
        cls,
        setting: SpaceGroupSetting,
        cell: UnitCell,
        *,
        metric_tolerance: float = DEFAULT_METRIC_TOLERANCE,
    ) -> "SymmetryContext":
        if not isinstance(setting, SpaceGroupSetting):
            raise TypeError("setting must be SpaceGroupSetting")
        provenance: tuple[tuple[str, object], ...] = (
            ("source", "space_group_setting"),
            ("setting_id", setting.setting_id),
            ("hall_symbol", setting.hall_symbol),
            ("catalog_number", setting.number),
        )
        return cls._build(
            setting.symmetry_operations,
            cell,
            setting_id=str(setting.setting_id),
            source_kind=SymmetrySourceKind.CATALOG_SETTING,
            diagnostics=(),
            provenance=provenance,
            metric_tolerance=metric_tolerance,
        )

    def operation_by_key(self, operation_key: str) -> AffineOperation:
        """Resolve a canonical operation key within this context."""

        try:
            return self._operation_lookup[operation_key]
        except KeyError as exc:
            raise KeyError(operation_key) from exc

    def compose_operation_keys(
        self,
        left_operation_key: str,
        right_operation_key: str,
    ) -> tuple[str, tuple[int, int, int]]:
        """Return normalized product key and its exact fractional carry."""

        try:
            return self._operation_product_lookup[
                (left_operation_key, right_operation_key)
            ]
        except KeyError as exc:
            raise KeyError((left_operation_key, right_operation_key)) from exc

    def inverse_operation_key(self, operation_key: str) -> str:
        """Return the exact inverse operation key in this context."""

        try:
            return self._inverse_operation_lookup[operation_key]
        except KeyError as exc:
            raise KeyError(operation_key) from exc


__all__ = [
    "DEFAULT_METRIC_TOLERANCE",
    "DirectBasisConvention",
    "SymmetryContext",
    "SymmetryContextInvariantError",
    "SymmetrySourceKind",
    "canonical_operation_key",
]
