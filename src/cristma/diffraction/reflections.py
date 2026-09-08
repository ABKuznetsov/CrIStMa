"""Deterministic crystallographic reflection generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from cristma.core import UnitCell
from cristma.crystallography import SpaceGroupSetting
from cristma.diagnostics import Diagnostic, Severity
from cristma.symmetry import AffineOperation

from .diagnostics import (
    DiffractionInvariantError,
    INCOMPATIBLE_CELL_AND_SETTING,
    ORBIT_METRIC_MISMATCH,
    SEARCH_LIMIT_REACHED,
)
from .extinction import ExtinctionAnalyzer, reciprocal_action
from .models import (
    MillerIndex,
    Reflection,
    ReflectionGenerationProvenance,
    ReflectionProvenance,
    ReflectionSet,
    ReflectionSetStatus,
)
from .reciprocal import ReciprocalMetric, cell_fingerprint, enumerate_integer_ellipsoid


GENERATOR_METHOD = "bounded_integer_ellipsoid"
GENERATOR_VERSION = "1"
RECIPROCAL_CONVENTION = "h^T G* h; no 2*pi"


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _cell_values(cell: UnitCell) -> tuple[float, float, float, float, float, float]:
    measured = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    return tuple(float(item.value) for item in measured)  # type: ignore[arg-type]


def _rotation_key(operation: AffineOperation) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(value for value in row) for row in operation.rotation)


def _reflection_id(setting_id: int, hkl: MillerIndex) -> str:
    return f"reflection:{setting_id}:{hkl.h}:{hkl.k}:{hkl.l}"


@dataclass(frozen=True, slots=True)
class ReflectionGenerator:
    max_candidates: int | None = 1_000_000
    boundary_tolerance: float = 1e-12
    metric_compatibility_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if self.max_candidates is not None and (
            isinstance(self.max_candidates, bool)
            or not isinstance(self.max_candidates, int)
            or self.max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer or None")
        _positive_finite(self.boundary_tolerance, "boundary_tolerance")
        _positive_finite(self.metric_compatibility_tolerance, "metric_compatibility_tolerance")

    def get_config(self) -> dict[str, int | float | None]:
        return {
            "max_candidates": self.max_candidates,
            "boundary_tolerance": self.boundary_tolerance,
            "metric_compatibility_tolerance": self.metric_compatibility_tolerance,
        }

    def clone(self, **changes: Any) -> ReflectionGenerator:
        return replace(self, **changes)

    def _rotations(self, setting: SpaceGroupSetting) -> tuple[AffineOperation, ...]:
        operation_ids = tuple(operation.id for operation in setting.symmetry_operations)
        if any(not isinstance(value, str) or not value.strip() for value in operation_ids):
            raise ValueError("every symmetry operation must have a non-empty ID")
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("symmetry operation IDs must be unique within a setting")
        unique: dict[tuple[tuple[object, ...], ...], AffineOperation] = {}
        for operation in setting.symmetry_operations:
            unique.setdefault(_rotation_key(operation), operation)
        return tuple(unique[key] for key in sorted(unique, key=repr))

    def _check_metric(
        self,
        cell: UnitCell,
        reciprocal: ReciprocalMetric,
        setting: SpaceGroupSetting,
        rotations: tuple[AffineOperation, ...],
    ) -> None:
        scale = max(1.0, float(np.max(np.abs(reciprocal.metric))))
        for operation in rotations:
            rotation = np.asarray(operation.rotation, dtype=float)
            transformed = rotation @ reciprocal.metric @ rotation.T
            residual = float(np.max(np.abs(transformed - reciprocal.metric))) / scale
            if residual > self.metric_compatibility_tolerance:
                raise DiffractionInvariantError(
                    INCOMPATIBLE_CELL_AND_SETTING,
                    "unit-cell metric is incompatible with a setting rotation",
                    {
                        "space_group_setting_id": setting.setting_id,
                        "cell": _cell_values(cell),
                        "operation_id": operation.id,
                        "rotation": operation.rotation,
                        "metric_residual": residual,
                    },
                )

    @staticmethod
    def _orbit(seed: MillerIndex, rotations: tuple[AffineOperation, ...]) -> tuple[MillerIndex, ...]:
        return tuple(sorted({reciprocal_action(operation, seed) for operation in rotations}))

    def _validate_orbit_metric(
        self,
        orbit: tuple[MillerIndex, ...],
        reciprocal: ReciprocalMetric,
        d_min: float,
        setting: SpaceGroupSetting,
    ) -> tuple[float, float]:
        representative = orbit[0]
        norm_squared = reciprocal.norm_squared(representative)
        boundary = 1.0 / (d_min * d_min) * (1.0 + self.boundary_tolerance)
        for member in orbit:
            member_norm = reciprocal.norm_squared(member)
            if (
                not math.isclose(
                    member_norm,
                    norm_squared,
                    rel_tol=self.metric_compatibility_tolerance,
                    abs_tol=self.metric_compatibility_tolerance,
                )
                or member_norm > boundary
            ):
                raise DiffractionInvariantError(
                    ORBIT_METRIC_MISMATCH,
                    "reciprocal orbit does not preserve metric or d_min boundary",
                    {
                        "space_group_setting_id": setting.setting_id,
                        "representative_hkl": representative.as_tuple(),
                        "member_hkl": member.as_tuple(),
                        "representative_norm_squared": norm_squared,
                        "member_norm_squared": member_norm,
                    },
                )
        norm = math.sqrt(norm_squared)
        return norm, 1.0 / norm

    def _extinction_for_orbit(
        self,
        orbit: tuple[MillerIndex, ...],
        setting: SpaceGroupSetting,
        analyzer: ExtinctionAnalyzer,
    ):
        # Systematic absence is invariant under the reciprocal point-group
        # action.  Re-evaluating every member would repeat the same exact
        # phase-character calculation in the hottest generation loop.
        return analyzer.analyze(orbit[0], setting)

    def generate(
        self,
        cell: UnitCell,
        space_group: SpaceGroupSetting,
        d_min: float,
    ) -> ReflectionSet:
        if not isinstance(cell, UnitCell):
            raise TypeError("cell must be UnitCell")
        if not isinstance(space_group, SpaceGroupSetting):
            raise TypeError("space_group must be SpaceGroupSetting")
        d_min = _positive_finite(d_min, "d_min")
        reciprocal = ReciprocalMetric.from_cell(cell)
        rotations = self._rotations(space_group)
        self._check_metric(cell, reciprocal, space_group, rotations)
        enumeration = enumerate_integer_ellipsoid(
            reciprocal,
            d_min,
            self.max_candidates,
            self.boundary_tolerance,
        )

        analyzer = ExtinctionAnalyzer()
        visited: set[MillerIndex] = set()
        reflection_data: dict[MillerIndex, tuple[tuple[MillerIndex, ...], str | None]] = {}
        for seed in enumeration.indices:
            if seed in visited:
                continue
            orbit = self._orbit(seed, rotations)
            representative = orbit[0]
            mate_orbit = self._orbit(representative.negated(), rotations)
            mate_representative = mate_orbit[0]
            if mate_representative == representative:
                reflection_data[representative] = (orbit, None)
                visited.update(orbit)
                continue
            reflection_data[representative] = (
                orbit,
                _reflection_id(space_group.setting_id, mate_representative),
            )
            reflection_data[mate_representative] = (
                mate_orbit,
                _reflection_id(space_group.setting_id, representative),
            )
            visited.update(orbit)
            visited.update(mate_orbit)

        reflections = []
        for representative in sorted(reflection_data):
            orbit, mate_id = reflection_data[representative]
            norm, spacing = self._validate_orbit_metric(
                orbit, reciprocal, d_min, space_group
            )
            reflections.append(
                Reflection(
                    reflection_id=_reflection_id(space_group.setting_id, representative),
                    representative_hkl=representative,
                    equivalent_hkls=orbit,
                    d_spacing=spacing,
                    reciprocal_norm=norm,
                    multiplicity_crystallographic=len(orbit),
                    friedel_mate_id=mate_id,
                    extinction=self._extinction_for_orbit(orbit, space_group, analyzer),
                    provenance=ReflectionProvenance(
                        space_group.setting_id,
                        "reciprocal_point_group_orbit",
                    ),
                )
            )

        status = (
            ReflectionSetStatus.COMPLETE
            if enumeration.complete
            else ReflectionSetStatus.INCOMPLETE
        )
        diagnostics = ()
        if status is ReflectionSetStatus.INCOMPLETE:
            diagnostics = (
                Diagnostic(
                    Severity.WARNING,
                    SEARCH_LIMIT_REACHED,
                    f"reflection search stopped after {enumeration.integer_points_tested} integer candidates",
                    recovery="increase max_candidates or use a larger d_min",
                ),
            )
        provenance = ReflectionGenerationProvenance(
            method=GENERATOR_METHOD,
            version=GENERATOR_VERSION,
            space_group_setting_id=space_group.setting_id,
            cell_fingerprint=cell_fingerprint(cell),
            d_min=d_min,
            reciprocal_convention=RECIPROCAL_CONVENTION,
            boundary_tolerance=self.boundary_tolerance,
            metric_compatibility_tolerance=self.metric_compatibility_tolerance,
            max_candidates=self.max_candidates,
            integer_points_tested=enumeration.integer_points_tested,
            reflections_within_d_min=enumeration.reflections_within_d_min,
            orbits_created=len(reflections),
            status=status,
        )
        return ReflectionSet(
            tuple(reflections),
            space_group.setting_id,
            d_min,
            status,
            diagnostics,
            provenance,
        )


__all__ = ["ReflectionGenerator"]
