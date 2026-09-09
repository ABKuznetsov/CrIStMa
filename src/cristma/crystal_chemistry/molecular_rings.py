"""Immutable models for bounded orbit-first molecular ring analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.diagnostics import Diagnostic

from .models import ResolutionStatus


@dataclass(frozen=True, slots=True)
class MolecularRingSearchPolicy:
    """Explicit resource bounds for molecular shortest-return searches."""

    maximum_ring_size: int = 12
    maximum_states_per_bond: int = 50_000
    maximum_paths_per_bond: int = 4_096

    def __post_init__(self) -> None:
        for name in (
            "maximum_ring_size",
            "maximum_states_per_bond",
            "maximum_paths_per_bond",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_ring_size < 3:
            raise ValueError("maximum_ring_size must be at least three")

    def get_config(self) -> dict[str, int]:
        return {
            "maximum_paths_per_bond": self.maximum_paths_per_bond,
            "maximum_ring_size": self.maximum_ring_size,
            "maximum_states_per_bond": self.maximum_states_per_bond,
        }

    def clone(self, **changes: object) -> "MolecularRingSearchPolicy":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, order=True)
class MolecularAtomOrbitRef:
    """One physical molecular vertex in an exact lifted quotient graph."""

    independent_site_id: str
    periodic_relation: PeriodicSymmetryRelation

    def __post_init__(self) -> None:
        if not isinstance(self.independent_site_id, str) or not self.independent_site_id:
            raise ValueError("molecular atom-orbit reference ID must not be empty")


@dataclass(frozen=True, slots=True)
class MolecularRingOrbit:
    """One symmetry-unique chordless ring inside a discrete molecule orbit."""

    ring_orbit_id: str
    parent_component_orbit_id: str
    atom_refs: tuple[MolecularAtomOrbitRef, ...]
    bond_orbit_ids: tuple[str, ...]
    multiplicity_in_reference_cell: int
    centroid_cartesian: tuple[float, float, float]
    normal_cartesian: tuple[float, float, float] | None
    planarity_rms_angstrom: float | None
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.ring_orbit_id or not self.parent_component_orbit_id:
            raise ValueError("molecular ring identities must not be empty")
        if len(self.atom_refs) < 3 or len(set(self.atom_refs)) < 3:
            raise ValueError("molecular ring requires three distinct physical vertices")
        if len(self.bond_orbit_ids) != len(self.atom_refs):
            raise ValueError("molecular ring requires one bond per atom reference")
        if any(not isinstance(value, str) or not value for value in self.bond_orbit_ids):
            raise ValueError("molecular ring bond IDs must not be empty")
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ):
            raise TypeError("molecular ring multiplicity must be an integer")
        if self.multiplicity_in_reference_cell <= 0:
            raise ValueError("molecular ring multiplicity must be positive")
        if len(self.centroid_cartesian) != 3 or not all(
            math.isfinite(value) for value in self.centroid_cartesian
        ):
            raise ValueError("molecular ring centroid must contain three finite values")
        geometry_missing = (
            self.normal_cartesian is None or self.planarity_rms_angstrom is None
        )
        if (self.normal_cartesian is None) != (self.planarity_rms_angstrom is None):
            raise ValueError("molecular ring normal and planarity must be present together")
        if geometry_missing:
            if self.status is ResolutionStatus.RESOLVED:
                raise ValueError("resolved ring requires non-degenerate geometry")
            if not self.diagnostics:
                raise ValueError("ring without geometry requires a diagnostic")
            return
        normal = self.normal_cartesian
        if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
            raise ValueError("molecular ring normal must contain three finite values")
        norm = math.sqrt(math.fsum(value * value for value in normal))
        if not math.isclose(norm, 1.0, abs_tol=1e-12):
            raise ValueError("molecular ring normal must be a unit vector")
        if not math.isfinite(self.planarity_rms_angstrom) or self.planarity_rms_angstrom < 0:
            raise ValueError("molecular ring planarity RMS must be finite and non-negative")

    @property
    def size(self) -> int:
        return len(self.atom_refs)


__all__ = [
    "MolecularAtomOrbitRef",
    "MolecularRingOrbit",
    "MolecularRingSearchPolicy",
]
