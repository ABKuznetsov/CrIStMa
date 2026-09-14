"""Immutable conservative secondary-building-unit results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic
from cristma.structure import SiteComponent

from .models import ResolutionStatus

if TYPE_CHECKING:
    from .reticular_models import ReticularPrecursorGraph


class SBUAtomSourceRole(StrEnum):
    NODE = "node"
    TERMINAL_LIGAND = "terminal_ligand"


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _sorted_ids(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} IDs must not be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} IDs must be unique and sorted")


def _sorted_key_values(values: tuple[tuple[str, object], ...], name: str) -> None:
    keys = tuple(key for key, _ in values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{name} keys must be unique and sorted")


def _member_key(member: "SBUAtomMember") -> tuple[object, ...]:
    return (
        member.source_role.value,
        member.independent_site_id,
        member.periodic_relation,
    )


@dataclass(frozen=True, slots=True)
class SBUAtomMember:
    independent_site_id: str
    periodic_relation: PeriodicSymmetryRelation
    multiplicity_per_sbu: int
    components: tuple[SiteComponent, ...]
    source_role: SBUAtomSourceRole

    def __post_init__(self) -> None:
        if not isinstance(self.independent_site_id, str) or not self.independent_site_id:
            raise ValueError("SBU atom site ID must not be empty")
        if not isinstance(self.periodic_relation, PeriodicSymmetryRelation):
            raise TypeError("SBU atom member requires an exact periodic relation")
        _positive_integer(self.multiplicity_per_sbu, "SBU atom multiplicity")
        if not self.components or any(
            not isinstance(component, SiteComponent) for component in self.components
        ):
            raise ValueError("SBU atom member requires site components")
        labels = tuple(component.species.label for component in self.components)
        if len(labels) != len(set(labels)):
            raise ValueError("SBU atom member components must be unique")
        if not isinstance(self.source_role, SBUAtomSourceRole):
            raise TypeError("SBU atom source role must be SBUAtomSourceRole")


@dataclass(frozen=True, slots=True)
class SecondaryBuildingUnitOrbit:
    sbu_orbit_id: str
    node_orbit_ids: tuple[str, ...]
    terminal_ligand_orbit_ids: tuple[str, ...]
    source_connection_orbit_ids: tuple[str, ...]
    source_contact_orbit_ids: tuple[str, ...]
    atom_members: tuple[SBUAtomMember, ...]
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sbu_orbit_id, str) or not self.sbu_orbit_id:
            raise ValueError("SBU orbit ID must not be empty")
        _sorted_ids(self.node_orbit_ids, "SBU node orbit")
        if len(self.node_orbit_ids) != 1:
            raise ValueError("conservative SBU requires exactly one node orbit")
        _sorted_ids(self.terminal_ligand_orbit_ids, "SBU terminal ligand orbit")
        _sorted_ids(self.source_connection_orbit_ids, "SBU source connection orbit")
        _sorted_ids(self.source_contact_orbit_ids, "SBU source contact orbit")
        if not self.atom_members:
            raise ValueError("SBU requires atom members")
        if self.atom_members != tuple(sorted(self.atom_members, key=_member_key)):
            raise ValueError("SBU atom members must be deterministically sorted")
        member_keys = tuple(_member_key(item) for item in self.atom_members)
        if len(member_keys) != len(set(member_keys)):
            raise ValueError("SBU atom members must be unique")
        if sum(
            item.source_role is SBUAtomSourceRole.NODE for item in self.atom_members
        ) != 1:
            raise ValueError("conservative SBU requires one representative node member")
        _positive_integer(self.multiplicity_in_reference_cell, "SBU multiplicity")
        if not isinstance(self.status, ResolutionStatus):
            raise TypeError("SBU status must be ResolutionStatus")
        _sorted_key_values(self.provenance, "SBU provenance")

    @property
    def geometric_atom_count(self) -> int:
        return sum(item.multiplicity_per_sbu for item in self.atom_members)

    @property
    def occupancy_weighted_composition(self) -> tuple[tuple[str, float], ...]:
        counts: dict[str, float] = {}
        for member in self.atom_members:
            for component in member.components:
                label = component.species.label
                counts[label] = counts.get(label, 0.0) + (
                    member.multiplicity_per_sbu * float(component.occupancy.value)
                )
        return tuple(sorted(counts.items()))


@dataclass(frozen=True, slots=True)
class SecondaryBuildingUnitResult:
    _precursor_graph: "ReticularPrecursorGraph" = field(repr=False, compare=False)
    sbu_orbits: tuple[SecondaryBuildingUnitOrbit, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]
    precursor_graph_fingerprint: str

    def __post_init__(self) -> None:
        if not self.precursor_graph_fingerprint:
            raise ValueError("SBU result requires precursor fingerprint")
        if self.precursor_graph_fingerprint != self._precursor_graph.fingerprint:
            raise ValueError("SBU result precursor fingerprint disagrees")
        identifiers = tuple(item.sbu_orbit_id for item in self.sbu_orbits)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("SBU orbit IDs must be unique and sorted")
        if any(
            item.status is ResolutionStatus.INCOMPLETE for item in self.sbu_orbits
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete SBU must propagate to result status")
        _sorted_key_values(self.configuration, "SBU result configuration")
        _sorted_key_values(self.provenance, "SBU result provenance")

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "precursor_graph_fingerprint": self.precursor_graph_fingerprint,
                "sbus": tuple(
                    (
                        item.sbu_orbit_id,
                        item.node_orbit_ids,
                        item.terminal_ligand_orbit_ids,
                        tuple(
                            (
                                member.independent_site_id,
                                member.periodic_relation.operation_key,
                                member.periodic_relation.lattice_translation,
                                member.multiplicity_per_sbu,
                                tuple(
                                    (
                                        component.species.label,
                                        float(component.occupancy.value).hex(),
                                    )
                                    for component in member.components
                                ),
                                member.source_role.value,
                            )
                            for member in item.atom_members
                        ),
                    )
                    for item in self.sbu_orbits
                ),
                "configuration": self.configuration,
            }
        )


__all__ = [
    "SBUAtomMember",
    "SBUAtomSourceRole",
    "SecondaryBuildingUnitOrbit",
    "SecondaryBuildingUnitResult",
]
