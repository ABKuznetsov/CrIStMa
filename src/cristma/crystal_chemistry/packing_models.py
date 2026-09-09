"""Immutable results for exact orbit-first molecular packing graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.diagnostics import Diagnostic

from .models import ResolutionStatus

if TYPE_CHECKING:
    from .ch_pi import CHPiAnalysisResult
    from .intermolecular_models import IntermolecularContactAnalysisResult
    from .pi_stacking import PiStackingAnalysisResult


Translation = tuple[int, int, int]


class PackingInteractionKind(StrEnum):
    VAN_DER_WAALS_CONTACT = "van_der_waals_contact"
    HYDROGEN_BOND = "hydrogen_bond"
    PI_STACKING = "pi_stacking"
    CH_PI = "ch_pi"


@dataclass(frozen=True, slots=True, order=True)
class PackingInteractionRef:
    kind: PackingInteractionKind
    source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PackingInteractionKind):
            raise TypeError("packing interaction kind must be PackingInteractionKind")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("packing interaction source ID must not be empty")


@dataclass(frozen=True, slots=True)
class PackingConnectionOrbit:
    packing_connection_orbit_id: str
    first_molecule_orbit_id: str
    second_molecule_orbit_id: str
    canonical_molecule_relation: PeriodicSymmetryRelation
    interaction_refs: tuple[PackingInteractionRef, ...]
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.packing_connection_orbit_id,
            self.first_molecule_orbit_id,
            self.second_molecule_orbit_id,
        )
        if not all(isinstance(value, str) and value for value in identities):
            raise ValueError("packing connection identities must not be empty")
        if self.first_molecule_orbit_id > self.second_molecule_orbit_id:
            raise ValueError("packing connection endpoints must use canonical order")
        if not isinstance(self.canonical_molecule_relation, PeriodicSymmetryRelation):
            raise TypeError("packing connection requires an exact periodic relation")
        if not self.interaction_refs:
            raise ValueError("packing connection requires interaction evidence")
        if self.interaction_refs != tuple(sorted(set(self.interaction_refs))):
            raise ValueError("packing interaction refs must be unique and sorted")
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ):
            raise TypeError("packing connection multiplicity must be an integer")
        if self.multiplicity_in_reference_cell <= 0:
            raise ValueError("packing connection multiplicity must be positive")
        _validate_key_values(self.provenance, "packing connection provenance")


@dataclass(frozen=True, slots=True)
class MolecularPackingComponent:
    packing_component_id: str
    molecule_orbit_ids: tuple[str, ...]
    connection_orbit_ids: tuple[str, ...]
    image_relations: tuple[tuple[str, PeriodicSymmetryRelation], ...]
    closure_relations: tuple[PeriodicSymmetryRelation, ...]
    periodic_rank: int
    periodic_generators: tuple[Translation, ...]
    status: ResolutionStatus

    def __post_init__(self) -> None:
        if not isinstance(self.packing_component_id, str) or not self.packing_component_id:
            raise ValueError("packing component ID must not be empty")
        if not self.molecule_orbit_ids:
            raise ValueError("packing component requires a molecule orbit")
        _validate_sorted_ids(self.molecule_orbit_ids, "packing component molecule")
        _validate_sorted_ids(self.connection_orbit_ids, "packing component connection")
        image_ids = tuple(item[0] for item in self.image_relations)
        if image_ids != self.molecule_orbit_ids:
            raise ValueError("packing component requires one image relation per molecule")
        if self.closure_relations != tuple(sorted(set(self.closure_relations))):
            raise ValueError("packing component closure relations must be unique and sorted")
        if isinstance(self.periodic_rank, bool) or not isinstance(self.periodic_rank, int):
            raise TypeError("packing periodic rank must be an integer")
        if self.periodic_rank not in range(4):
            raise ValueError("packing periodic rank must lie within [0, 3]")
        if self.periodic_rank != len(self.periodic_generators):
            raise ValueError("packing periodic rank and generator count disagree")
        for generator in self.periodic_generators:
            if (
                len(generator) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in generator)
            ):
                raise TypeError("packing periodic generators must be integer triples")


@dataclass(frozen=True, slots=True)
class MolecularPackingGraph:
    _intermolecular_analysis: "IntermolecularContactAnalysisResult" = field(
        repr=False, compare=False
    )
    _pi_stacking_analysis: "PiStackingAnalysisResult" = field(
        repr=False, compare=False
    )
    _ch_pi_analysis: "CHPiAnalysisResult" = field(repr=False, compare=False)
    molecule_orbit_ids: tuple[str, ...]
    connection_orbits: tuple[PackingConnectionOrbit, ...]
    components: tuple[MolecularPackingComponent, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        _validate_sorted_ids(self.molecule_orbit_ids, "packing graph molecule")
        connection_ids = tuple(
            item.packing_connection_orbit_id for item in self.connection_orbits
        )
        component_ids = tuple(item.packing_component_id for item in self.components)
        _validate_sorted_ids(connection_ids, "packing graph connection")
        _validate_sorted_ids(component_ids, "packing graph component")
        known_molecules = set(self.molecule_orbit_ids)
        known_connections = set(connection_ids)
        if any(
            connection.first_molecule_orbit_id not in known_molecules
            or connection.second_molecule_orbit_id not in known_molecules
            for connection in self.connection_orbits
        ):
            raise ValueError("packing connection references an unknown molecule")
        reported_molecules = [
            molecule_id
            for component in self.components
            for molecule_id in component.molecule_orbit_ids
        ]
        if sorted(reported_molecules) != list(self.molecule_orbit_ids):
            raise ValueError("packing components must partition molecule orbits")
        reported_connections = [
            connection_id
            for component in self.components
            for connection_id in component.connection_orbit_ids
        ]
        if sorted(reported_connections) != list(connection_ids):
            unknown = set(reported_connections) - known_connections
            if unknown:
                raise ValueError("packing component references an unknown connection")
            raise ValueError("packing components must partition connection orbits")
        component_by_connection = {
            connection_id: set(component.molecule_orbit_ids)
            for component in self.components
            for connection_id in component.connection_orbit_ids
        }
        if any(
            {
                connection.first_molecule_orbit_id,
                connection.second_molecule_orbit_id,
            } - component_by_connection[connection.packing_connection_orbit_id]
            for connection in self.connection_orbits
        ):
            raise ValueError("packing connection endpoints lie outside their component")
        if (
            any(item.status is ResolutionStatus.INCOMPLETE for item in self.connection_orbits)
            or any(item.status is ResolutionStatus.INCOMPLETE for item in self.components)
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete packing result must propagate")
        _validate_key_values(self.configuration, "packing graph configuration")
        _validate_key_values(self.provenance, "packing graph provenance")


def _validate_sorted_ids(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} IDs must not be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} IDs must be unique and sorted")


def _validate_key_values(values: tuple[tuple[str, object], ...], name: str) -> None:
    keys = tuple(key for key, _ in values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{name} keys must be unique and sorted")


__all__ = [
    "MolecularPackingComponent",
    "MolecularPackingGraph",
    "PackingConnectionOrbit",
    "PackingInteractionKind",
    "PackingInteractionRef",
]
