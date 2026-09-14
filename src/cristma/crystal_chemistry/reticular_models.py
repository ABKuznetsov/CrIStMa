"""Immutable results for orbit-first reticular precursor graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic

from .models import ResolutionStatus

if TYPE_CHECKING:
    from .contact_analysis import ContactAnalysisResult
    from .molecular import MolecularAnalysisResult


Translation = tuple[int, int, int]


class LigandRole(StrEnum):
    TERMINAL = "terminal"
    LINKER = "linker"


class ReticularComponentKind(StrEnum):
    DISCRETE_COMPLEX = "discrete_complex"
    PERIODIC_CHAIN = "periodic_chain"
    PERIODIC_LAYER = "periodic_layer"
    PERIODIC_FRAMEWORK = "periodic_framework"


_KIND_BY_RANK = {
    0: ReticularComponentKind.DISCRETE_COMPLEX,
    1: ReticularComponentKind.PERIODIC_CHAIN,
    2: ReticularComponentKind.PERIODIC_LAYER,
    3: ReticularComponentKind.PERIODIC_FRAMEWORK,
}


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must not be empty")


def _validate_sorted_ids(
    values: tuple[str, ...],
    name: str,
    *,
    required: bool = False,
) -> None:
    if required and not values:
        raise ValueError(f"{name} IDs must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} IDs must not be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} IDs must be unique and sorted")


def _validate_multiplicity(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} multiplicity must be an integer")
    if value <= 0:
        raise ValueError(f"{name} multiplicity must be positive")


def _validate_key_values(values: tuple[tuple[str, object], ...], name: str) -> None:
    keys = tuple(key for key, _ in values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{name} keys must be unique and sorted")


@dataclass(frozen=True, slots=True)
class CoordinationNodeOrbit:
    node_orbit_id: str
    center_independent_site_id: str
    source_shell_orbit_ids: tuple[str, ...]
    source_resolved_contact_orbit_ids: tuple[str, ...]
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.node_orbit_id, "coordination node ID")
        _validate_id(self.center_independent_site_id, "coordination node site ID")
        _validate_sorted_ids(self.source_shell_orbit_ids, "coordination node shell")
        _validate_sorted_ids(
            self.source_resolved_contact_orbit_ids,
            "coordination node contact",
            required=True,
        )
        _validate_multiplicity(
            self.multiplicity_in_reference_cell, "coordination node"
        )
        _validate_key_values(self.provenance, "coordination node provenance")


@dataclass(frozen=True, slots=True)
class LigandComponentOrbit:
    ligand_orbit_id: str
    source_component_orbit_id: str
    independent_site_ids: tuple[str, ...]
    bond_orbit_ids: tuple[str, ...]
    connection_orbit_ids: tuple[str, ...]
    role: LigandRole
    attachment_count_per_ligand_instance: int
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.ligand_orbit_id, "ligand orbit ID")
        _validate_id(self.source_component_orbit_id, "source molecular component ID")
        _validate_sorted_ids(
            self.independent_site_ids, "ligand independent site", required=True
        )
        _validate_sorted_ids(self.bond_orbit_ids, "ligand bond")
        _validate_sorted_ids(
            self.connection_orbit_ids, "ligand connection", required=True
        )
        if not isinstance(self.role, LigandRole):
            raise TypeError("ligand role must be LigandRole")
        if (
            isinstance(self.attachment_count_per_ligand_instance, bool)
            or not isinstance(self.attachment_count_per_ligand_instance, int)
        ):
            raise TypeError("ligand attachment count must be an integer")
        if self.attachment_count_per_ligand_instance <= 0:
            raise ValueError("ligand attachment count must be positive")
        expected_role = (
            LigandRole.TERMINAL
            if self.attachment_count_per_ligand_instance == 1
            else LigandRole.LINKER
        )
        if self.role is not expected_role:
            raise ValueError("ligand role and attachment count disagree")
        _validate_multiplicity(self.multiplicity_in_reference_cell, "ligand")
        _validate_key_values(self.provenance, "ligand provenance")


@dataclass(frozen=True, slots=True)
class ReticularConnectionOrbit:
    connection_orbit_id: str
    node_orbit_id: str
    ligand_orbit_id: str
    canonical_node_ligand_relation: PeriodicSymmetryRelation
    equivalent_node_ligand_relations: tuple[PeriodicSymmetryRelation, ...]
    donor_independent_site_ids: tuple[str, ...]
    source_resolved_contact_orbit_ids: tuple[str, ...]
    interpretation_ids: tuple[str, ...]
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.connection_orbit_id, "reticular connection ID")
        _validate_id(self.node_orbit_id, "reticular connection node ID")
        _validate_id(self.ligand_orbit_id, "reticular connection ligand ID")
        if not isinstance(
            self.canonical_node_ligand_relation, PeriodicSymmetryRelation
        ):
            raise TypeError("reticular connection requires an exact periodic relation")
        if (
            not self.equivalent_node_ligand_relations
            or self.equivalent_node_ligand_relations
            != tuple(sorted(set(self.equivalent_node_ligand_relations)))
        ):
            raise ValueError(
                "equivalent node-ligand relations must be non-empty, unique, and sorted"
            )
        _validate_sorted_ids(
            self.donor_independent_site_ids, "reticular donor", required=True
        )
        _validate_sorted_ids(
            self.source_resolved_contact_orbit_ids,
            "reticular source contact",
            required=True,
        )
        _validate_sorted_ids(
            self.interpretation_ids,
            "reticular interpretation",
            required=True,
        )
        _validate_multiplicity(
            self.multiplicity_in_reference_cell, "reticular connection"
        )
        _validate_key_values(self.provenance, "reticular connection provenance")


@dataclass(frozen=True, slots=True)
class ReticularComponent:
    component_id: str
    node_orbit_ids: tuple[str, ...]
    ligand_orbit_ids: tuple[str, ...]
    connection_orbit_ids: tuple[str, ...]
    image_relations: tuple[tuple[str, PeriodicSymmetryRelation], ...]
    closure_relations: tuple[PeriodicSymmetryRelation, ...]
    periodic_rank: int
    periodic_generators: tuple[Translation, ...]
    kind: ReticularComponentKind
    status: ResolutionStatus

    def __post_init__(self) -> None:
        _validate_id(self.component_id, "reticular component ID")
        _validate_sorted_ids(
            self.node_orbit_ids, "reticular component node", required=True
        )
        _validate_sorted_ids(
            self.ligand_orbit_ids, "reticular component ligand", required=True
        )
        _validate_sorted_ids(
            self.connection_orbit_ids,
            "reticular component connection",
            required=True,
        )
        expected_image_ids = tuple(sorted(
            (*(f"node:{item}" for item in self.node_orbit_ids),
             *(f"ligand:{item}" for item in self.ligand_orbit_ids))
        ))
        image_ids = tuple(item_id for item_id, _ in self.image_relations)
        if image_ids != expected_image_ids:
            raise ValueError(
                "reticular component requires one sorted image relation per endpoint"
            )
        if self.closure_relations != tuple(sorted(set(self.closure_relations))):
            raise ValueError(
                "reticular component closure relations must be unique and sorted"
            )
        if isinstance(self.periodic_rank, bool) or not isinstance(
            self.periodic_rank, int
        ):
            raise TypeError("reticular periodic rank must be an integer")
        if self.periodic_rank not in range(4):
            raise ValueError("reticular periodic rank must lie within [0, 3]")
        if self.periodic_rank != len(self.periodic_generators):
            raise ValueError("reticular periodic rank and generator count disagree")
        if self.kind is not _KIND_BY_RANK[self.periodic_rank]:
            raise ValueError("reticular periodic rank and component kind disagree")
        for generator in self.periodic_generators:
            if len(generator) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in generator
            ):
                raise TypeError("reticular periodic generators must be integer triples")


@dataclass(frozen=True, slots=True)
class ReticularPrecursorGraph:
    _contact_analysis: "ContactAnalysisResult" = field(repr=False, compare=False)
    _molecular_analysis: "MolecularAnalysisResult" = field(
        repr=False, compare=False
    )
    node_orbits: tuple[CoordinationNodeOrbit, ...]
    ligand_orbits: tuple[LigandComponentOrbit, ...]
    connection_orbits: tuple[ReticularConnectionOrbit, ...]
    components: tuple[ReticularComponent, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if (
            self._molecular_analysis.contact_analysis_fingerprint
            != self._contact_analysis.fingerprint
        ):
            raise ValueError("reticular source lineage disagrees")
        node_ids = tuple(item.node_orbit_id for item in self.node_orbits)
        ligand_ids = tuple(item.ligand_orbit_id for item in self.ligand_orbits)
        connection_ids = tuple(
            item.connection_orbit_id for item in self.connection_orbits
        )
        component_ids = tuple(item.component_id for item in self.components)
        _validate_sorted_ids(node_ids, "reticular graph node")
        _validate_sorted_ids(ligand_ids, "reticular graph ligand")
        _validate_sorted_ids(connection_ids, "reticular graph connection")
        _validate_sorted_ids(component_ids, "reticular graph component")
        known_nodes = set(node_ids)
        known_ligands = set(ligand_ids)
        known_connections = set(connection_ids)
        for connection in self.connection_orbits:
            if connection.node_orbit_id not in known_nodes:
                raise ValueError("reticular connection references an unknown node")
            if connection.ligand_orbit_id not in known_ligands:
                raise ValueError("reticular connection references an unknown ligand")
        connection_by_ligand = {
            ligand_id: tuple(sorted(
                item.connection_orbit_id
                for item in self.connection_orbits
                if item.ligand_orbit_id == ligand_id
            ))
            for ligand_id in ligand_ids
        }
        if any(
            ligand.connection_orbit_ids
            != connection_by_ligand[ligand.ligand_orbit_id]
            for ligand in self.ligand_orbits
        ):
            raise ValueError("ligand connection references disagree with graph edges")
        reported_nodes = [
            item for component in self.components for item in component.node_orbit_ids
        ]
        reported_ligands = [
            item for component in self.components for item in component.ligand_orbit_ids
        ]
        reported_connections = [
            item
            for component in self.components
            for item in component.connection_orbit_ids
        ]
        if sorted(reported_nodes) != list(node_ids):
            raise ValueError("reticular components must partition node orbits")
        if sorted(reported_ligands) != list(ligand_ids):
            raise ValueError("reticular components must partition ligand orbits")
        if sorted(reported_connections) != list(connection_ids):
            unknown = set(reported_connections) - known_connections
            if unknown:
                raise ValueError("reticular component references an unknown connection")
            raise ValueError("reticular components must partition connection orbits")
        component_by_connection = {
            connection_id: component
            for component in self.components
            for connection_id in component.connection_orbit_ids
        }
        for connection in self.connection_orbits:
            component = component_by_connection[connection.connection_orbit_id]
            if (
                connection.node_orbit_id not in component.node_orbit_ids
                or connection.ligand_orbit_id not in component.ligand_orbit_ids
            ):
                raise ValueError(
                    "reticular connection endpoints lie outside their component"
                )
        if (
            any(item.status is ResolutionStatus.INCOMPLETE for item in self.node_orbits)
            or any(
                item.status is ResolutionStatus.INCOMPLETE
                for item in self.ligand_orbits
            )
            or any(
                item.status is ResolutionStatus.INCOMPLETE
                for item in self.connection_orbits
            )
            or any(
                item.status is ResolutionStatus.INCOMPLETE for item in self.components
            )
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete reticular result must propagate")
        _validate_key_values(self.configuration, "reticular graph configuration")
        _validate_key_values(self.provenance, "reticular graph provenance")

    @property
    def fingerprint(self) -> str:
        """Return deterministic lineage identity for dependent calculations."""

        return _digest(
            {
                "contact_analysis_fingerprint": self._contact_analysis.fingerprint,
                "molecular_contact_analysis_fingerprint": (
                    self._molecular_analysis.contact_analysis_fingerprint
                ),
                "node_orbit_ids": tuple(
                    item.node_orbit_id for item in self.node_orbits
                ),
                "ligands": tuple(
                    (
                        item.ligand_orbit_id,
                        item.attachment_count_per_ligand_instance,
                        item.role.value,
                    )
                    for item in self.ligand_orbits
                ),
                "connections": tuple(
                    (
                        item.connection_orbit_id,
                        tuple(
                            (relation.operation_key, relation.lattice_translation)
                            for relation in item.equivalent_node_ligand_relations
                        ),
                    )
                    for item in self.connection_orbits
                ),
                "component_ids": tuple(
                    item.component_id for item in self.components
                ),
                "configuration": self.configuration,
            }
        )


__all__ = [
    "CoordinationNodeOrbit",
    "LigandComponentOrbit",
    "LigandRole",
    "ReticularComponent",
    "ReticularComponentKind",
    "ReticularConnectionOrbit",
    "ReticularPrecursorGraph",
]
