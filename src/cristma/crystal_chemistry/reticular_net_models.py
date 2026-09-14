"""Immutable results for hybrid-contracted periodic reticular nets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic

from .models import ResolutionStatus
from .sbu_models import SecondaryBuildingUnitOrbit

if TYPE_CHECKING:
    from .reticular_models import ReticularPrecursorGraph
    from .sbu_models import SecondaryBuildingUnitResult


Translation = tuple[int, int, int]


class ReticularNetVertexKind(StrEnum):
    SBU = "sbu"
    ORGANIC_LINKER = "organic_linker"


class ReticularNetComponentKind(StrEnum):
    DISCRETE = "discrete"
    CHAIN = "chain"
    LAYER = "layer"
    FRAMEWORK = "framework"


_KIND_BY_RANK = {
    0: ReticularNetComponentKind.DISCRETE,
    1: ReticularNetComponentKind.CHAIN,
    2: ReticularNetComponentKind.LAYER,
    3: ReticularNetComponentKind.FRAMEWORK,
}


def _id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must not be empty")


def _ids(values: tuple[str, ...], name: str, *, required: bool = False) -> None:
    if required and not values:
        raise ValueError(f"{name} IDs must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} IDs must not be empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} IDs must be unique and sorted")


def _multiplicity(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} multiplicity must be an integer")
    if value <= 0:
        raise ValueError(f"{name} multiplicity must be positive")


def _key_values(values: tuple[tuple[str, object], ...], name: str) -> None:
    keys = tuple(key for key, _ in values)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{name} keys must be unique and sorted")


@dataclass(frozen=True, slots=True, order=True)
class ReticularAttachmentRef:
    connection_orbit_id: str
    periodic_relation: PeriodicSymmetryRelation

    def __post_init__(self) -> None:
        _id(self.connection_orbit_id, "reticular attachment connection ID")
        if not isinstance(self.periodic_relation, PeriodicSymmetryRelation):
            raise TypeError("reticular attachment requires an exact periodic relation")


@dataclass(frozen=True, slots=True)
class ReticularNetVertexOrbit:
    net_vertex_orbit_id: str
    kind: ReticularNetVertexKind
    source_sbu_orbit_id: str | None
    source_ligand_orbit_id: str | None
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _id(self.net_vertex_orbit_id, "reticular net vertex ID")
        if not isinstance(self.kind, ReticularNetVertexKind):
            raise TypeError("reticular net vertex kind must be ReticularNetVertexKind")
        if self.kind is ReticularNetVertexKind.SBU:
            if not self.source_sbu_orbit_id or self.source_ligand_orbit_id is not None:
                raise ValueError("SBU net vertex requires only an SBU source ID")
        elif not self.source_ligand_orbit_id or self.source_sbu_orbit_id is not None:
            raise ValueError("organic net vertex requires only a ligand source ID")
        _multiplicity(self.multiplicity_in_reference_cell, "reticular net vertex")
        if not isinstance(self.status, ResolutionStatus):
            raise TypeError("reticular net vertex status must be ResolutionStatus")
        _key_values(self.provenance, "reticular net vertex provenance")


@dataclass(frozen=True, slots=True)
class ReticularNetEdgeOrbit:
    net_edge_orbit_id: str
    first_vertex_orbit_id: str
    second_vertex_orbit_id: str
    periodic_relation: PeriodicSymmetryRelation
    linker_orbit_id: str
    source_attachments: tuple[ReticularAttachmentRef, ...]
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _id(self.net_edge_orbit_id, "reticular net edge ID")
        _id(self.first_vertex_orbit_id, "first reticular net vertex ID")
        _id(self.second_vertex_orbit_id, "second reticular net vertex ID")
        _id(self.linker_orbit_id, "reticular net edge linker ID")
        if not isinstance(self.periodic_relation, PeriodicSymmetryRelation):
            raise TypeError("reticular net edge requires an exact periodic relation")
        if (
            not self.source_attachments
            or self.source_attachments != tuple(sorted(set(self.source_attachments)))
        ):
            raise ValueError(
                "reticular net source attachments must be non-empty, unique, and sorted"
            )
        _multiplicity(self.multiplicity_in_reference_cell, "reticular net edge")
        if not isinstance(self.status, ResolutionStatus):
            raise TypeError("reticular net edge status must be ResolutionStatus")
        _key_values(self.provenance, "reticular net edge provenance")


@dataclass(frozen=True, slots=True)
class ReticularNetComponent:
    net_component_id: str
    vertex_orbit_ids: tuple[str, ...]
    edge_orbit_ids: tuple[str, ...]
    image_relations: tuple[tuple[str, PeriodicSymmetryRelation], ...]
    closure_relations: tuple[PeriodicSymmetryRelation, ...]
    periodic_rank: int
    periodic_generators: tuple[Translation, ...]
    kind: ReticularNetComponentKind
    status: ResolutionStatus

    def __post_init__(self) -> None:
        _id(self.net_component_id, "reticular net component ID")
        _ids(self.vertex_orbit_ids, "reticular net component vertex", required=True)
        _ids(self.edge_orbit_ids, "reticular net component edge")
        if tuple(item_id for item_id, _ in self.image_relations) != self.vertex_orbit_ids:
            raise ValueError(
                "reticular net component requires one sorted image relation per vertex"
            )
        if self.closure_relations != tuple(sorted(set(self.closure_relations))):
            raise ValueError(
                "reticular net component closure relations must be unique and sorted"
            )
        if isinstance(self.periodic_rank, bool) or not isinstance(self.periodic_rank, int):
            raise TypeError("reticular net periodic rank must be an integer")
        if self.periodic_rank not in range(4):
            raise ValueError("reticular net periodic rank must lie within [0, 3]")
        if self.periodic_rank != len(self.periodic_generators):
            raise ValueError("reticular net periodic rank and generator count disagree")
        if self.kind is not _KIND_BY_RANK[self.periodic_rank]:
            raise ValueError("reticular net periodic rank and component kind disagree")
        for generator in self.periodic_generators:
            if len(generator) != 3 or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in generator
            ):
                raise TypeError("reticular net generators must be integer triples")
        if not isinstance(self.status, ResolutionStatus):
            raise TypeError("reticular net component status must be ResolutionStatus")


@dataclass(frozen=True, slots=True)
class ReticularNetResult:
    _precursor_graph: "ReticularPrecursorGraph" = field(repr=False, compare=False)
    _sbu_result: "SecondaryBuildingUnitResult" = field(repr=False, compare=False)
    sbu_orbits: tuple[SecondaryBuildingUnitOrbit, ...]
    vertex_orbits: tuple[ReticularNetVertexOrbit, ...]
    edge_orbits: tuple[ReticularNetEdgeOrbit, ...]
    components: tuple[ReticularNetComponent, ...]
    precursor_graph_fingerprint: str
    sbu_result_fingerprint: str
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if self.precursor_graph_fingerprint != self._precursor_graph.fingerprint:
            raise ValueError("reticular net precursor fingerprint disagrees")
        if self.sbu_result_fingerprint != self._sbu_result.fingerprint:
            raise ValueError("reticular net SBU fingerprint disagrees")
        if self.sbu_orbits != self._sbu_result.sbu_orbits:
            raise ValueError("reticular net SBU orbits disagree with source result")
        vertex_ids = tuple(item.net_vertex_orbit_id for item in self.vertex_orbits)
        edge_ids = tuple(item.net_edge_orbit_id for item in self.edge_orbits)
        component_ids = tuple(item.net_component_id for item in self.components)
        _ids(vertex_ids, "reticular net vertex")
        _ids(edge_ids, "reticular net edge")
        _ids(component_ids, "reticular net component")
        known_vertices = set(vertex_ids)
        known_edges = set(edge_ids)
        for edge in self.edge_orbits:
            if (
                edge.first_vertex_orbit_id not in known_vertices
                or edge.second_vertex_orbit_id not in known_vertices
            ):
                raise ValueError("reticular net edge references an unknown vertex")
        reported_vertices = [
            item for component in self.components for item in component.vertex_orbit_ids
        ]
        reported_edges = [
            item for component in self.components for item in component.edge_orbit_ids
        ]
        if sorted(reported_vertices) != list(vertex_ids):
            raise ValueError("reticular net components must partition vertex orbits")
        if sorted(reported_edges) != list(edge_ids):
            if set(reported_edges) - known_edges:
                raise ValueError("reticular net component references an unknown edge")
            raise ValueError("reticular net components must partition edge orbits")
        component_by_edge = {
            edge_id: component
            for component in self.components
            for edge_id in component.edge_orbit_ids
        }
        for edge in self.edge_orbits:
            component = component_by_edge[edge.net_edge_orbit_id]
            if (
                edge.first_vertex_orbit_id not in component.vertex_orbit_ids
                or edge.second_vertex_orbit_id not in component.vertex_orbit_ids
            ):
                raise ValueError("reticular net edge endpoints lie outside component")
        if (
            any(item.status is ResolutionStatus.INCOMPLETE for item in self.vertex_orbits)
            or any(item.status is ResolutionStatus.INCOMPLETE for item in self.edge_orbits)
            or any(item.status is ResolutionStatus.INCOMPLETE for item in self.components)
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete reticular net object must propagate")
        if not isinstance(self.status, ResolutionStatus):
            raise TypeError("reticular net result status must be ResolutionStatus")
        _key_values(self.configuration, "reticular net configuration")
        _key_values(self.provenance, "reticular net provenance")

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "precursor_graph_fingerprint": self.precursor_graph_fingerprint,
                "sbu_result_fingerprint": self.sbu_result_fingerprint,
                "vertices": tuple(
                    (
                        item.net_vertex_orbit_id,
                        item.kind.value,
                        item.source_sbu_orbit_id,
                        item.source_ligand_orbit_id,
                    )
                    for item in self.vertex_orbits
                ),
                "edges": tuple(
                    (
                        item.net_edge_orbit_id,
                        item.first_vertex_orbit_id,
                        item.second_vertex_orbit_id,
                        item.periodic_relation.operation_key,
                        item.periodic_relation.lattice_translation,
                        item.linker_orbit_id,
                        tuple(
                            (
                                ref.connection_orbit_id,
                                ref.periodic_relation.operation_key,
                                ref.periodic_relation.lattice_translation,
                            )
                            for ref in item.source_attachments
                        ),
                    )
                    for item in self.edge_orbits
                ),
                "components": tuple(
                    (
                        item.net_component_id,
                        item.vertex_orbit_ids,
                        item.edge_orbit_ids,
                        item.periodic_rank,
                        item.periodic_generators,
                    )
                    for item in self.components
                ),
                "configuration": self.configuration,
            }
        )


__all__ = [
    "ReticularAttachmentRef",
    "ReticularNetComponent",
    "ReticularNetComponentKind",
    "ReticularNetEdgeOrbit",
    "ReticularNetResult",
    "ReticularNetVertexKind",
    "ReticularNetVertexOrbit",
]
