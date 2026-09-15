"""Hybrid contraction of reticular precursor graphs into exact periodic nets."""

from __future__ import annotations

from dataclasses import dataclass

from cristma.crystallography.pair_canonical import periodic_endpoint_instance
from cristma.crystallography.symmetry_context import _digest

from .intermolecular import molecular_component_stabilizer
from .models import ResolutionStatus
from .periodic_connectivity import _relation_graph_components
from .reticular_models import LigandRole, ReticularPrecursorGraph
from .reticular_net_models import (
    ReticularAttachmentRef,
    ReticularNetComponent,
    ReticularNetComponentKind,
    ReticularNetEdgeOrbit,
    ReticularNetResult,
    ReticularNetVertexKind,
    ReticularNetVertexOrbit,
)
from .sbu_models import SecondaryBuildingUnitResult


_COMPONENT_KIND_BY_RANK = (
    ReticularNetComponentKind.DISCRETE,
    ReticularNetComponentKind.CHAIN,
    ReticularNetComponentKind.LAYER,
    ReticularNetComponentKind.FRAMEWORK,
)


def _status(*statuses: ResolutionStatus) -> ResolutionStatus:
    if ResolutionStatus.INCOMPLETE in statuses:
        return ResolutionStatus.INCOMPLETE
    if ResolutionStatus.AMBIGUOUS in statuses:
        return ResolutionStatus.AMBIGUOUS
    if statuses and all(item is ResolutionStatus.NOT_APPLICABLE for item in statuses):
        return ResolutionStatus.NOT_APPLICABLE
    return ResolutionStatus.RESOLVED


def _sbu_vertex_id(sbu_orbit_id: str) -> str:
    return "reticular-net-vertex:sbu:" + _digest(sbu_orbit_id)


def _ligand_vertex_id(ligand_orbit_id: str) -> str:
    return "reticular-net-vertex:ligand:" + _digest(ligand_orbit_id)


def _attachment_instances(ligand, connections, precursor):
    """Return one exact attachment reference per concrete SBU instance."""

    context = precursor._contact_analysis._symmetry_context
    mapping = precursor._contact_analysis._asymmetric_unit_mapping
    node_by_id = {item.node_orbit_id: item for item in precursor.node_orbits}
    by_endpoint = {}
    for connection in connections:
        node = node_by_id[connection.node_orbit_id]
        for relation in connection.equivalent_node_ligand_relations:
            endpoint = periodic_endpoint_instance(
                node.center_independent_site_id,
                relation.inverse(context),
                mapping,
            )
            attachment = ReticularAttachmentRef(
                connection.connection_orbit_id, relation
            )
            candidate = (connection.node_orbit_id, endpoint, attachment)
            previous = by_endpoint.get(endpoint)
            if previous is None or candidate < previous:
                by_endpoint[endpoint] = candidate
    instances = tuple(sorted(by_endpoint.values()))
    if (
        ligand.status is ResolutionStatus.RESOLVED
        and len(instances) != ligand.attachment_count_per_ligand_instance
    ):
        raise ValueError(
            "resolved ligand attachment count disagrees with exact SBU instances"
        )
    return instances


def _orient_edge(first_vertex_id, second_vertex_id, relation, context):
    direct = (first_vertex_id, second_vertex_id, relation)
    reverse = (second_vertex_id, first_vertex_id, relation.inverse(context))
    return min(direct, reverse)


@dataclass(frozen=True, slots=True)
class ReticularNetAnalyzer:
    """Build a hybrid-contracted periodic net without materialization."""

    def get_config(self) -> dict[str, object]:
        return {}

    def clone(self, **changes: object) -> "ReticularNetAnalyzer":
        if changes:
            raise TypeError(
                "unknown ReticularNetAnalyzer configuration: "
                + ", ".join(sorted(changes))
            )
        return self

    def analyze(
        self,
        precursor_graph: ReticularPrecursorGraph,
        sbu_result: SecondaryBuildingUnitResult,
    ) -> ReticularNetResult:
        if not isinstance(precursor_graph, ReticularPrecursorGraph):
            raise TypeError("precursor_graph must be ReticularPrecursorGraph")
        if not isinstance(sbu_result, SecondaryBuildingUnitResult):
            raise TypeError("sbu_result must be SecondaryBuildingUnitResult")
        if sbu_result.precursor_graph_fingerprint != precursor_graph.fingerprint:
            raise ValueError("reticular net source lineage disagrees")

        configuration: tuple[tuple[str, object], ...] = ()
        provenance = (("method", "cristma.hybrid_reticular_net:1"),)
        context = precursor_graph._contact_analysis._symmetry_context
        mapping = precursor_graph._contact_analysis._asymmetric_unit_mapping
        molecular_result = precursor_graph._molecular_analysis

        node_by_id = {item.node_orbit_id: item for item in precursor_graph.node_orbits}
        ligand_by_id = {
            item.ligand_orbit_id: item for item in precursor_graph.ligand_orbits
        }
        connection_by_id = {
            item.connection_orbit_id: item
            for item in precursor_graph.connection_orbits
        }
        component_by_id = {
            item.component_orbit_id: item for item in molecular_result.molecule_orbits
        }
        sbu_by_node_id = {
            item.node_orbit_ids[0]: item for item in sbu_result.sbu_orbits
        }
        if set(sbu_by_node_id) != set(node_by_id):
            raise ValueError("SBU result does not cover every precursor node orbit")

        vertices = [
            ReticularNetVertexOrbit(
                _sbu_vertex_id(sbu.sbu_orbit_id),
                ReticularNetVertexKind.SBU,
                sbu.sbu_orbit_id,
                None,
                sbu.multiplicity_in_reference_cell,
                sbu.status,
                sbu.diagnostics,
                (("method", "cristma.reticular_net_sbu_vertices:1"),),
            )
            for sbu in sbu_result.sbu_orbits
        ]
        sbu_vertex_by_node_id = {
            node_id: _sbu_vertex_id(sbu.sbu_orbit_id)
            for node_id, sbu in sbu_by_node_id.items()
        }

        edges: list[ReticularNetEdgeOrbit] = []
        retained_ligand_ids: set[str] = set()
        for ligand in precursor_graph.ligand_orbits:
            connections = tuple(
                connection_by_id[item] for item in ligand.connection_orbit_ids
            )
            attachments = _attachment_instances(
                ligand, connections, precursor_graph
            )
            if (
                ligand.status is ResolutionStatus.RESOLVED
                and ligand.role is LigandRole.TERMINAL
            ):
                continue
            if (
                ligand.status is ResolutionStatus.RESOLVED
                and ligand.attachment_count_per_ligand_instance == 2
            ):
                first, second = attachments
                first_vertex = sbu_vertex_by_node_id[first[0]]
                second_vertex = sbu_vertex_by_node_id[second[0]]
                relation = first[2].periodic_relation.compose(
                    second[2].periodic_relation.inverse(context), context
                )
                first_vertex, second_vertex, relation = _orient_edge(
                    first_vertex, second_vertex, relation, context
                )
                source_attachments = tuple(sorted((first[2], second[2])))
                edge_id = "reticular-net-edge:contracted:" + _digest(
                    {
                        "linker_orbit_id": ligand.ligand_orbit_id,
                        "first_vertex_orbit_id": first_vertex,
                        "second_vertex_orbit_id": second_vertex,
                        "periodic_relation": (
                            relation.operation_key,
                            relation.lattice_translation,
                        ),
                        "source_attachments": tuple(
                            (
                                item.connection_orbit_id,
                                item.periodic_relation.operation_key,
                                item.periodic_relation.lattice_translation,
                            )
                            for item in source_attachments
                        ),
                    }
                )
                edges.append(
                    ReticularNetEdgeOrbit(
                        edge_id,
                        first_vertex,
                        second_vertex,
                        relation,
                        ligand.ligand_orbit_id,
                        source_attachments,
                        ligand.multiplicity_in_reference_cell,
                        _status(
                            ligand.status,
                            *(connection.status for connection in connections),
                        ),
                        tuple(dict.fromkeys(
                            diagnostic
                            for connection in connections
                            for diagnostic in connection.diagnostics
                        )),
                        (("method", "cristma.degree_two_linker_contraction:1"),),
                    )
                )
                continue

            retained_ligand_ids.add(ligand.ligand_orbit_id)
            ligand_vertex_id = _ligand_vertex_id(ligand.ligand_orbit_id)
            vertices.append(
                ReticularNetVertexOrbit(
                    ligand_vertex_id,
                    ReticularNetVertexKind.ORGANIC_LINKER,
                    None,
                    ligand.ligand_orbit_id,
                    ligand.multiplicity_in_reference_cell,
                    ligand.status,
                    ligand.diagnostics,
                    (("method", "cristma.retained_reticular_linker:1"),),
                )
            )
            for node_id, _, attachment in attachments:
                sbu_vertex_id = sbu_vertex_by_node_id[node_id]
                connection = connection_by_id[attachment.connection_orbit_id]
                edge_id = "reticular-net-edge:attachment:" + _digest(
                    {
                        "linker_orbit_id": ligand.ligand_orbit_id,
                        "connection_orbit_id": attachment.connection_orbit_id,
                        "periodic_relation": (
                            attachment.periodic_relation.operation_key,
                            attachment.periodic_relation.lattice_translation,
                        ),
                    }
                )
                edges.append(
                    ReticularNetEdgeOrbit(
                        edge_id,
                        sbu_vertex_id,
                        ligand_vertex_id,
                        attachment.periodic_relation,
                        ligand.ligand_orbit_id,
                        (attachment,),
                        ligand.multiplicity_in_reference_cell,
                        _status(ligand.status, connection.status),
                        tuple(dict.fromkeys((*ligand.diagnostics, *connection.diagnostics))),
                        (("method", "cristma.retained_linker_attachment:1"),),
                    )
                )

        ordered_vertices = tuple(
            sorted(vertices, key=lambda item: item.net_vertex_orbit_id)
        )
        ordered_edges = tuple(sorted(edges, key=lambda item: item.net_edge_orbit_id))
        vertex_ids = tuple(item.net_vertex_orbit_id for item in ordered_vertices)
        graph_edges = [
            (
                item.net_edge_orbit_id,
                item.first_vertex_orbit_id,
                item.second_vertex_orbit_id,
                item.periodic_relation,
            )
            for item in ordered_edges
        ]

        # Endpoint stabilizers are exact quotient-graph loops and contribute
        # to periodic closure without becoming public chemical net edges.
        for node_id, sbu in sorted(sbu_by_node_id.items()):
            vertex_id = _sbu_vertex_id(sbu.sbu_orbit_id)
            node = node_by_id[node_id]
            for index, relation in enumerate(
                mapping.by_site_id[
                    node.center_independent_site_id
                ].stabilizer_relations
            ):
                graph_edges.append(
                    (f"_sbu-stabilizer:{vertex_id}:{index}", vertex_id, vertex_id, relation)
                )
        for ligand_id in sorted(retained_ligand_ids):
            ligand = ligand_by_id[ligand_id]
            component = component_by_id[ligand.source_component_orbit_id]
            vertex_id = _ligand_vertex_id(ligand_id)
            for index, relation in enumerate(
                molecular_component_stabilizer(component, context)
            ):
                graph_edges.append(
                    (
                        f"_ligand-stabilizer:{vertex_id}:{index}",
                        vertex_id,
                        vertex_id,
                        relation,
                    )
                )

        graph_components = _relation_graph_components(
            vertex_ids, tuple(graph_edges), context
        )
        public_edge_ids = {item.net_edge_orbit_id for item in ordered_edges}
        vertex_by_id = {
            item.net_vertex_orbit_id: item for item in ordered_vertices
        }
        edge_by_id = {item.net_edge_orbit_id: item for item in ordered_edges}
        components = []
        for component in graph_components:
            edge_ids = tuple(
                item for item in component.edge_ids if item in public_edge_ids
            )
            component_status = _status(
                *(vertex_by_id[item].status for item in component.node_ids),
                *(edge_by_id[item].status for item in edge_ids),
            )
            component_id = "reticular-net-component:" + _digest(
                {
                    "vertex_orbit_ids": component.node_ids,
                    "edge_orbit_ids": edge_ids,
                    "closure_relations": tuple(
                        (item.operation_key, item.lattice_translation)
                        for item in component.closure_relations
                    ),
                }
            )
            components.append(
                ReticularNetComponent(
                    component_id,
                    component.node_ids,
                    edge_ids,
                    component.image_relations,
                    component.closure_relations,
                    component.rank,
                    component.periodic_generators,
                    _COMPONENT_KIND_BY_RANK[component.rank],
                    component_status,
                )
            )
        ordered_components = tuple(
            sorted(components, key=lambda item: item.net_component_id)
        )
        result_status = _status(
            precursor_graph.status,
            sbu_result.status,
            *(item.status for item in ordered_vertices),
            *(item.status for item in ordered_edges),
            *(item.status for item in ordered_components),
        )
        return ReticularNetResult(
            precursor_graph,
            sbu_result,
            sbu_result.sbu_orbits,
            ordered_vertices,
            ordered_edges,
            ordered_components,
            precursor_graph.fingerprint,
            sbu_result.fingerprint,
            result_status,
            tuple(dict.fromkeys((*precursor_graph.diagnostics, *sbu_result.diagnostics))),
            configuration,
            provenance,
        )


__all__ = ["ReticularNetAnalyzer"]
