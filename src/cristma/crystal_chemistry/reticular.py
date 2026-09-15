"""Exact orbit-first precursor graph for metal-organic connectivity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cristma.chemistry import DecompositionMode
from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.pair_canonical import periodic_endpoint_instance
from cristma.crystallography.symmetry_context import _digest

from .contact_analysis import ContactAnalysisResult
from .intermolecular import molecular_component_stabilizer
from .models import ResolutionStatus
from .molecular import (
    MolecularAnalysisResult,
    MolecularComponentKind,
    MolecularComponentOrbit,
)
from .periodic_connectivity import _relation_graph_components
from .reticular_models import (
    CoordinationNodeOrbit,
    LigandComponentOrbit,
    LigandRole,
    ReticularComponent,
    ReticularComponentKind,
    ReticularConnectionOrbit,
    ReticularPrecursorGraph,
)


def _certain_primary_incidence_ids(shell) -> tuple[str, ...]:
    """Return only incidences present in every admissible shell interpretation."""

    if shell.selected is not None:
        return shell.selected.primary_incidence_ids
    if not shell.alternatives:
        return ()
    certain = set(shell.alternatives[0].primary_incidence_ids)
    for alternative in shell.alternatives[1:]:
        certain &= set(alternative.primary_incidence_ids)
    return tuple(sorted(certain))


def _status(*statuses: ResolutionStatus) -> ResolutionStatus:
    if ResolutionStatus.INCOMPLETE in statuses:
        return ResolutionStatus.INCOMPLETE
    if ResolutionStatus.AMBIGUOUS in statuses:
        return ResolutionStatus.AMBIGUOUS
    if statuses and all(item is ResolutionStatus.NOT_APPLICABLE for item in statuses):
        return ResolutionStatus.NOT_APPLICABLE
    return ResolutionStatus.RESOLVED


def _canonical_directed_relation(
    relation: PeriodicSymmetryRelation,
    center_stabilizers: tuple[PeriodicSymmetryRelation, ...],
    ligand_stabilizers: tuple[PeriodicSymmetryRelation, ...],
    context,
) -> PeriodicSymmetryRelation:
    return min(
        left.compose(relation, context).compose(right, context)
        for left in center_stabilizers
        for right in ligand_stabilizers
    )


def _connection_multiplicity(
    center_site_id: str,
    component,
    attachment_relations: tuple[PeriodicSymmetryRelation, ...],
    contact_result: ContactAnalysisResult,
) -> int:
    context = contact_result._symmetry_context
    mapping = contact_result._asymmetric_unit_mapping
    attached_sbu_instances = {
        periodic_endpoint_instance(
            center_site_id,
            relation.inverse(context),
            mapping,
        )
        for relation in attachment_relations
    }
    return (
        len(attached_sbu_instances) * component.multiplicity_in_reference_cell
    )


@dataclass(frozen=True, slots=True)
class _ConnectionSeed:
    node_id: str
    ligand_id: str
    center_site_id: str
    relation: PeriodicSymmetryRelation
    attachment_relations: tuple[PeriodicSymmetryRelation, ...]
    donor_site_id: str
    resolved_contact_orbit_id: str
    interpretation_id: str
    status: ResolutionStatus


@dataclass(frozen=True, slots=True)
class ReticularAnalyzer:
    """Build a symmetry-quotient node-ligand graph without materialization."""

    def get_config(self) -> dict[str, object]:
        return {}

    def clone(self, **changes: object) -> "ReticularAnalyzer":
        if changes:
            raise TypeError(
                "unknown ReticularAnalyzer configuration: "
                + ", ".join(sorted(changes))
            )
        return self

    def analyze(
        self,
        contact_result: ContactAnalysisResult,
        molecular_result: MolecularAnalysisResult,
    ) -> ReticularPrecursorGraph:
        if not isinstance(contact_result, ContactAnalysisResult):
            raise TypeError("contact_result must be ContactAnalysisResult")
        if not isinstance(molecular_result, MolecularAnalysisResult):
            raise TypeError("molecular_result must be MolecularAnalysisResult")
        if molecular_result.contact_analysis_fingerprint != contact_result.fingerprint:
            raise ValueError("reticular source lineage disagrees")

        configuration: tuple[tuple[str, object], ...] = ()
        provenance = (("method", "cristma.reticular_precursor_graph:1"),)
        if contact_result.grammar.mode is not DecompositionMode.METAL_ORGANIC:
            return ReticularPrecursorGraph(
                contact_result,
                molecular_result,
                (),
                (),
                (),
                (),
                ResolutionStatus.NOT_APPLICABLE,
                (),
                configuration,
                provenance,
            )

        incidence_by_id = {
            item.incidence_orbit_id: item
            for item in contact_result.contact_incidence_orbits
        }
        contact_by_id = {
            item.resolved_contact_orbit_id: item
            for item in contact_result.contact_orbits
        }
        component_by_site = {
            site_id: component
            for component in molecular_result.molecule_orbits
            for site_id in component.independent_site_ids
        }
        shell_by_incidence: dict[str, object] = {}
        selected_incidence_ids: set[str] = set()
        for shell in contact_result.coordination_shell_orbits:
            for incidence_id in _certain_primary_incidence_ids(shell):
                selected_incidence_ids.add(incidence_id)
                shell_by_incidence[incidence_id] = shell

        eligible = tuple(
            incidence_by_id[item]
            for item in sorted(selected_incidence_ids)
            if incidence_by_id[item].ligand_independent_site_id in component_by_site
        )
        if not eligible:
            result_status = _status(contact_result.status, molecular_result.status)
            if result_status is ResolutionStatus.RESOLVED:
                result_status = ResolutionStatus.NOT_APPLICABLE
            return ReticularPrecursorGraph(
                contact_result,
                molecular_result,
                (),
                (),
                (),
                (),
                result_status,
                tuple(dict.fromkeys((*contact_result.diagnostics, *molecular_result.diagnostics))),
                configuration,
                provenance,
            )

        context = contact_result._symmetry_context
        mapping = contact_result._asymmetric_unit_mapping
        node_shells: dict[str, set[str]] = defaultdict(set)
        node_contacts: dict[str, set[str]] = defaultdict(set)
        node_statuses: dict[str, list[ResolutionStatus]] = defaultdict(list)
        seeds: list[_ConnectionSeed] = []
        for incidence in eligible:
            shell = shell_by_incidence[incidence.incidence_orbit_id]
            center_site_id = incidence.center_independent_site_id
            node_id = "coordination-node-orbit:" + _digest(
                {"center_independent_site_id": center_site_id}
            )
            component = component_by_site[incidence.ligand_independent_site_id]
            ligand_id = "ligand-component-orbit:" + _digest(
                {"source_component_orbit_id": component.component_orbit_id}
            )
            component_images = dict(component.image_relations)
            attachment_relations = tuple(sorted(set(
                oriented_relation.compose(
                    component_images[
                        incidence.ligand_independent_site_id
                    ].inverse(context),
                    context,
                )
                for oriented_relation in incidence.equivalent_oriented_relations
            )))
            canonical_relations = {
                _canonical_directed_relation(
                    attachment_relation,
                    mapping.by_site_id[center_site_id].stabilizer_relations,
                    molecular_component_stabilizer(component, context),
                    context,
                )
                for attachment_relation in attachment_relations
            }
            if len(canonical_relations) != 1:
                raise ValueError(
                    "one incidence orbit produced multiple node-ligand orbit identities"
                )
            relation = next(iter(canonical_relations))
            contact = contact_by_id[incidence.resolved_contact_orbit_id]
            seed_status = _status(shell.status, incidence.status, contact.status)
            seeds.append(
                _ConnectionSeed(
                    node_id,
                    ligand_id,
                    center_site_id,
                    relation,
                    attachment_relations,
                    incidence.ligand_independent_site_id,
                    incidence.resolved_contact_orbit_id,
                    incidence.interpretation_id,
                    seed_status,
                )
            )
            node_shells[node_id].add(shell.shell_orbit_id)
            node_contacts[node_id].add(incidence.resolved_contact_orbit_id)
            node_statuses[node_id].append(seed_status)

        grouped_seeds: dict[
            tuple[str, str, PeriodicSymmetryRelation], list[_ConnectionSeed]
        ] = defaultdict(list)
        for seed in seeds:
            grouped_seeds[(seed.node_id, seed.ligand_id, seed.relation)].append(seed)

        connection_orbits: list[ReticularConnectionOrbit] = []
        for (node_id, ligand_id, relation), group in sorted(grouped_seeds.items()):
            first = group[0]
            component = component_by_site[first.donor_site_id]
            attachment_relations = tuple(sorted(set(
                attachment_relation
                for item in group
                for attachment_relation in item.attachment_relations
            )))
            connection_id = "reticular-connection-orbit:" + _digest(
                {
                    "node_orbit_id": node_id,
                    "ligand_orbit_id": ligand_id,
                    "canonical_relation": (
                        relation.operation_key,
                        relation.lattice_translation,
                    ),
                }
            )
            connection_orbits.append(
                ReticularConnectionOrbit(
                    connection_id,
                    node_id,
                    ligand_id,
                    relation,
                    attachment_relations,
                    tuple(sorted({item.donor_site_id for item in group})),
                    tuple(sorted({item.resolved_contact_orbit_id for item in group})),
                    tuple(sorted({item.interpretation_id for item in group})),
                    _connection_multiplicity(
                        first.center_site_id,
                        component,
                        attachment_relations,
                        contact_result,
                    ),
                    _status(*(item.status for item in group)),
                    provenance=(("method", "cristma.reticular_connection_orbits:1"),),
                )
            )
        connections = tuple(
            sorted(connection_orbits, key=lambda item: item.connection_orbit_id)
        )

        nodes = tuple(
            CoordinationNodeOrbit(
                node_id,
                next(seed.center_site_id for seed in seeds if seed.node_id == node_id),
                tuple(sorted(node_shells[node_id])),
                tuple(sorted(node_contacts[node_id])),
                len(mapping.by_site_id[
                    next(seed.center_site_id for seed in seeds if seed.node_id == node_id)
                ].reference_cell_images),
                _status(*node_statuses[node_id]),
                provenance=(("method", "cristma.coordination_node_orbits:1"),),
            )
            for node_id in sorted(node_shells)
        )

        components_by_ligand_id = {
            "ligand-component-orbit:" + _digest(
                {"source_component_orbit_id": component.component_orbit_id}
            ): component
            for component in molecular_result.molecule_orbits
        }
        ligands: list[LigandComponentOrbit] = []
        for ligand_id in sorted({item.ligand_orbit_id for item in connections}):
            component = components_by_ligand_id[ligand_id]
            attached = tuple(
                item for item in connections if item.ligand_orbit_id == ligand_id
            )
            total_multiplicity = sum(
                item.multiplicity_in_reference_cell for item in attached
            )
            if total_multiplicity % component.multiplicity_in_reference_cell:
                raise ValueError(
                    "reticular attachment multiplicity is incompatible with ligand multiplicity"
                )
            attachments_per_instance = (
                total_multiplicity // component.multiplicity_in_reference_cell
            )
            role = (
                LigandRole.TERMINAL
                if attachments_per_instance == 1
                else LigandRole.LINKER
            )
            ligands.append(
                LigandComponentOrbit(
                    ligand_id,
                    component.component_orbit_id,
                    component.independent_site_ids,
                    component.bond_orbit_ids,
                    tuple(item.connection_orbit_id for item in attached),
                    role,
                    attachments_per_instance,
                    component.multiplicity_in_reference_cell,
                    _status(component.status, *(item.status for item in attached)),
                    provenance=(
                        ("attachments_per_ligand_instance", attachments_per_instance),
                        ("method", "cristma.ligand_component_orbits:1"),
                    ),
                )
            )
        ligand_orbits = tuple(ligands)

        graph_node_ids = tuple(sorted(
            (*(f"node:{item.node_orbit_id}" for item in nodes),
             *(f"ligand:{item.ligand_orbit_id}" for item in ligand_orbits))
        ))
        graph_edges = [
            (
                item.connection_orbit_id,
                f"node:{item.node_orbit_id}",
                f"ligand:{item.ligand_orbit_id}",
                item.canonical_node_ligand_relation,
            )
            for item in connections
        ]
        # Vertex stabilizers are exact loops of the symmetry-quotient graph.
        # Omitting them turns a single edge orbit of a periodic framework into
        # an apparently finite two-vertex graph.
        for node in nodes:
            for index, relation in enumerate(
                mapping.by_site_id[node.center_independent_site_id].stabilizer_relations
            ):
                node_key = f"node:{node.node_orbit_id}"
                graph_edges.append((f"_node-stabilizer:{node.node_orbit_id}:{index}", node_key, node_key, relation))
        for ligand in ligand_orbits:
            component = components_by_ligand_id[ligand.ligand_orbit_id]
            for index, relation in enumerate(molecular_component_stabilizer(component, context)):
                ligand_key = f"ligand:{ligand.ligand_orbit_id}"
                graph_edges.append((f"_ligand-stabilizer:{ligand.ligand_orbit_id}:{index}", ligand_key, ligand_key, relation))

        graph_components = _relation_graph_components(
            graph_node_ids,
            tuple(graph_edges),
            context,
        )
        connection_ids = {item.connection_orbit_id for item in connections}
        connection_by_id = {
            item.connection_orbit_id: item for item in connections
        }
        node_by_id = {item.node_orbit_id: item for item in nodes}
        ligand_by_id = {item.ligand_orbit_id: item for item in ligand_orbits}
        components: list[ReticularComponent] = []
        for component in graph_components:
            real_edge_ids = tuple(
                item for item in component.edge_ids if item in connection_ids
            )
            node_ids = tuple(
                sorted(item.removeprefix("node:") for item in component.node_ids if item.startswith("node:"))
            )
            ligand_ids = tuple(
                sorted(item.removeprefix("ligand:") for item in component.node_ids if item.startswith("ligand:"))
            )
            statuses = tuple(node_by_id[item].status for item in node_ids) + tuple(
                ligand_by_id[item].status for item in ligand_ids
            ) + tuple(connection_by_id[item].status for item in real_edge_ids)
            component_id = "reticular-component:" + _digest(
                (
                    ("node_orbit_ids", node_ids),
                    ("ligand_orbit_ids", ligand_ids),
                    ("connection_orbit_ids", real_edge_ids),
                )
            )
            components.append(
                ReticularComponent(
                    component_id,
                    node_ids,
                    ligand_ids,
                    real_edge_ids,
                    component.image_relations,
                    component.closure_relations,
                    component.rank,
                    component.periodic_generators,
                    ReticularComponentKind(
                        (
                            "discrete_complex",
                            "periodic_chain",
                            "periodic_layer",
                            "periodic_framework",
                        )[component.rank]
                    ),
                    _status(*statuses),
                )
            )
        ordered_components = tuple(
            sorted(components, key=lambda item: item.component_id)
        )
        result_status = _status(
            contact_result.status,
            molecular_result.status,
            *(item.status for item in ordered_components),
        )
        return ReticularPrecursorGraph(
            contact_result,
            molecular_result,
            nodes,
            ligand_orbits,
            connections,
            ordered_components,
            result_status,
            tuple(dict.fromkeys((*contact_result.diagnostics, *molecular_result.diagnostics))),
            configuration,
            provenance,
        )


__all__ = ["ReticularAnalyzer"]
