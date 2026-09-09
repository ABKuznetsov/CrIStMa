"""Exact source-relation conversion for molecular packing graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from cristma.crystallography import PeriodicSymmetryRelation, identity_relation
from cristma.crystallography.pair_canonical import canonical_instance_owner
from cristma.crystallography.symmetry_context import _digest

from .ch_pi import CHPiAnalysisResult
from .intermolecular import (
    canonical_molecule_contact_relation,
    molecule_relation_for_atomic_pair,
)
from .intermolecular_models import IntermolecularContactAnalysisResult
from .models import ResolutionStatus
from .molecular import MolecularComponentOrbit
from .packing_models import (
    MolecularPackingComponent,
    MolecularPackingGraph,
    PackingConnectionOrbit,
    PackingInteractionKind,
    PackingInteractionRef,
)
from .periodic_connectivity import _relation_graph_components
from .pi_stacking import PiStackingAnalysisResult, _feature


MoleculeContactDescriptor = tuple[str, str, PeriodicSymmetryRelation]


@dataclass(frozen=True, slots=True, order=True)
class _PackingEvidence:
    descriptor: MoleculeContactDescriptor
    reference: PackingInteractionRef
    status: ResolutionStatus


def _validate_source_lineage(
    intermolecular: IntermolecularContactAnalysisResult,
    stacking: PiStackingAnalysisResult,
    ch_pi: CHPiAnalysisResult,
):
    contact_analysis = intermolecular._contact_analysis
    molecular_analysis = intermolecular._molecular_analysis
    pi_system_analysis = stacking._pi_system_analysis
    if (
        pi_system_analysis._contact_analysis is not contact_analysis
        or pi_system_analysis._molecular_analysis is not molecular_analysis
        or ch_pi._contact_analysis is not contact_analysis
        or ch_pi._molecular_analysis is not molecular_analysis
        or ch_pi._pi_system_analysis is not pi_system_analysis
    ):
        raise ValueError("molecular packing source lineage disagrees")
    return contact_analysis, molecular_analysis, pi_system_analysis


def _pi_stack_descriptor(candidate, features, molecules, context):
    first_feature = features[candidate.first_pi_system_candidate_id]
    second_feature = features[candidate.second_pi_system_candidate_id]
    first_component = molecules[first_feature.candidate.parent_molecule_orbit_id]
    second_component = molecules[second_feature.candidate.parent_molecule_orbit_id]
    first_ref = first_feature.atom_refs[0]
    second_ref = second_feature.atom_refs[0]
    atomic_relation = (
        first_ref.periodic_relation.inverse(context)
        .compose(candidate.canonical_periodic_relation, context)
        .compose(second_ref.periodic_relation, context)
    )
    molecule_relation = molecule_relation_for_atomic_pair(
        first_component,
        second_component,
        first_ref.independent_site_id,
        second_ref.independent_site_id,
        atomic_relation,
        context,
    )
    return canonical_molecule_contact_relation(
        first_component, second_component, molecule_relation, context
    )


def _ch_pi_descriptor(candidate, features, site_to_molecule, molecules, context):
    feature = features[candidate.pi_system_candidate_id]
    first_component = site_to_molecule[candidate.hydrogen_site_id]
    second_component = molecules[feature.candidate.parent_molecule_orbit_id]
    target_ref = feature.atom_refs[0]
    atomic_relation = candidate.hydrogen_pi_periodic_relation.compose(
        target_ref.periodic_relation, context
    )
    molecule_relation = molecule_relation_for_atomic_pair(
        first_component,
        second_component,
        candidate.hydrogen_site_id,
        target_ref.independent_site_id,
        atomic_relation,
        context,
    )
    return canonical_molecule_contact_relation(
        first_component, second_component, molecule_relation, context
    )


def _collect_packing_evidence(
    intermolecular: IntermolecularContactAnalysisResult,
    stacking: PiStackingAnalysisResult,
    ch_pi: CHPiAnalysisResult,
) -> tuple[_PackingEvidence, ...]:
    """Convert all interaction evidence into canonical molecule relations."""

    contact_analysis, molecular_analysis, pi_system_analysis = (
        _validate_source_lineage(intermolecular, stacking, ch_pi)
    )
    context = contact_analysis._symmetry_context
    matrix = contact_analysis._structure.cell.matrix
    molecules = {
        item.component_orbit_id: item for item in molecular_analysis.molecule_orbits
    }
    site_to_molecule = {
        site_id: molecule
        for molecule in molecular_analysis.molecule_orbits
        for site_id in molecule.independent_site_ids
    }
    features = {
        item.pi_system_candidate_id: _feature(item, matrix, context)
        for item in pi_system_analysis.candidates
    }
    evidence = []
    for contact in intermolecular.contact_orbits:
        descriptor = (
            contact.first_molecule_orbit_id,
            contact.second_molecule_orbit_id,
            contact.canonical_molecule_relation,
        )
        evidence.append(_PackingEvidence(
            descriptor,
            PackingInteractionRef(
                PackingInteractionKind.VAN_DER_WAALS_CONTACT,
                contact.intermolecular_contact_orbit_id,
            ),
            contact.status,
        ))
        evidence.extend(
            _PackingEvidence(
                descriptor,
                PackingInteractionRef(
                    PackingInteractionKind.HYDROGEN_BOND,
                    interpretation.interpretation_id,
                ),
                interpretation.status,
            )
            for interpretation in contact.interpretations
        )
    for candidate in stacking.candidates:
        evidence.append(_PackingEvidence(
            _pi_stack_descriptor(candidate, features, molecules, context),
            PackingInteractionRef(
                PackingInteractionKind.PI_STACKING,
                candidate.pi_stacking_candidate_id,
            ),
            candidate.status,
        ))
    for candidate in ch_pi.candidates:
        evidence.append(_PackingEvidence(
            _ch_pi_descriptor(
                candidate, features, site_to_molecule, molecules, context
            ),
            PackingInteractionRef(
                PackingInteractionKind.CH_PI,
                candidate.ch_pi_candidate_id,
            ),
            candidate.status,
        ))
    return tuple(sorted(set(evidence)))


def _molecule_pair_multiplicity(
    descriptor: MoleculeContactDescriptor,
    molecules: dict[str, MolecularComponentOrbit],
    context,
    mapping,
) -> int:
    """Count unique undirected molecule-pair owners in one reference cell."""

    first_id, second_id, relation = descriptor
    first = molecules[first_id]
    second = molecules[second_id]
    first_anchor = first.independent_site_ids[0]
    second_anchor = second.independent_site_ids[0]
    first_anchor_relation = dict(first.image_relations)[first_anchor]
    second_anchor_relation = dict(second.image_relations)[second_anchor]
    identity = identity_relation(context)
    if first_anchor_relation != identity or second_anchor_relation != identity:
        raise ValueError("molecular component canonical anchor is not its reference frame")
    owners = set()
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        owners.add(canonical_instance_owner(
            first_anchor,
            action,
            second_anchor,
            action.compose(relation, context),
            mapping,
        ))
    return len(owners)


def _aggregate_status(statuses, *, applicable: bool = True) -> ResolutionStatus:
    statuses = tuple(statuses)
    if ResolutionStatus.INCOMPLETE in statuses:
        return ResolutionStatus.INCOMPLETE
    if ResolutionStatus.AMBIGUOUS in statuses:
        return ResolutionStatus.AMBIGUOUS
    if not applicable or statuses and all(
        item is ResolutionStatus.NOT_APPLICABLE for item in statuses
    ):
        return ResolutionStatus.NOT_APPLICABLE
    return ResolutionStatus.RESOLVED


@dataclass(frozen=True, slots=True)
class PackingAnalyzer:
    """Assemble existing molecular interactions into one exact quotient graph."""

    def analyze(
        self,
        intermolecular_analysis: IntermolecularContactAnalysisResult,
        pi_stacking_analysis: PiStackingAnalysisResult,
        ch_pi_analysis: CHPiAnalysisResult,
    ) -> MolecularPackingGraph:
        if not isinstance(
            intermolecular_analysis, IntermolecularContactAnalysisResult
        ):
            raise TypeError(
                "intermolecular_analysis must be IntermolecularContactAnalysisResult"
            )
        if not isinstance(pi_stacking_analysis, PiStackingAnalysisResult):
            raise TypeError("pi_stacking_analysis must be PiStackingAnalysisResult")
        if not isinstance(ch_pi_analysis, CHPiAnalysisResult):
            raise TypeError("ch_pi_analysis must be CHPiAnalysisResult")
        contact_analysis, molecular_analysis, _ = _validate_source_lineage(
            intermolecular_analysis, pi_stacking_analysis, ch_pi_analysis
        )
        context = contact_analysis._symmetry_context
        mapping = contact_analysis._asymmetric_unit_mapping
        molecules = {
            item.component_orbit_id: item for item in molecular_analysis.molecule_orbits
        }
        evidence = _collect_packing_evidence(
            intermolecular_analysis, pi_stacking_analysis, ch_pi_analysis
        )
        grouped = defaultdict(list)
        for item in evidence:
            grouped[item.descriptor].append(item)

        connections = []
        for descriptor, items in sorted(grouped.items()):
            first_id, second_id, relation = descriptor
            status = _aggregate_status(item.status for item in items)
            connection_id = "packing-connection-orbit:" + _digest({
                "symmetry_action": context.symmetry_action_fingerprint,
                "first_molecule_orbit_id": first_id,
                "second_molecule_orbit_id": second_id,
                "operation_key": relation.operation_key,
                "lattice_translation": relation.lattice_translation,
            })
            connections.append(PackingConnectionOrbit(
                connection_id,
                first_id,
                second_id,
                relation,
                tuple(sorted({item.reference for item in items})),
                _molecule_pair_multiplicity(
                    descriptor, molecules, context, mapping
                ),
                status,
                (),
                (("method", "cristma.packing_connection_orbit:1"),),
            ))
        ordered_connections = tuple(sorted(
            connections, key=lambda item: item.packing_connection_orbit_id
        ))

        source_status = _aggregate_status((
            intermolecular_analysis.status,
            pi_stacking_analysis.status,
            ch_pi_analysis.status,
        ), applicable=bool(molecules))
        graph_components = _relation_graph_components(
            tuple(sorted(molecules)),
            tuple(
                (
                    item.packing_connection_orbit_id,
                    item.first_molecule_orbit_id,
                    item.second_molecule_orbit_id,
                    item.canonical_molecule_relation,
                )
                for item in ordered_connections
            ),
            context,
        )
        components = []
        connection_by_id = {
            item.packing_connection_orbit_id: item for item in ordered_connections
        }
        for item in graph_components:
            statuses = tuple(
                connection_by_id[connection_id].status
                for connection_id in item.edge_ids
            )
            component_status = _aggregate_status(
                (source_status, *statuses), applicable=True
            )
            component_id = "packing-component:" + _digest({
                "molecule_orbit_ids": item.node_ids,
                "connection_orbit_ids": item.edge_ids,
                "closure_relations": tuple(
                    (relation.operation_key, relation.lattice_translation)
                    for relation in item.closure_relations
                ),
                "periodic_generators": item.periodic_generators,
            })
            components.append(MolecularPackingComponent(
                component_id,
                item.node_ids,
                item.edge_ids,
                item.image_relations,
                item.closure_relations,
                item.rank,
                item.periodic_generators,
                component_status,
            ))
        ordered_components = tuple(sorted(
            components, key=lambda item: item.packing_component_id
        ))
        overall_status = _aggregate_status(
            (
                source_status,
                *(item.status for item in ordered_connections),
                *(item.status for item in ordered_components),
            ),
            applicable=bool(molecules),
        )
        return MolecularPackingGraph(
            intermolecular_analysis,
            pi_stacking_analysis,
            ch_pi_analysis,
            tuple(sorted(molecules)),
            ordered_connections,
            ordered_components,
            overall_status,
            (),
            (),
            tuple(sorted((
                ("ch_pi_configuration_fingerprint", _digest({
                    "configuration": ch_pi_analysis.configuration,
                })),
                ("intermolecular_configuration_fingerprint", _digest({
                    "configuration": intermolecular_analysis.configuration,
                })),
                ("method", "cristma.molecular_packing_graph:1"),
                ("pi_stacking_configuration_fingerprint", _digest({
                    "configuration": pi_stacking_analysis.configuration,
                })),
                ("symmetry_context_fingerprint", context.fingerprint),
            ))),
        )


__all__ = ["PackingAnalyzer"]
