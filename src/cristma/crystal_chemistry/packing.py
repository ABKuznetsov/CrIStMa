"""Exact source-relation conversion for molecular packing graphs."""

from __future__ import annotations

from dataclasses import dataclass

from cristma.crystallography import PeriodicSymmetryRelation, identity_relation
from cristma.crystallography.pair_canonical import canonical_instance_owner

from .ch_pi import CHPiAnalysisResult
from .intermolecular import (
    canonical_molecule_contact_relation,
    molecule_relation_for_atomic_pair,
)
from .intermolecular_models import IntermolecularContactAnalysisResult
from .models import ResolutionStatus
from .molecular import MolecularComponentOrbit
from .packing_models import PackingInteractionKind, PackingInteractionRef
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


__all__: list[str] = []
