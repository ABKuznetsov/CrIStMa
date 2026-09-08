"""Stabilizer quotient, undirected ownership, and pair-orbit identity."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from cristma.structure import CrystalStructure

from .asu_mapping import AsymmetricUnitMapping, SiteOrbitMapping
from .periodic_relation import PeriodicSymmetryRelation, identity_relation
from .symmetry_context import SymmetryContext, _digest
from .symmetry_pairs import (
    PairCandidateResult,
    PairTableStatus,
    SymmetryContactOrbit,
    SymmetryPairCandidate,
    SymmetryPairSearchPolicy,
    SymmetryPairTable,
    _source_fractional,
)


CanonicalPairDescriptor = tuple[str, str, PeriodicSymmetryRelation]
EndpointInstance = tuple[str, str, tuple[int, int, int]]
PairInstanceOwner = tuple[EndpointInstance, EndpointInstance]


def canonical_undirected_relation(
    first_id: str,
    second_id: str,
    relation: PeriodicSymmetryRelation,
    first_stabilizers: tuple[PeriodicSymmetryRelation, ...],
    second_stabilizers: tuple[PeriodicSymmetryRelation, ...],
    context: SymmetryContext,
) -> CanonicalPairDescriptor:
    """Quotient an exact relation by endpoint stabilizers and reversal."""

    if not first_id or not second_id:
        raise ValueError("endpoint identities must not be empty")
    if not first_stabilizers or not second_stabilizers:
        raise ValueError("endpoint stabilizers must not be empty")
    descriptors: list[CanonicalPairDescriptor] = []
    for left in first_stabilizers:
        for right in second_stabilizers:
            equivalent = left.compose(relation, context).compose(right, context)
            descriptors.append((first_id, second_id, equivalent))
            descriptors.append((second_id, first_id, equivalent.inverse(context)))
    return min(descriptors)


def canonical_pair_relation(
    candidate: SymmetryPairCandidate,
    first_mapping: SiteOrbitMapping,
    second_mapping: SiteOrbitMapping,
    context: SymmetryContext,
) -> CanonicalPairDescriptor:
    """Quotient a pair relation by both endpoint stabilizers and reversal."""

    if candidate.first_site_id != first_mapping.independent_site_id:
        raise ValueError("first pair endpoint and site mapping disagree")
    if candidate.second_site_id != second_mapping.independent_site_id:
        raise ValueError("second pair endpoint and site mapping disagree")
    return canonical_undirected_relation(
        candidate.first_site_id,
        candidate.second_site_id,
        candidate.relation,
        first_mapping.stabilizer_relations,
        second_mapping.stabilizer_relations,
        context,
    )


def periodic_endpoint_instance(
    site_id: str,
    relation: PeriodicSymmetryRelation,
    mapping: AsymmetricUnitMapping,
) -> EndpointInstance:
    matches = tuple(
        (image, equivalent)
        for image in mapping.by_site_id[site_id].reference_cell_images
        for equivalent in image.equivalent_relations
        if equivalent.operation_key == relation.operation_key
    )
    if len(matches) != 1:
        raise ValueError("operation does not identify one site image relation")
    image, normalized = matches[0]
    cell_translation = tuple(
        relation.lattice_translation[index] - normalized.lattice_translation[index]
        for index in range(3)
    )
    return site_id, image.image_id, cell_translation


def canonical_instance_owner(
    first_site_id: str,
    first_relation: PeriodicSymmetryRelation,
    second_site_id: str,
    second_relation: PeriodicSymmetryRelation,
    mapping: AsymmetricUnitMapping,
) -> PairInstanceOwner:
    """Remove global lattice translation and endpoint direction from an instance."""

    first = periodic_endpoint_instance(first_site_id, first_relation, mapping)
    second = periodic_endpoint_instance(second_site_id, second_relation, mapping)

    def anchored(
        owner: EndpointInstance,
        other: EndpointInstance,
    ) -> PairInstanceOwner:
        owner_cell = owner[2]
        relative = tuple(other[2][index] - owner_cell[index] for index in range(3))
        return (
            (owner[0], owner[1], (0, 0, 0)),
            (other[0], other[1], relative),
        )

    return min(anchored(first, second), anchored(second, first))


def _multiplicity_in_reference_cell(
    first_site_id: str,
    second_site_id: str,
    relation: PeriodicSymmetryRelation,
    context: SymmetryContext,
    mapping: AsymmetricUnitMapping,
) -> int:
    owners = set()
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        owners.add(
            canonical_instance_owner(
                first_site_id,
                action,
                second_site_id,
                action.compose(relation, context),
                mapping,
            )
        )
    return len(owners)


def _relation_geometry(
    first_site_id: str,
    second_site_id: str,
    relation: PeriodicSymmetryRelation,
    structure: CrystalStructure,
    context: SymmetryContext,
    mapping: AsymmetricUnitMapping,
) -> tuple[float, tuple[float, float, float]]:
    first = np.asarray(_source_fractional(mapping, first_site_id, context))
    second = np.asarray(_source_fractional(mapping, second_site_id, context))
    operation = context.operation_by_key(relation.operation_key)
    transformed = np.asarray(
        tuple(
            math.fsum(
                float(coefficient) * float(coordinate)
                for coefficient, coordinate in zip(row, second, strict=True)
            )
            + float(offset)
            + relation.lattice_translation[index]
            for index, (row, offset) in enumerate(
                zip(operation.rotation, operation.translation, strict=True)
            )
        )
    )
    vector = (transformed - first) @ structure.cell.matrix
    return float(np.linalg.norm(vector)), tuple(float(value) for value in vector)


def build_symmetry_pair_table(
    result: PairCandidateResult,
    structure: CrystalStructure,
    context: SymmetryContext,
    mapping: AsymmetricUnitMapping,
    policy: SymmetryPairSearchPolicy,
) -> SymmetryPairTable:
    """Aggregate raw candidates into exact geometric pair orbits."""

    grouped: dict[CanonicalPairDescriptor, list[SymmetryPairCandidate]] = defaultdict(list)
    for candidate in result.candidates:
        descriptor = canonical_pair_relation(
            candidate,
            mapping.by_site_id[candidate.first_site_id],
            mapping.by_site_id[candidate.second_site_id],
            context,
        )
        grouped[descriptor].append(candidate)

    status = PairTableStatus.COMPLETE if result.complete else PairTableStatus.INCOMPLETE
    orbits: list[SymmetryContactOrbit] = []
    for descriptor, candidates in grouped.items():
        first_id, second_id, canonical_relation = descriptor
        distance, vector = _relation_geometry(
            first_id,
            second_id,
            canonical_relation,
            structure,
            context,
            mapping,
        )
        geometry_orbit_id = "geometry-orbit:" + _digest(
            {
                "symmetry_action": context.symmetry_action_fingerprint,
                "first_site_id": first_id,
                "second_site_id": second_id,
                "operation_key": canonical_relation.operation_key,
                "lattice_translation": canonical_relation.lattice_translation,
            }
        )
        equivalent_relations = tuple(
            sorted({candidate.relation for candidate in candidates})
        )
        orbits.append(
            SymmetryContactOrbit(
                geometry_orbit_id=geometry_orbit_id,
                first_independent_site_id=first_id,
                second_independent_site_id=second_id,
                canonical_relation=canonical_relation,
                equivalent_relations=equivalent_relations,
                endpoint_stabilizers=(
                    mapping.by_site_id[first_id].stabilizer_relations,
                    mapping.by_site_id[second_id].stabilizer_relations,
                ),
                representative_distance=distance,
                representative_vector_cartesian=vector,
                multiplicity_in_reference_cell=_multiplicity_in_reference_cell(
                    first_id,
                    second_id,
                    canonical_relation,
                    context,
                    mapping,
                ),
                status=status,
                diagnostics=result.diagnostics if not result.complete else (),
                provenance=(("canonicalization", "endpoint_stabilizer_double_coset"),),
            )
        )
    orbits.sort(key=lambda orbit: orbit.geometry_orbit_id)
    return SymmetryPairTable(
        contact_orbits=tuple(orbits),
        symmetry_context_fingerprint=context.fingerprint,
        asymmetric_unit_mapping_fingerprint=mapping.fingerprint,
        cutoff=policy.cutoff,
        distance_tolerance=policy.distance_tolerance,
        status=status,
        diagnostics=result.diagnostics,
        provenance=(
            *result.provenance,
            ("canonicalization", "endpoint_stabilizer_double_coset"),
            ("ownership", "canonical_endpoint_reference_cell"),
        ),
    )


__all__ = [
    "CanonicalPairDescriptor",
    "EndpointInstance",
    "PairInstanceOwner",
    "build_symmetry_pair_table",
    "canonical_instance_owner",
    "canonical_pair_relation",
    "canonical_undirected_relation",
    "periodic_endpoint_instance",
]
