"""Bounded orbit-first geometry for molecular pi-stacking candidates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import math
from numbers import Real

import numpy as np

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.pair_canonical import canonical_undirected_relation
from cristma.crystallography.symmetry_context import _digest
from cristma.crystallography.symmetry_pairs import _translation_ranges
from cristma.diagnostics import Diagnostic, Severity

from .models import ResolutionStatus
from .molecular_rings import MolecularAtomOrbitRef
from .pi_systems import PiSystemAnalysisResult, PiSystemCandidate


@dataclass(frozen=True, slots=True)
class PiStackingPolicy:
    minimum_centroid_distance_angstrom: float = 3.2
    maximum_centroid_distance_angstrom: float = 4.2
    maximum_plane_angle_deg: float = 20.0
    maximum_lateral_offset_angstrom: float = 2.0
    distance_tolerance_angstrom: float = 1e-12
    maximum_feature_pairs: int = 1_000_000
    maximum_lattice_translations: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "minimum_centroid_distance_angstrom",
            "maximum_centroid_distance_angstrom",
            "maximum_plane_angle_deg",
            "maximum_lateral_offset_angstrom",
            "distance_tolerance_angstrom",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, normalized)
        if self.minimum_centroid_distance_angstrom <= 0.0:
            raise ValueError("minimum centroid distance must be positive")
        if (
            self.maximum_centroid_distance_angstrom
            < self.minimum_centroid_distance_angstrom
        ):
            raise ValueError("maximum centroid distance must not be below minimum")
        if self.maximum_plane_angle_deg > 90.0:
            raise ValueError("maximum plane angle must not exceed 90 degrees")
        for name in ("maximum_feature_pairs", "maximum_lattice_translations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def get_config(self) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(
            (name, getattr(self, name))
            for name in self.__dataclass_fields__
        ))

    def clone(self, **changes: object) -> "PiStackingPolicy":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PiStackingCandidate:
    pi_stacking_candidate_id: str
    first_pi_system_candidate_id: str
    second_pi_system_candidate_id: str
    canonical_periodic_relation: PeriodicSymmetryRelation
    centroid_distance_angstrom: float
    plane_angle_deg: float
    lateral_offset_angstrom: float
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.pi_stacking_candidate_id,
            self.first_pi_system_candidate_id,
            self.second_pi_system_candidate_id,
        )
        if not all(isinstance(value, str) and value for value in identities):
            raise ValueError("pi-stacking identities must not be empty")
        if self.first_pi_system_candidate_id > self.second_pi_system_candidate_id:
            raise ValueError("pi-stacking endpoints must use canonical order")
        for name in (
            "centroid_distance_angstrom",
            "plane_angle_deg",
            "lateral_offset_angstrom",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.plane_angle_deg <= 90.0:
            raise ValueError("plane angle must lie within [0, 90] degrees")
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ) or self.multiplicity_in_reference_cell <= 0:
            raise ValueError("pi-stacking multiplicity must be a positive integer")


@dataclass(frozen=True, slots=True)
class PiStackingAnalysisResult:
    _pi_system_analysis: PiSystemAnalysisResult = field(repr=False, compare=False)
    candidates: tuple[PiStackingCandidate, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        identifiers = tuple(item.pi_stacking_candidate_id for item in self.candidates)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("pi-stacking candidate IDs must be unique and sorted")
        if any(
            item.status is ResolutionStatus.INCOMPLETE for item in self.candidates
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete pi-stacking candidate must propagate")


@dataclass(frozen=True, slots=True)
class _Feature:
    candidate: PiSystemCandidate
    atom_refs: tuple[MolecularAtomOrbitRef, ...]
    centroid_fractional: tuple[float, float, float]


def _translated_refs(refs, action, context):
    return tuple(
        MolecularAtomOrbitRef(
            ref.independent_site_id,
            action.compose(ref.periodic_relation, context),
        )
        for ref in refs
    )


def _orientation_keys(refs):
    count = len(refs)
    keys = []
    for direction in (1, -1):
        for start in range(count):
            indices = tuple((start + direction * step) % count for step in range(count))
            keys.append(tuple(
                (
                    refs[index].independent_site_id,
                    refs[index].periodic_relation.operation_key,
                    refs[index].periodic_relation.lattice_translation,
                )
                for index in indices
            ))
    return tuple(keys)


def _translate_refs(refs, translation, context):
    action = PeriodicSymmetryRelation(context.identity_operation_key, translation)
    return _translated_refs(refs, action, context)


def _normalized_ring_key(refs, context):
    candidates = []
    for key in _orientation_keys(refs):
        anchor = key[0][2]
        candidates.append(tuple(
            (site_id, operation_key, tuple(value - anchor[i] for i, value in enumerate(lattice)))
            for site_id, operation_key, lattice in key
        ))
    return min(candidates)


def _feature(candidate, matrix, context):
    fractional = np.asarray(candidate.centroid_cartesian) @ np.linalg.inv(matrix)
    owner = tuple(math.floor(float(value)) for value in fractional)
    shift = tuple(-value for value in owner)
    return _Feature(
        candidate,
        _translate_refs(candidate.atom_refs, shift, context),
        tuple(float(value - owner[index]) for index, value in enumerate(fractional)),
    )


def _ring_stabilizer(feature, context):
    base = _normalized_ring_key(feature.atom_refs, context)
    stabilizers = []
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        transformed = _translated_refs(feature.atom_refs, action, context)
        if _normalized_ring_key(transformed, context) != base:
            continue
        raw_key = min(_orientation_keys(transformed))
        anchor = raw_key[0][2]
        stabilizers.append(
            PeriodicSymmetryRelation(
                operation_key,
                tuple(-value for value in anchor),
            )
        )
    if not stabilizers:
        raise ValueError("pi-system feature has no identity stabilizer")
    return tuple(sorted(set(stabilizers)))


def _transformed_centroid(feature, relation, context):
    operation = context.operation_by_key(relation.operation_key)
    return np.asarray(tuple(
        math.fsum(
            float(coefficient) * coordinate
            for coefficient, coordinate in zip(
                row, feature.centroid_fractional, strict=True
            )
        ) + float(offset) + relation.lattice_translation[index]
        for index, (row, offset) in enumerate(
            zip(operation.rotation, operation.translation, strict=True)
        )
    ))


def _transformed_normal(normal, relation, matrix, context):
    rotation = np.asarray(
        context.operation_by_key(relation.operation_key).rotation,
        dtype=float,
    )
    cartesian_rotation = np.linalg.inv(matrix) @ rotation.T @ matrix
    transformed = np.asarray(normal) @ cartesian_rotation
    return transformed / np.linalg.norm(transformed)


def _geometry(first, second, relation, matrix, context):
    first_centroid = np.asarray(first.centroid_fractional)
    second_centroid = _transformed_centroid(second, relation, context)
    vector = (second_centroid - first_centroid) @ matrix
    distance = float(np.linalg.norm(vector))
    first_normal = np.asarray(first.candidate.normal_cartesian)
    second_normal = _transformed_normal(
        second.candidate.normal_cartesian, relation, matrix, context
    )
    cosine = float(np.clip(abs(np.dot(first_normal, second_normal)), 0.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    if np.dot(first_normal, second_normal) < 0.0:
        second_normal = -second_normal
    mean_normal = first_normal + second_normal
    mean_normal /= np.linalg.norm(mean_normal)
    lateral = float(np.linalg.norm(vector - np.dot(vector, mean_normal) * mean_normal))
    return distance, angle, lateral


def _pair_owner_key(first, second, relation, context):
    descriptors = []
    for owner, other, action in (
        (first, second, relation),
        (second, first, relation.inverse(context)),
    ):
        owner_key = min(_orientation_keys(owner.atom_refs))
        anchor = owner_key[0][2]
        shift = tuple(-value for value in anchor)
        owner_refs = _translate_refs(owner.atom_refs, shift, context)
        other_refs = _translate_refs(
            _translated_refs(other.atom_refs, action, context), shift, context
        )
        descriptors.append((
            owner.candidate.pi_system_candidate_id,
            min(_orientation_keys(owner_refs)),
            other.candidate.pi_system_candidate_id,
            min(_orientation_keys(other_refs)),
        ))
    return min(descriptors)


def _multiplicity(first, second, relation, context):
    owners = set()
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        owners.add(_pair_owner_key(
            replace(first, atom_refs=_translated_refs(first.atom_refs, action, context)),
            replace(second, atom_refs=_translated_refs(second.atom_refs, action, context)),
            action.compose(relation, context).compose(action.inverse(context), context),
            context,
        ))
    return len(owners)


@dataclass(frozen=True, slots=True)
class PiStackingAnalyzer:
    policy: PiStackingPolicy = field(default_factory=PiStackingPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PiStackingPolicy):
            raise TypeError("policy must be PiStackingPolicy")

    def analyze(self, contact_analysis, molecular_analysis, pi_system_analysis):
        if not isinstance(pi_system_analysis, PiSystemAnalysisResult):
            raise TypeError("pi_system_analysis must be PiSystemAnalysisResult")
        if (
            pi_system_analysis._contact_analysis is not contact_analysis
            or pi_system_analysis._molecular_analysis is not molecular_analysis
        ):
            raise ValueError("pi-stacking source lineage disagrees")
        provenance = tuple(sorted((
            ("method", "cristma.pi_stacking_analysis:1"),
            (
                "symmetry_context_fingerprint",
                contact_analysis._symmetry_context.fingerprint,
            ),
        )))
        if pi_system_analysis.status is ResolutionStatus.NOT_APPLICABLE:
            return PiStackingAnalysisResult(
                pi_system_analysis, (), ResolutionStatus.NOT_APPLICABLE,
                pi_system_analysis.diagnostics, self.policy.get_config(), provenance,
            )

        context = contact_analysis._symmetry_context
        matrix = contact_analysis._structure.cell.matrix
        features = tuple(
            _feature(item, matrix, context)
            for item in pi_system_analysis.candidates
        )
        feature_by_id = {
            item.candidate.pi_system_candidate_id: item for item in features
        }
        stabilizers = {
            item.candidate.pi_system_candidate_id: _ring_stabilizer(item, context)
            for item in features
        }
        ranges = _translation_ranges(
            matrix,
            self.policy.maximum_centroid_distance_angstrom
            + self.policy.distance_tolerance_angstrom,
        )
        translations = tuple(product(*ranges))
        complete = True
        diagnostics = list(pi_system_analysis.diagnostics)
        if len(translations) > self.policy.maximum_lattice_translations:
            translations = translations[: self.policy.maximum_lattice_translations]
            complete = False
            diagnostics.append(Diagnostic(
                Severity.WARNING,
                "crystal_chemistry.pi_stacking.search_limit_reached",
                "Pi-stacking search reached maximum_lattice_translations",
            ))

        descriptors = set()
        tested = 0
        stopped = False
        for first_index, first in enumerate(features):
            for second in features[first_index:]:
                for operation_key in context.operation_keys:
                    for translation in translations:
                        tested += 1
                        if tested > self.policy.maximum_feature_pairs:
                            complete = False
                            stopped = True
                            diagnostics.append(Diagnostic(
                                Severity.WARNING,
                                "crystal_chemistry.pi_stacking.search_limit_reached",
                                "Pi-stacking search reached maximum_feature_pairs",
                            ))
                            break
                        relation = PeriodicSymmetryRelation(operation_key, translation)
                        descriptor = canonical_undirected_relation(
                            first.candidate.pi_system_candidate_id,
                            second.candidate.pi_system_candidate_id,
                            relation,
                            stabilizers[first.candidate.pi_system_candidate_id],
                            stabilizers[second.candidate.pi_system_candidate_id],
                            context,
                        )
                        identity_descriptor = canonical_undirected_relation(
                            first.candidate.pi_system_candidate_id,
                            second.candidate.pi_system_candidate_id,
                            PeriodicSymmetryRelation(
                                context.identity_operation_key, (0, 0, 0)
                            ),
                            stabilizers[first.candidate.pi_system_candidate_id],
                            stabilizers[second.candidate.pi_system_candidate_id],
                            context,
                        )
                        if (
                            first.candidate.parent_molecule_orbit_id
                            == second.candidate.parent_molecule_orbit_id
                            and descriptor == identity_descriptor
                        ):
                            continue
                        distance, angle, lateral = _geometry(
                            first, second, relation, matrix, context
                        )
                        tolerance = self.policy.distance_tolerance_angstrom
                        if not (
                            self.policy.minimum_centroid_distance_angstrom - tolerance
                            <= distance
                            <= self.policy.maximum_centroid_distance_angstrom + tolerance
                            and angle <= self.policy.maximum_plane_angle_deg
                            and lateral <= self.policy.maximum_lateral_offset_angstrom + tolerance
                        ):
                            continue
                        descriptors.add(descriptor)
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break

        candidates = []
        for first_id, second_id, relation in sorted(descriptors):
            first = feature_by_id[first_id]
            second = feature_by_id[second_id]
            distance, angle, lateral = _geometry(first, second, relation, matrix, context)
            status = (
                ResolutionStatus.INCOMPLETE
                if not complete
                or ResolutionStatus.INCOMPLETE in (
                    first.candidate.status, second.candidate.status
                )
                else ResolutionStatus.AMBIGUOUS
                if ResolutionStatus.AMBIGUOUS in (
                    first.candidate.status, second.candidate.status
                )
                else ResolutionStatus.RESOLVED
            )
            candidate_id = "pi-stacking-candidate:" + _digest({
                "first_pi_system_candidate_id": first_id,
                "second_pi_system_candidate_id": second_id,
                "operation_key": relation.operation_key,
                "lattice_translation": relation.lattice_translation,
            })
            candidates.append(PiStackingCandidate(
                candidate_id, first_id, second_id, relation,
                distance, angle, lateral,
                _multiplicity(first, second, relation, context),
                status,
                (),
                (("method", "cristma.pi_stacking_candidate:1"),),
            ))
        ordered = tuple(sorted(candidates, key=lambda item: item.pi_stacking_candidate_id))
        status = (
            ResolutionStatus.INCOMPLETE
            if not complete or pi_system_analysis.status is ResolutionStatus.INCOMPLETE
            else ResolutionStatus.AMBIGUOUS
            if pi_system_analysis.status is ResolutionStatus.AMBIGUOUS
            else ResolutionStatus.RESOLVED
        )
        return PiStackingAnalysisResult(
            pi_system_analysis,
            ordered,
            status,
            tuple(dict.fromkeys(diagnostics)),
            self.policy.get_config(),
            provenance,
        )


__all__ = [
    "PiStackingAnalysisResult",
    "PiStackingAnalyzer",
    "PiStackingCandidate",
    "PiStackingPolicy",
]
