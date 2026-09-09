"""Directed orbit-first C-H...pi geometry from explicit molecular bonds."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import math
from numbers import Real

import numpy as np

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.pair_canonical import periodic_endpoint_instance
from cristma.crystallography.symmetry_context import _digest
from cristma.crystallography.symmetry_pairs import _source_fractional, _translation_ranges
from cristma.diagnostics import Diagnostic, Severity

from .contact_analysis import ContactAnalysisResult
from .intermolecular import molecule_relation_for_atomic_pair, _same_molecule_instance
from .models import ResolutionStatus
from .molecular import MolecularAnalysisResult, MolecularBondOrbit
from .pi_stacking import (
    _Feature,
    _feature,
    _orientation_keys,
    _ring_stabilizer,
    _translated_refs,
    _transformed_centroid,
)
from .pi_systems import PiSystemAnalysisResult


@dataclass(frozen=True, slots=True)
class CHPiPolicy:
    maximum_hydrogen_centroid_distance_angstrom: float = 3.0
    minimum_carbon_hydrogen_centroid_angle_deg: float = 120.0
    distance_tolerance_angstrom: float = 1e-12
    maximum_feature_pairs: int = 1_000_000
    maximum_lattice_translations: int = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "maximum_hydrogen_centroid_distance_angstrom",
            "minimum_carbon_hydrogen_centroid_angle_deg",
            "distance_tolerance_angstrom",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, normalized)
        if self.maximum_hydrogen_centroid_distance_angstrom <= 0.0:
            raise ValueError("maximum hydrogen-centroid distance must be positive")
        if self.minimum_carbon_hydrogen_centroid_angle_deg > 180.0:
            raise ValueError("minimum C-H-centroid angle must not exceed 180 degrees")
        for name in ("maximum_feature_pairs", "maximum_lattice_translations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def get_config(self) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(
            (name, getattr(self, name)) for name in self.__dataclass_fields__
        ))

    def clone(self, **changes: object) -> "CHPiPolicy":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class CHPiCandidate:
    ch_pi_candidate_id: str
    donor_carbon_site_id: str
    hydrogen_site_id: str
    donor_hydrogen_bond_orbit_id: str
    pi_system_candidate_id: str
    hydrogen_pi_periodic_relation: PeriodicSymmetryRelation
    hydrogen_centroid_distance_angstrom: float
    carbon_hydrogen_centroid_angle_deg: float
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.ch_pi_candidate_id,
            self.donor_carbon_site_id,
            self.hydrogen_site_id,
            self.donor_hydrogen_bond_orbit_id,
            self.pi_system_candidate_id,
        )
        if not all(isinstance(value, str) and value for value in identities):
            raise ValueError("C-H...pi identities must not be empty")
        if (
            not math.isfinite(self.hydrogen_centroid_distance_angstrom)
            or self.hydrogen_centroid_distance_angstrom <= 0.0
        ):
            raise ValueError("hydrogen-centroid distance must be finite and positive")
        if not (
            math.isfinite(self.carbon_hydrogen_centroid_angle_deg)
            and 0.0 <= self.carbon_hydrogen_centroid_angle_deg <= 180.0
        ):
            raise ValueError("C-H-centroid angle must lie within [0, 180] degrees")
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ) or self.multiplicity_in_reference_cell <= 0:
            raise ValueError("C-H...pi multiplicity must be a positive integer")


@dataclass(frozen=True, slots=True)
class CHPiAnalysisResult:
    _contact_analysis: ContactAnalysisResult = field(repr=False, compare=False)
    _molecular_analysis: MolecularAnalysisResult = field(repr=False, compare=False)
    _pi_system_analysis: PiSystemAnalysisResult = field(repr=False, compare=False)
    candidates: tuple[CHPiCandidate, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        identifiers = tuple(item.ch_pi_candidate_id for item in self.candidates)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("C-H...pi candidate IDs must be unique and sorted")
        if any(
            item.status is ResolutionStatus.INCOMPLETE for item in self.candidates
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete C-H...pi candidate must propagate")


@dataclass(frozen=True, slots=True)
class _Donor:
    bond: MolecularBondOrbit
    carbon_site_id: str
    hydrogen_site_id: str
    hydrogen_to_carbon_relation: PeriodicSymmetryRelation


def _donors(molecular: MolecularAnalysisResult, context) -> tuple[_Donor, ...]:
    result = []
    for bond in molecular.bond_orbits:
        accepted = set()
        for item in bond.accepted_component_pairs:
            accepted.add(item.species_symbols)
        if ("C", "H") in accepted:
            result.append(_Donor(
                bond,
                bond.first_independent_site_id,
                bond.second_independent_site_id,
                bond.periodic_relation.inverse(context),
            ))
        if ("H", "C") in accepted:
            result.append(_Donor(
                bond,
                bond.second_independent_site_id,
                bond.first_independent_site_id,
                bond.periodic_relation,
            ))
    return tuple(sorted(result, key=lambda item: (
        item.bond.bond_orbit_id, item.hydrogen_site_id, item.carbon_site_id
    )))


def _canonical_directed_relation(donor, feature, relation, contact_analysis):
    context = contact_analysis._symmetry_context
    hydrogen_stabilizers = contact_analysis._asymmetric_unit_mapping.by_site_id[
        donor.hydrogen_site_id
    ].stabilizer_relations
    return min(
        left.compose(relation, context).compose(right, context)
        for left in hydrogen_stabilizers
        for right in _ring_stabilizer(feature, context)
    )


def _relation_vector(first_site_id, second_site_id, relation, contact_analysis):
    context = contact_analysis._symmetry_context
    mapping = contact_analysis._asymmetric_unit_mapping
    operation = context.operation_by_key(relation.operation_key)
    first = np.asarray(_source_fractional(mapping, first_site_id, context))
    second = _source_fractional(mapping, second_site_id, context)
    transformed = np.asarray(tuple(
        math.fsum(
            float(coefficient) * float(coordinate)
            for coefficient, coordinate in zip(row, second, strict=True)
        ) + float(offset) + relation.lattice_translation[index]
        for index, (row, offset) in enumerate(
            zip(operation.rotation, operation.translation, strict=True)
        )
    ))
    return (transformed - first) @ contact_analysis._structure.cell.matrix


def _geometry(donor, feature, relation, contact_analysis):
    context = contact_analysis._symmetry_context
    matrix = contact_analysis._structure.cell.matrix
    mapping = contact_analysis._asymmetric_unit_mapping
    hydrogen = np.asarray(_source_fractional(
        mapping, donor.hydrogen_site_id, context
    ))
    centroid = _transformed_centroid(feature, relation, context)
    hydrogen_to_centroid = (centroid - hydrogen) @ matrix
    hydrogen_to_carbon = _relation_vector(
        donor.hydrogen_site_id,
        donor.carbon_site_id,
        donor.hydrogen_to_carbon_relation,
        contact_analysis,
    )
    distance = float(np.linalg.norm(hydrogen_to_centroid))
    denominator = float(distance * np.linalg.norm(hydrogen_to_carbon))
    if denominator <= 0.0:
        return distance, 0.0
    cosine = float(np.clip(
        np.dot(hydrogen_to_carbon, hydrogen_to_centroid) / denominator,
        -1.0,
        1.0,
    ))
    return distance, math.degrees(math.acos(cosine))


def _is_own_molecule(donor, feature, relation, molecular, contact_analysis):
    component_by_id = {
        item.component_orbit_id: item for item in molecular.molecule_orbits
    }
    site_to_component = {
        site_id: component
        for component in molecular.molecule_orbits
        for site_id in component.independent_site_ids
    }
    donor_component = site_to_component.get(donor.hydrogen_site_id)
    target_component = component_by_id.get(feature.candidate.parent_molecule_orbit_id)
    if donor_component is None or target_component is None:
        raise ValueError("C-H...pi endpoint is outside a discrete molecule")
    target_ref = feature.atom_refs[0]
    atomic_relation = relation.compose(target_ref.periodic_relation,
                                       contact_analysis._symmetry_context)
    molecule_relation = molecule_relation_for_atomic_pair(
        donor_component,
        target_component,
        donor.hydrogen_site_id,
        target_ref.independent_site_id,
        atomic_relation,
        contact_analysis._symmetry_context,
    )
    return _same_molecule_instance(
        donor_component,
        target_component,
        molecule_relation,
        contact_analysis._symmetry_context,
    )


def _multiplicity(donor, feature, relation, contact_analysis):
    context = contact_analysis._symmetry_context
    mapping = contact_analysis._asymmetric_unit_mapping
    owners = set()
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        hydrogen_instance = periodic_endpoint_instance(
            donor.hydrogen_site_id, action, mapping
        )
        shift = PeriodicSymmetryRelation(
            context.identity_operation_key,
            tuple(-value for value in hydrogen_instance[2]),
        )
        target_action = shift.compose(
            action.compose(relation, context), context
        )
        target_refs = _translated_refs(feature.atom_refs, target_action, context)
        owners.add((
            hydrogen_instance[1],
            feature.candidate.pi_system_candidate_id,
            min(_orientation_keys(target_refs)),
        ))
    return len(owners)


@dataclass(frozen=True, slots=True)
class CHPiAnalyzer:
    policy: CHPiPolicy = field(default_factory=CHPiPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CHPiPolicy):
            raise TypeError("policy must be CHPiPolicy")

    def analyze(
        self,
        contact_analysis: ContactAnalysisResult,
        molecular_analysis: MolecularAnalysisResult,
        pi_system_analysis: PiSystemAnalysisResult,
    ) -> CHPiAnalysisResult:
        if not isinstance(contact_analysis, ContactAnalysisResult):
            raise TypeError("contact_analysis must be ContactAnalysisResult")
        if not isinstance(molecular_analysis, MolecularAnalysisResult):
            raise TypeError("molecular_analysis must be MolecularAnalysisResult")
        if not isinstance(pi_system_analysis, PiSystemAnalysisResult):
            raise TypeError("pi_system_analysis must be PiSystemAnalysisResult")
        if (
            molecular_analysis.contact_analysis_fingerprint
            != contact_analysis.fingerprint
            or pi_system_analysis._contact_analysis is not contact_analysis
            or pi_system_analysis._molecular_analysis is not molecular_analysis
        ):
            raise ValueError("C-H...pi source lineage disagrees")

        provenance = tuple(sorted((
            ("method", "cristma.ch_pi_analysis:1"),
            ("symmetry_context_fingerprint",
             contact_analysis._symmetry_context.fingerprint),
        )))
        diagnostics = list(pi_system_analysis.diagnostics)
        structure = contact_analysis._structure
        has_explicit_hydrogen = any(
            component.element == "H"
            for site in structure.sites
            for component in site.components
        )
        if not has_explicit_hydrogen:
            diagnostics.append(Diagnostic(
                Severity.INFO,
                "crystal_chemistry.ch_pi.explicit_hydrogen_absent",
                "C-H...pi candidates were not evaluated because no explicit H/D site is present",
            ))

        context = contact_analysis._symmetry_context
        matrix = structure.cell.matrix
        donors = _donors(molecular_analysis, context)
        features = tuple(
            _feature(item, matrix, context) for item in pi_system_analysis.candidates
        )
        ranges = _translation_ranges(
            matrix,
            self.policy.maximum_hydrogen_centroid_distance_angstrom
            + self.policy.distance_tolerance_angstrom,
        )
        translations = tuple(product(*ranges))
        complete = True
        if len(translations) > self.policy.maximum_lattice_translations:
            translations = translations[:self.policy.maximum_lattice_translations]
            complete = False
            diagnostics.append(Diagnostic(
                Severity.WARNING,
                "crystal_chemistry.ch_pi.search_limit_reached",
                "C-H...pi search reached maximum_lattice_translations",
            ))

        descriptors = set()
        tested = 0
        stopped = False
        for donor in donors:
            for feature in features:
                for operation_key in context.operation_keys:
                    for translation in translations:
                        tested += 1
                        if tested > self.policy.maximum_feature_pairs:
                            complete = False
                            stopped = True
                            diagnostics.append(Diagnostic(
                                Severity.WARNING,
                                "crystal_chemistry.ch_pi.search_limit_reached",
                                "C-H...pi search reached maximum_feature_pairs",
                            ))
                            break
                        raw = PeriodicSymmetryRelation(operation_key, translation)
                        relation = _canonical_directed_relation(
                            donor, feature, raw, contact_analysis
                        )
                        if _is_own_molecule(
                            donor, feature, relation, molecular_analysis, contact_analysis
                        ):
                            continue
                        distance, angle = _geometry(
                            donor, feature, relation, contact_analysis
                        )
                        tolerance = self.policy.distance_tolerance_angstrom
                        if (
                            distance <= self.policy.maximum_hydrogen_centroid_distance_angstrom + tolerance
                            and angle >= self.policy.minimum_carbon_hydrogen_centroid_angle_deg
                        ):
                            descriptors.add((
                                donor.bond.bond_orbit_id,
                                donor.carbon_site_id,
                                donor.hydrogen_site_id,
                                feature.candidate.pi_system_candidate_id,
                                relation,
                            ))
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break

        donor_by_id = {item.bond.bond_orbit_id: item for item in donors}
        feature_by_id = {
            item.candidate.pi_system_candidate_id: item for item in features
        }
        candidates = []
        for bond_id, carbon_id, hydrogen_id, pi_id, relation in sorted(descriptors):
            donor = donor_by_id[bond_id]
            feature = feature_by_id[pi_id]
            distance, angle = _geometry(donor, feature, relation, contact_analysis)
            status = (
                ResolutionStatus.INCOMPLETE
                if not complete or ResolutionStatus.INCOMPLETE in (
                    donor.bond.status, feature.candidate.status
                )
                else ResolutionStatus.AMBIGUOUS
                if ResolutionStatus.AMBIGUOUS in (
                    donor.bond.status, feature.candidate.status
                )
                else ResolutionStatus.RESOLVED
            )
            candidate_id = "ch-pi-candidate:" + _digest({
                "donor_hydrogen_bond_orbit_id": bond_id,
                "pi_system_candidate_id": pi_id,
                "operation_key": relation.operation_key,
                "lattice_translation": relation.lattice_translation,
            })
            candidates.append(CHPiCandidate(
                candidate_id,
                carbon_id,
                hydrogen_id,
                bond_id,
                pi_id,
                relation,
                distance,
                angle,
                _multiplicity(donor, feature, relation, contact_analysis),
                status,
                (),
                (
                    ("criterion", "explicit_C-H_or_C-D_to_pi_centroid_geometry"),
                    ("method", "cristma.ch_pi_candidate:1"),
                ),
            ))
        ordered = tuple(sorted(candidates, key=lambda item: item.ch_pi_candidate_id))
        status = (
            ResolutionStatus.INCOMPLETE
            if not complete or pi_system_analysis.status is ResolutionStatus.INCOMPLETE
            or molecular_analysis.status is ResolutionStatus.INCOMPLETE
            else ResolutionStatus.AMBIGUOUS
            if pi_system_analysis.status is ResolutionStatus.AMBIGUOUS
            or molecular_analysis.status is ResolutionStatus.AMBIGUOUS
            else ResolutionStatus.NOT_APPLICABLE
            if pi_system_analysis.status is ResolutionStatus.NOT_APPLICABLE
            else ResolutionStatus.RESOLVED
        )
        return CHPiAnalysisResult(
            contact_analysis,
            molecular_analysis,
            pi_system_analysis,
            ordered,
            status,
            tuple(dict.fromkeys(diagnostics)),
            self.policy.get_config(),
            provenance,
        )


__all__ = ["CHPiAnalysisResult", "CHPiAnalyzer", "CHPiCandidate", "CHPiPolicy"]
