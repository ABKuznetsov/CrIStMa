"""Orbit-first molecular connectivity calculated from resolved contact orbits."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
import math

from cristma.chemistry import InteractionLayer, InteractionPriority
from cristma.crystallography import (
    PairTableStatus,
    PeriodicSymmetryRelation,
    identity_relation,
)
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic, Severity

from .contact_analysis import ContactAnalysisResult
from .models import ComponentPairInterpretation, ResolutionStatus
from .periodic_connectivity import integer_translation_lattice_basis


Translation = tuple[int, int, int]
ZERO: Translation = (0, 0, 0)


@dataclass(frozen=True, slots=True)
class MolecularAnalysisPolicy:
    """Explicit distance criterion for covalent molecular connectivity."""

    maximum_bond_rho: float

    def __post_init__(self) -> None:
        value = self.maximum_bond_rho
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("maximum_bond_rho must be a real number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError("maximum_bond_rho must be finite and positive")
        object.__setattr__(self, "maximum_bond_rho", normalized)

    def get_config(self) -> dict[str, float]:
        return {"maximum_bond_rho": self.maximum_bond_rho}

    def clone(self, **changes: float) -> "MolecularAnalysisPolicy":
        return replace(self, **changes)


class MolecularComponentKind(StrEnum):
    DISCRETE_MOLECULE = "discrete_molecule"
    PERIODIC_CHAIN = "periodic_chain"
    PERIODIC_LAYER = "periodic_layer"
    PERIODIC_FRAMEWORK = "periodic_framework"


_KIND_BY_RANK = {
    0: MolecularComponentKind.DISCRETE_MOLECULE,
    1: MolecularComponentKind.PERIODIC_CHAIN,
    2: MolecularComponentKind.PERIODIC_LAYER,
    3: MolecularComponentKind.PERIODIC_FRAMEWORK,
}


@dataclass(frozen=True, slots=True)
class MolecularBondOrbit:
    bond_orbit_id: str
    resolved_contact_orbit_id: str
    geometry_orbit_id: str
    interpretation_id: str
    first_independent_site_id: str
    second_independent_site_id: str
    periodic_relation: PeriodicSymmetryRelation
    accepted_component_pairs: tuple[ComponentPairInterpretation, ...]
    representative_distance: float
    multiplicity_in_reference_cell: int
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.bond_orbit_id,
                self.resolved_contact_orbit_id,
                self.geometry_orbit_id,
                self.interpretation_id,
                self.first_independent_site_id,
                self.second_independent_site_id,
            )
        ):
            raise ValueError("molecular bond identities must not be empty")
        if not self.accepted_component_pairs:
            raise ValueError("molecular bond requires an accepted component pair")
        if not math.isfinite(self.representative_distance) or self.representative_distance <= 0:
            raise ValueError("molecular bond distance must be finite and positive")
        if self.multiplicity_in_reference_cell <= 0:
            raise ValueError("molecular bond multiplicity must be positive")


@dataclass(frozen=True, slots=True)
class MolecularComponentOrbit:
    component_orbit_id: str
    independent_site_ids: tuple[str, ...]
    bond_orbit_ids: tuple[str, ...]
    image_relations: tuple[tuple[str, PeriodicSymmetryRelation], ...]
    closure_relations: tuple[PeriodicSymmetryRelation, ...]
    periodic_rank: int
    periodic_generators: tuple[Translation, ...]
    multiplicity_in_reference_cell: int
    kind: MolecularComponentKind
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.component_orbit_id or not self.independent_site_ids:
            raise ValueError("molecular component requires identity and sites")
        if self.independent_site_ids != tuple(sorted(set(self.independent_site_ids))):
            raise ValueError("molecular component site IDs must be unique and sorted")
        if self.bond_orbit_ids != tuple(sorted(set(self.bond_orbit_ids))):
            raise ValueError("molecular component bond IDs must be unique and sorted")
        if tuple(site_id for site_id, _ in self.image_relations) != self.independent_site_ids:
            raise ValueError("molecular component requires one relation per site")
        if self.periodic_rank not in range(4):
            raise ValueError("molecular component rank must be between zero and three")
        if self.periodic_rank != len(self.periodic_generators):
            raise ValueError("molecular component rank and generators disagree")
        if self.kind is not _KIND_BY_RANK[self.periodic_rank]:
            raise ValueError("molecular component kind and periodic rank disagree")
        if self.multiplicity_in_reference_cell <= 0:
            raise ValueError("molecular component multiplicity must be positive")


@dataclass(frozen=True, slots=True)
class MolecularAnalysisResult:
    bond_orbits: tuple[MolecularBondOrbit, ...]
    component_orbits: tuple[MolecularComponentOrbit, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        for values, name, attribute in (
            (self.bond_orbits, "bond", "bond_orbit_id"),
            (self.component_orbits, "component", "component_orbit_id"),
        ):
            identifiers = tuple(getattr(item, attribute) for item in values)
            if identifiers != tuple(sorted(set(identifiers))):
                raise ValueError(f"molecular {name} orbit IDs must be unique and sorted")
        known_bonds = {item.bond_orbit_id for item in self.bond_orbits}
        if any(
            not set(component.bond_orbit_ids) <= known_bonds
            for component in self.component_orbits
        ):
            raise ValueError("molecular component references an unknown bond orbit")

    @property
    def molecule_orbits(self) -> tuple[MolecularComponentOrbit, ...]:
        return tuple(
            item
            for item in self.component_orbits
            if item.kind is MolecularComponentKind.DISCRETE_MOLECULE
        )


def _negative(vector: Translation) -> Translation:
    return tuple(-value for value in vector)  # type: ignore[return-value]


def _canonical_vector(vector: Translation) -> Translation:
    for value in vector:
        if value < 0:
            return _negative(vector)
        if value > 0:
            return vector
    return vector


def _subgroup_invariants(relations, context) -> tuple[int, tuple[Translation, ...]]:
    if not relations:
        return 1, ()
    generators = tuple(sorted(set(
        (*relations, *(item.inverse(context) for item in relations))
    )))
    identity = identity_relation(context)
    representatives = {identity.operation_key: identity}
    queue = deque((identity.operation_key,))
    translations: set[Translation] = set()
    while queue:
        representative = representatives[queue.popleft()]
        for generator in generators:
            candidate = representative.compose(generator, context)
            existing = representatives.get(candidate.operation_key)
            if existing is None:
                representatives[candidate.operation_key] = candidate
                queue.append(candidate.operation_key)
                continue
            delta = candidate.compose(existing.inverse(context), context)
            if delta.operation_key != context.identity_operation_key:
                raise ValueError("molecular subgroup reduction did not produce a translation")
            if delta.lattice_translation != ZERO:
                translations.add(_canonical_vector(delta.lattice_translation))
    basis = integer_translation_lattice_basis(tuple(sorted(translations)))
    return len(representatives), basis


def _component_status(bonds: tuple[MolecularBondOrbit, ...]) -> ResolutionStatus:
    if any(item.status is ResolutionStatus.INCOMPLETE for item in bonds):
        return ResolutionStatus.INCOMPLETE
    if any(item.status is ResolutionStatus.AMBIGUOUS for item in bonds):
        return ResolutionStatus.AMBIGUOUS
    return ResolutionStatus.RESOLVED


@dataclass(frozen=True, slots=True)
class MolecularAnalyzer:
    policy: MolecularAnalysisPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.policy, MolecularAnalysisPolicy):
            raise TypeError("policy must be MolecularAnalysisPolicy")

    def analyze(self, contact_result: ContactAnalysisResult) -> MolecularAnalysisResult:
        if not isinstance(contact_result, ContactAnalysisResult):
            raise TypeError("contact_result must be ContactAnalysisResult")
        requests = tuple(
            item
            for item in contact_result.grammar.candidate_interactions
            if item.layer is InteractionLayer.INTRAMOLECULAR
            and item.priority is InteractionPriority.PRIMARY
        )
        if not requests:
            return MolecularAnalysisResult(
                (),
                (),
                ResolutionStatus.NOT_APPLICABLE,
                (),
                tuple(sorted(self.policy.get_config().items())),
                (("method", "cristma.molecular_analysis:1"),),
            )

        participating_elements = {
            symbol
            for request in requests
            for symbol in (*request.first_elements, *request.second_elements)
        }
        node_ids = {
            site.id
            for site in contact_result._structure.sites
            if any(
                component.element in participating_elements
                for component in site.components
            )
        }
        geometry_by_id = {
            item.geometry_orbit_id: item
            for item in contact_result.pair_table.contact_orbits
        }
        bonds: list[MolecularBondOrbit] = []
        diagnostics: list[Diagnostic] = []
        for contact in contact_result.contact_orbits:
            geometry = geometry_by_id[contact.geometry_orbit_id]
            for interpretation in contact.interpretations:
                if (
                    interpretation.interaction_layer is not InteractionLayer.INTRAMOLECULAR
                    or interpretation.grammar_priority is not InteractionPriority.PRIMARY
                ):
                    continue
                accepted = tuple(
                    item
                    for item in interpretation.component_pair_interpretations
                    if item.normalized_distance <= self.policy.maximum_bond_rho
                )
                if not accepted:
                    if interpretation.status is ResolutionStatus.INCOMPLETE:
                        diagnostics.extend(contact.diagnostics)
                    continue
                status = (
                    ResolutionStatus.INCOMPLETE
                    if contact_result.pair_table.status is PairTableStatus.INCOMPLETE
                    or contact.status is ResolutionStatus.INCOMPLETE
                    or interpretation.status is ResolutionStatus.INCOMPLETE
                    else ResolutionStatus.AMBIGUOUS
                    if len(accepted) != len(interpretation.component_pair_interpretations)
                    else ResolutionStatus.RESOLVED
                )
                bond_diagnostics = list(contact.diagnostics)
                if status is ResolutionStatus.AMBIGUOUS:
                    diagnostic = Diagnostic(
                        Severity.WARNING,
                        "crystal_chemistry.molecular.mixed_component_bond_ambiguous",
                        "Mixed-position components disagree at the molecular bond boundary",
                    )
                    bond_diagnostics.append(diagnostic)
                    diagnostics.append(diagnostic)
                bond_id = "molecular-bond-orbit:" + _digest({
                    "resolved_contact_orbit_id": contact.resolved_contact_orbit_id,
                    "interpretation_id": interpretation.interpretation_id,
                })
                bonds.append(MolecularBondOrbit(
                    bond_id,
                    contact.resolved_contact_orbit_id,
                    geometry.geometry_orbit_id,
                    interpretation.interpretation_id,
                    geometry.first_independent_site_id,
                    geometry.second_independent_site_id,
                    geometry.canonical_relation,
                    accepted,
                    geometry.representative_distance,
                    geometry.multiplicity_in_reference_cell,
                    status,
                    tuple(dict.fromkeys(bond_diagnostics)),
                    (
                        ("method", "cristma.molecular_bond_resolution:1"),
                        ("maximum_bond_rho", self.policy.maximum_bond_rho),
                    ),
                ))

        ordered_bonds = tuple(sorted(bonds, key=lambda item: item.bond_orbit_id))
        adjacency: dict[
            str,
            list[tuple[str, PeriodicSymmetryRelation, str]],
        ] = {item: [] for item in node_ids}
        for bond in ordered_bonds:
            if bond.first_independent_site_id not in node_ids or bond.second_independent_site_id not in node_ids:
                raise ValueError("molecular bond endpoint is outside the molecular site set")
            adjacency[bond.first_independent_site_id].append((
                bond.second_independent_site_id,
                bond.periodic_relation,
                bond.bond_orbit_id,
            ))
            adjacency[bond.second_independent_site_id].append((
                bond.first_independent_site_id,
                bond.periodic_relation.inverse(contact_result._symmetry_context),
                bond.bond_orbit_id,
            ))
        for values in adjacency.values():
            values.sort(key=lambda item: (item[2], item[0], item[1]))

        bond_by_id = {item.bond_orbit_id: item for item in ordered_bonds}
        components: list[MolecularComponentOrbit] = []
        remaining = set(node_ids)
        context = contact_result._symmetry_context
        while remaining:
            root = min(remaining)
            images = {root: identity_relation(context)}
            used_bonds: set[str] = set()
            closures: set[PeriodicSymmetryRelation] = set()
            queue = deque((root,))
            while queue:
                current = queue.popleft()
                for neighbor, relation, bond_id in adjacency[current]:
                    used_bonds.add(bond_id)
                    proposed = images[current].compose(relation, context)
                    known = images.get(neighbor)
                    if known is None:
                        images[neighbor] = proposed
                        queue.append(neighbor)
                    else:
                        closure = proposed.compose(known.inverse(context), context)
                        if closure != identity_relation(context):
                            closures.add(closure)
            remaining -= set(images)
            for site_id, path in images.items():
                for stabilizer in contact_result._asymmetric_unit_mapping.by_site_id[
                    site_id
                ].stabilizer_relations:
                    conjugate = path.compose(stabilizer, context).compose(
                        path.inverse(context), context
                    )
                    if conjugate != identity_relation(context):
                        closures.add(conjugate)
            subgroup_order, basis = _subgroup_invariants(tuple(sorted(closures)), context)
            group_order = len(context.operations)
            if group_order % subgroup_order:
                raise ValueError("molecular component stabilizer does not divide group order")
            component_bonds = tuple(bond_by_id[item] for item in sorted(used_bonds))
            status = (
                ResolutionStatus.INCOMPLETE
                if contact_result.pair_table.status is PairTableStatus.INCOMPLETE
                else _component_status(component_bonds)
            )
            site_ids = tuple(sorted(images))
            bond_ids = tuple(sorted(used_bonds))
            component_id = "molecular-component-orbit:" + _digest({
                "site_ids": site_ids,
                "bond_orbit_ids": bond_ids,
            })
            components.append(MolecularComponentOrbit(
                component_id,
                site_ids,
                bond_ids,
                tuple(sorted(images.items())),
                tuple(sorted(closures)),
                len(basis),
                basis,
                group_order // subgroup_order,
                _KIND_BY_RANK[len(basis)],
                status,
                tuple(dict.fromkeys(
                    diagnostic
                    for bond in component_bonds
                    for diagnostic in bond.diagnostics
                )),
                (
                    ("method", "cristma.molecular_component_orbits:1"),
                    ("stabilizer_operation_order", subgroup_order),
                ),
            ))

        ordered_components = tuple(
            sorted(components, key=lambda item: item.component_orbit_id)
        )
        statuses = tuple(item.status for item in ordered_components)
        result_status = (
            ResolutionStatus.INCOMPLETE
            if contact_result.pair_table.status is PairTableStatus.INCOMPLETE
            or ResolutionStatus.INCOMPLETE in statuses
            else ResolutionStatus.AMBIGUOUS
            if ResolutionStatus.AMBIGUOUS in statuses
            else ResolutionStatus.RESOLVED
        )
        return MolecularAnalysisResult(
            ordered_bonds,
            ordered_components,
            result_status,
            tuple(dict.fromkeys((*contact_result.pair_table.diagnostics, *diagnostics))),
            tuple(sorted(self.policy.get_config().items())),
            (
                ("method", "cristma.molecular_analysis:1"),
                ("symmetry_context_fingerprint", context.fingerprint),
                ("contact_analysis_method", "cristma.contact_analysis:1"),
            ),
        )


__all__ = [
    "MolecularAnalysisPolicy",
    "MolecularAnalysisResult",
    "MolecularAnalyzer",
    "MolecularBondOrbit",
    "MolecularComponentKind",
    "MolecularComponentOrbit",
]
