"""Top-level orchestration for orbit-first direct-space contact analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from cristma.chemistry import CompositionGrammar
from cristma.crystallography import (
    AsymmetricUnitMapper,
    AsymmetricUnitMapping,
    PairTableStatus,
    SymmetryContext,
    SymmetryPairFinder,
    SymmetryPairTable,
)
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic
from cristma.reference_data import ReferenceData
from cristma.structure import CrystalStructure

from .models import ResolutionStatus
from .incidence_orbits import ContactIncidenceBuilder, ContactIncidenceOrbit
from .orbit_contacts import ContactOrbitResolver, ResolvedContactOrbit
from .policy import ShellResolutionPolicy
from .shell_orbits import (
    CoordinationShellOrbit,
    CoordinationShellOrbitResolver,
    _interpretation_context_id,
)


def aggregate_contact_analysis_status(
    pair_status: PairTableStatus,
    shell_statuses: tuple[ResolutionStatus, ...],
) -> ResolutionStatus:
    if pair_status is PairTableStatus.INCOMPLETE:
        return ResolutionStatus.INCOMPLETE
    if ResolutionStatus.INCOMPLETE in shell_statuses:
        return ResolutionStatus.INCOMPLETE
    if ResolutionStatus.AMBIGUOUS in shell_statuses:
        return ResolutionStatus.AMBIGUOUS
    if shell_statuses:
        return ResolutionStatus.RESOLVED
    return ResolutionStatus.NOT_APPLICABLE


@dataclass(frozen=True, slots=True)
class ContactAnalysisResult:
    _structure: CrystalStructure = field(repr=False, compare=False)
    _asymmetric_unit_mapping: AsymmetricUnitMapping = field(repr=False, compare=False)
    _symmetry_context: SymmetryContext = field(repr=False, compare=False)
    pair_table: SymmetryPairTable
    grammar: CompositionGrammar
    contact_orbits: tuple[ResolvedContactOrbit, ...]
    contact_incidence_orbits: tuple[ContactIncidenceOrbit, ...]
    coordination_shell_orbits: tuple[CoordinationShellOrbit, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if self.pair_table.asymmetric_unit_mapping_fingerprint != self._asymmetric_unit_mapping.fingerprint:
            raise ValueError("contact result and asymmetric-unit mapping disagree")
        if self.pair_table.symmetry_context_fingerprint != self._symmetry_context.fingerprint:
            raise ValueError("contact result and symmetry context disagree")
        if not isinstance(self.grammar, CompositionGrammar):
            raise TypeError("contact result grammar must be CompositionGrammar")
        if {site.id for site in self._structure.sites} != set(self._asymmetric_unit_mapping.by_site_id):
            raise ValueError("contact result mapping belongs to another structure")
        contact_by_id = {
            item.resolved_contact_orbit_id: item for item in self.contact_orbits
        }
        interpretation_by_id = {
            item.interpretation_id: item
            for orbit in self.contact_orbits
            for item in orbit.interpretations
        }
        incidence_by_id = {
            item.incidence_orbit_id: item for item in self.contact_incidence_orbits
        }
        if len(contact_by_id) != len(self.contact_orbits):
            raise ValueError("contact-analysis result contains duplicate contact-orbit IDs")
        if len(incidence_by_id) != len(self.contact_incidence_orbits):
            raise ValueError("contact-analysis result contains duplicate incidence-orbit IDs")
        for incidence in self.contact_incidence_orbits:
            if incidence.resolved_contact_orbit_id not in contact_by_id:
                raise ValueError("incidence references an unknown contact orbit")
            if incidence.interpretation_id not in interpretation_by_id:
                raise ValueError("incidence references an unknown interpretation")
        if self.pair_table.status is PairTableStatus.INCOMPLETE and any(
            shell.status is ResolutionStatus.RESOLVED
            for shell in self.coordination_shell_orbits
        ):
            raise ValueError("an incomplete pair table cannot produce a resolved shell")
        for shell in self.coordination_shell_orbits:
            eligible = {
                incidence.incidence_orbit_id
                for incidence in self.contact_incidence_orbits
                if incidence.center_independent_site_id == shell.center_independent_site_id
                and _interpretation_context_id(
                    interpretation_by_id[incidence.interpretation_id]
                ) == shell.interpretation_context_id
            }
            for alternative in shell.alternatives:
                reported = set(alternative.primary_incidence_ids) | set(
                    alternative.secondary_incidence_ids
                )
                if reported != eligible:
                    raise ValueError("shell alternative does not partition its incidence orbits")
                geometric_cn = sum(
                    incidence_by_id[item].incidence_multiplicity_per_center
                    for item in alternative.primary_incidence_ids
                )
                occupied = sum(
                    incidence_by_id[item].incidence_multiplicity_per_center
                    * incidence_by_id[item].effective_neighbor_occupancy
                    for item in alternative.primary_incidence_ids
                )
                if alternative.geometric_CN != geometric_cn:
                    raise ValueError("shell alternative geometric CN disagrees with incidence weights")
                if abs(alternative.mean_occupied_neighbors - occupied) > 1e-12:
                    raise ValueError("shell alternative occupancy disagrees with incidence weights")
        expected = aggregate_contact_analysis_status(
            self.pair_table.status,
            tuple(item.status for item in self.coordination_shell_orbits),
        )
        if self.status is not expected:
            raise ValueError("contact-analysis status does not match dependent results")

    def materialize_contacts(self, region, **filters):
        """Project immutable contact orbits outward into a requested periodic region."""

        from .materialization import ContactMaterializer

        return ContactMaterializer().materialize(self, region, **filters)

    @property
    def fingerprint(self) -> str:
        """Return deterministic lineage identity for dependent analyses."""

        return _digest({
            "structure_id": self._structure.id,
            "site_ids": tuple(site.id for site in self._structure.sites),
            "symmetry_context_fingerprint": self._symmetry_context.fingerprint,
            "mapping_fingerprint": self._asymmetric_unit_mapping.fingerprint,
            "geometry_orbit_ids": tuple(
                item.geometry_orbit_id for item in self.pair_table.contact_orbits
            ),
            "resolved_contact_orbit_ids": tuple(
                item.resolved_contact_orbit_id for item in self.contact_orbits
            ),
            "configuration": self.configuration,
            "grammar_method": (self.grammar.method_id, self.grammar.method_version),
        })

    @property
    def contacts(self):
        """Temporary read-only reference-cell compatibility view."""

        from .materialization import ReferenceCell

        return self.materialize_contacts(ReferenceCell())


def _search_cutoff(
    grammar: CompositionGrammar,
    reference: ReferenceData,
    policy: ShellResolutionPolicy,
) -> float:
    radius_sums: list[float] = []
    for request in grammar.candidate_interactions:
        for first in request.first_elements:
            for second in request.second_elements:
                try:
                    radius_sums.append(
                        reference.covalent_radii.find(first).value
                        + reference.covalent_radii.find(second).value
                    )
                except KeyError:
                    continue
    if not radius_sums:
        raise ValueError("grammar has no component pairs with known covalent radii")
    return max(radius_sums) * policy.search_rho_max


class ContactAnalyzer:
    """Run the orbit-first contact pipeline from an explicit symmetry context."""

    def __init__(
        self,
        policy: ShellResolutionPolicy,
        reference: ReferenceData | None = None,
        *,
        distance_tolerance: float = 1e-12,
        max_candidates: int | None = None,
        minimum_pair_cutoff: float | None = None,
    ) -> None:
        if not isinstance(policy, ShellResolutionPolicy):
            raise TypeError("policy must be ShellResolutionPolicy")
        self.policy = policy
        self.reference = reference or ReferenceData.default()
        self.distance_tolerance = distance_tolerance
        self.max_candidates = max_candidates
        if minimum_pair_cutoff is not None:
            if isinstance(minimum_pair_cutoff, bool) or not isinstance(
                minimum_pair_cutoff, (int, float)
            ):
                raise TypeError("minimum_pair_cutoff must be a real number or None")
            minimum_pair_cutoff = float(minimum_pair_cutoff)
            if not math.isfinite(minimum_pair_cutoff) or minimum_pair_cutoff <= 0.0:
                raise ValueError("minimum_pair_cutoff must be finite and positive")
        self.minimum_pair_cutoff = minimum_pair_cutoff

    def analyze(
        self,
        structure: CrystalStructure,
        symmetry_context: SymmetryContext,
        grammar: CompositionGrammar,
    ) -> ContactAnalysisResult:
        mapping = AsymmetricUnitMapper().build(structure, symmetry_context)
        if grammar.candidate_interactions or self.minimum_pair_cutoff is not None:
            cutoff = (
                _search_cutoff(grammar, self.reference, self.policy)
                if grammar.candidate_interactions
                else 0.0
            )
            cutoff = max(cutoff, self.minimum_pair_cutoff or 0.0)
            pair_table = SymmetryPairFinder(
                cutoff,
                distance_tolerance=self.distance_tolerance,
                max_candidates=self.max_candidates,
            ).find(structure, symmetry_context, mapping)
        else:
            pair_table = SymmetryPairTable(
                (),
                symmetry_context.fingerprint,
                mapping.fingerprint,
                0.0,
                self.distance_tolerance,
                PairTableStatus.COMPLETE,
                grammar.diagnostics,
                (("method", "cristma.symmetry_pair_search:not_applicable"),),
            )
        contact_resolution = ContactOrbitResolver(self.policy, self.reference).resolve(
            pair_table, structure, grammar,
        )
        incidences = ContactIncidenceBuilder().build(
            pair_table,
            contact_resolution.contact_orbits,
            structure,
            mapping,
            symmetry_context,
        )
        shells = CoordinationShellOrbitResolver(self.policy).resolve(
            pair_table,
            contact_resolution.contact_orbits,
            incidences,
        )
        diagnostics = tuple(dict.fromkeys(
            pair_table.diagnostics
            + contact_resolution.diagnostics
            + tuple(item for shell in shells for item in shell.diagnostics)
        ))
        status = aggregate_contact_analysis_status(
            pair_table.status,
            tuple(item.status for item in shells),
        )
        return ContactAnalysisResult(
            structure,
            mapping,
            symmetry_context,
            pair_table,
            grammar,
            contact_resolution.contact_orbits,
            incidences,
            shells,
            status,
            diagnostics,
            tuple(sorted({
                **self.policy.get_config(),
                "distance_tolerance": self.distance_tolerance,
                "max_candidates": self.max_candidates,
                "minimum_pair_cutoff": self.minimum_pair_cutoff,
            }.items())),
            (
                ("method", "cristma.contact_analysis:1"),
                ("symmetry_context_fingerprint", symmetry_context.fingerprint),
                ("grammar_method", f"{grammar.method_id}:{grammar.method_version}"),
                ("reference_version", grammar.reference_version),
            ),
        )


__all__ = [
    "ContactAnalysisResult",
    "ContactAnalyzer",
    "aggregate_contact_analysis_status",
]
