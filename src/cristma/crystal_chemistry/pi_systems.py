"""Geometry-qualified pi-system candidates derived from molecular ring orbits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from numbers import Real
from typing import TYPE_CHECKING

from cristma.chemistry.elements import normalize_element
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic, Severity

from .models import ResolutionStatus
from .molecular_rings import MolecularAtomOrbitRef

if TYPE_CHECKING:
    from .contact_analysis import ContactAnalysisResult
    from .molecular import MolecularAnalysisResult


_DEFAULT_PI_ELEMENTS = frozenset(
    {"B", "C", "N", "O", "P", "S", "Si", "Ge", "As", "Se"}
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class PiSystemPolicy:
    """Explicit geometric and compositional screening policy."""

    maximum_planarity_rms_angstrom: float = 0.12
    eligible_elements: frozenset[str] = _DEFAULT_PI_ELEMENTS

    def __post_init__(self) -> None:
        maximum = _finite(
            self.maximum_planarity_rms_angstrom,
            "maximum planarity RMS",
        )
        if maximum <= 0.0:
            raise ValueError("maximum planarity RMS must be positive")
        object.__setattr__(self, "maximum_planarity_rms_angstrom", maximum)
        if isinstance(self.eligible_elements, (str, bytes)):
            raise TypeError("eligible_elements must be a collection of elements")
        normalized = frozenset(
            normalize_element(item) for item in self.eligible_elements
        )
        if not normalized:
            raise ValueError("eligible_elements must not be empty")
        object.__setattr__(self, "eligible_elements", normalized)

    def get_config(self) -> tuple[tuple[str, object], ...]:
        return (
            ("eligible_elements", tuple(sorted(self.eligible_elements))),
            (
                "maximum_planarity_rms_angstrom",
                self.maximum_planarity_rms_angstrom,
            ),
        )

    def clone(self, **changes: object) -> "PiSystemPolicy":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PiComponentEvidence:
    """Eligibility evidence for one occupied component at a ring position."""

    element: str | None
    occupancy: float
    eligible: bool

    def __post_init__(self) -> None:
        if self.element is not None:
            object.__setattr__(self, "element", normalize_element(self.element))
        occupancy = _finite(self.occupancy, "component occupancy")
        if occupancy <= 0.0:
            raise ValueError("component occupancy must be positive")
        object.__setattr__(self, "occupancy", occupancy)
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be bool")


@dataclass(frozen=True, slots=True)
class PiSiteEvidence:
    """Complete component evidence for one independent ring site."""

    independent_site_id: str
    components: tuple[PiComponentEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.independent_site_id, str) or not self.independent_site_id:
            raise ValueError("pi-site evidence requires an independent-site ID")
        if not self.components:
            raise ValueError("pi-site evidence requires component evidence")
        ordering = tuple(
            sorted(
                set(self.components),
                key=lambda item: (item.element or "", item.occupancy, item.eligible),
            )
        )
        if self.components != ordering:
            raise ValueError("pi component evidence must be unique and sorted")


@dataclass(frozen=True, slots=True)
class PiSystemCandidate:
    """One geometry-qualified molecular-ring pi-system candidate."""

    pi_system_candidate_id: str
    parent_molecule_orbit_id: str
    source_ring_orbit_id: str
    atom_refs: tuple[MolecularAtomOrbitRef, ...]
    site_evidence: tuple[PiSiteEvidence, ...]
    multiplicity_in_reference_cell: int
    centroid_cartesian: tuple[float, float, float]
    normal_cartesian: tuple[float, float, float]
    planarity_rms_angstrom: float
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.pi_system_candidate_id,
            self.parent_molecule_orbit_id,
            self.source_ring_orbit_id,
        )
        if not all(isinstance(item, str) and item for item in identities):
            raise ValueError("pi-system candidate identities must not be empty")
        if len(self.atom_refs) < 3 or len(self.site_evidence) != len(self.atom_refs):
            raise ValueError("pi-system evidence must align with ring atom references")
        if tuple(item.independent_site_id for item in self.site_evidence) != tuple(
            item.independent_site_id for item in self.atom_refs
        ):
            raise ValueError("pi-system site evidence is not aligned with atom refs")
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ) or self.multiplicity_in_reference_cell <= 0:
            raise ValueError("pi-system multiplicity must be a positive integer")
        if len(self.centroid_cartesian) != 3 or not all(
            math.isfinite(item) for item in self.centroid_cartesian
        ):
            raise ValueError("pi-system centroid must contain three finite values")
        if len(self.normal_cartesian) != 3 or not all(
            math.isfinite(item) for item in self.normal_cartesian
        ):
            raise ValueError("pi-system normal must contain three finite values")
        if not math.isclose(
            math.sqrt(math.fsum(item * item for item in self.normal_cartesian)),
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("pi-system normal must be a unit vector")
        rms = _finite(self.planarity_rms_angstrom, "planarity RMS")
        if rms < 0.0:
            raise ValueError("planarity RMS must be non-negative")
        object.__setattr__(self, "planarity_rms_angstrom", rms)


@dataclass(frozen=True, slots=True)
class PiSystemAnalysisResult:
    """Immutable pi-system candidates retaining exact source lineage."""

    _contact_analysis: "ContactAnalysisResult" = field(repr=False, compare=False)
    _molecular_analysis: "MolecularAnalysisResult" = field(repr=False, compare=False)
    candidates: tuple[PiSystemCandidate, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        identifiers = tuple(item.pi_system_candidate_id for item in self.candidates)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("pi-system candidate IDs must be unique and sorted")
        if any(
            item.status is ResolutionStatus.INCOMPLETE for item in self.candidates
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete pi-system candidate must propagate to result")
        for values, name in (
            (self.configuration, "configuration"),
            (self.provenance, "provenance"),
        ):
            keys = tuple(key for key, _ in values)
            if keys != tuple(sorted(set(keys))):
                raise ValueError(f"pi-system {name} keys must be unique and sorted")


@dataclass(frozen=True, slots=True)
class PiSystemAnalyzer:
    policy: PiSystemPolicy = field(default_factory=PiSystemPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PiSystemPolicy):
            raise TypeError("policy must be PiSystemPolicy")

    def analyze(
        self,
        contact_analysis: "ContactAnalysisResult",
        molecular_analysis: "MolecularAnalysisResult",
    ) -> PiSystemAnalysisResult:
        from .contact_analysis import ContactAnalysisResult
        from .molecular import MolecularAnalysisResult

        if not isinstance(contact_analysis, ContactAnalysisResult):
            raise TypeError("contact_analysis must be ContactAnalysisResult")
        if not isinstance(molecular_analysis, MolecularAnalysisResult):
            raise TypeError("molecular_analysis must be MolecularAnalysisResult")
        if molecular_analysis.contact_analysis_fingerprint != contact_analysis.fingerprint:
            raise ValueError("pi-system source lineage disagrees")

        context = contact_analysis._symmetry_context
        provenance = tuple(sorted((
            ("method", "cristma.pi_system_analysis:1"),
            ("symmetry_context_fingerprint", context.fingerprint),
        )))
        if molecular_analysis.status is ResolutionStatus.NOT_APPLICABLE or not (
            molecular_analysis.molecule_orbits
        ):
            diagnostic = Diagnostic(
                Severity.INFO,
                "crystal_chemistry.pi_systems.not_discrete_molecular_domain",
                "Pi-system candidates require a discrete molecular component",
            )
            return PiSystemAnalysisResult(
                contact_analysis,
                molecular_analysis,
                (),
                ResolutionStatus.NOT_APPLICABLE,
                (diagnostic,),
                self.policy.get_config(),
                provenance,
            )

        sites = {site.id: site for site in contact_analysis._structure.sites}
        candidates: list[PiSystemCandidate] = []
        result_diagnostics: list[Diagnostic] = []
        for ring in molecular_analysis.ring_orbits:
            if (
                ring.normal_cartesian is None
                or ring.planarity_rms_angstrom is None
                or ring.planarity_rms_angstrom
                > self.policy.maximum_planarity_rms_angstrom
            ):
                continue
            site_evidence: list[PiSiteEvidence] = []
            has_unknown = False
            has_mixed_eligibility = False
            qualifies = True
            for atom_ref in ring.atom_refs:
                try:
                    site = sites[atom_ref.independent_site_id]
                except KeyError as exc:
                    raise ValueError(
                        "molecular ring references a site outside contact analysis"
                    ) from exc
                components = tuple(sorted(
                    (
                        PiComponentEvidence(
                            component.element,
                            float(component.occupancy.value),
                            component.element in self.policy.eligible_elements,
                        )
                        for component in site.components
                    ),
                    key=lambda item: (
                        item.element or "",
                        item.occupancy,
                        item.eligible,
                    ),
                ))
                site_evidence.append(PiSiteEvidence(site.id, components))
                eligibility = {item.eligible for item in components}
                has_unknown |= any(item.element is None for item in components)
                has_mixed_eligibility |= eligibility == {False, True}
                qualifies &= True in eligibility
            if not qualifies:
                continue

            diagnostics: list[Diagnostic] = []
            status = ResolutionStatus.RESOLVED
            if has_unknown:
                diagnostic = Diagnostic(
                    Severity.WARNING,
                    "crystal_chemistry.pi_systems.component_element_unresolved",
                    "A qualifying ring position also contains an unresolved component",
                )
                diagnostics.append(diagnostic)
                status = ResolutionStatus.INCOMPLETE
            elif has_mixed_eligibility:
                diagnostic = Diagnostic(
                    Severity.WARNING,
                    "crystal_chemistry.pi_systems.mixed_site_eligibility",
                    "Ring-position components disagree on pi-system eligibility",
                )
                diagnostics.append(diagnostic)
                status = ResolutionStatus.AMBIGUOUS
            result_diagnostics.extend(diagnostics)
            candidate_id = "pi-system-candidate:" + _digest({
                "source_ring_orbit_id": ring.ring_orbit_id,
                "symmetry_action": context.symmetry_action_fingerprint,
            })
            candidates.append(PiSystemCandidate(
                candidate_id,
                ring.parent_component_orbit_id,
                ring.ring_orbit_id,
                ring.atom_refs,
                tuple(site_evidence),
                ring.multiplicity_in_reference_cell,
                ring.centroid_cartesian,
                ring.normal_cartesian,
                ring.planarity_rms_angstrom,
                status,
                tuple(diagnostics),
                (
                    ("criterion", "ring_planarity_and_component_eligibility"),
                    ("method", "cristma.pi_system_candidate:1"),
                ),
            ))

        ordered = tuple(sorted(
            candidates,
            key=lambda item: item.pi_system_candidate_id,
        ))
        statuses = tuple(item.status for item in ordered)
        status = (
            ResolutionStatus.INCOMPLETE
            if molecular_analysis.status is ResolutionStatus.INCOMPLETE
            or ResolutionStatus.INCOMPLETE in statuses
            else ResolutionStatus.AMBIGUOUS
            if molecular_analysis.status is ResolutionStatus.AMBIGUOUS
            or ResolutionStatus.AMBIGUOUS in statuses
            else ResolutionStatus.RESOLVED
        )
        return PiSystemAnalysisResult(
            contact_analysis,
            molecular_analysis,
            ordered,
            status,
            tuple(dict.fromkeys((*molecular_analysis.diagnostics, *result_diagnostics))),
            self.policy.get_config(),
            provenance,
        )


__all__ = [
    "PiComponentEvidence",
    "PiSiteEvidence",
    "PiSystemAnalysisResult",
    "PiSystemAnalyzer",
    "PiSystemCandidate",
    "PiSystemPolicy",
]
