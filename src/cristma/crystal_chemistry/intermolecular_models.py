"""Immutable value types for orbit-first intermolecular-contact analysis."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING

from cristma.chemistry.elements import normalize_element
from cristma.crystallography import PeriodicSymmetryRelation
from cristma.diagnostics import Diagnostic
from cristma.structure import PeriodicAtomRef

from .models import ResolutionStatus

if TYPE_CHECKING:
    from .contact_analysis import ContactAnalysisResult
    from .molecular import MolecularAnalysisResult


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _element_set(values: object, name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a collection of element symbols")
    try:
        normalized = frozenset(normalize_element(value) for value in values)  # type: ignore[union-attr]
    except TypeError as error:
        raise TypeError(f"{name} must be a collection of element symbols") from error
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class IntermolecularContactPolicy:
    """Explicit geometric policy for the first intermolecular-contact slice."""

    maximum_distance_angstrom: float = 4.0
    maximum_vdw_gap_angstrom: float = 0.0
    hydrogen_acceptor_gap_angstrom: float = -0.12
    minimum_donor_hydrogen_acceptor_angle_deg: float = 120.0
    hydrogen_donor_elements: frozenset[str] = frozenset({"N", "O", "S"})
    hydrogen_acceptor_elements: frozenset[str] = frozenset(
        {"N", "O", "F", "S", "Cl", "Br", "I"}
    )
    require_explicit_hydrogen: bool = True

    def __post_init__(self) -> None:
        maximum_distance = _finite_number(
            self.maximum_distance_angstrom,
            "maximum_distance_angstrom",
        )
        maximum_gap = _finite_number(
            self.maximum_vdw_gap_angstrom,
            "maximum_vdw_gap_angstrom",
        )
        hydrogen_gap = _finite_number(
            self.hydrogen_acceptor_gap_angstrom,
            "hydrogen_acceptor_gap_angstrom",
        )
        minimum_angle = _finite_number(
            self.minimum_donor_hydrogen_acceptor_angle_deg,
            "minimum_donor_hydrogen_acceptor_angle_deg",
        )
        if maximum_distance <= 0.0:
            raise ValueError("maximum_distance_angstrom must be positive")
        if not 0.0 <= minimum_angle <= 180.0:
            raise ValueError(
                "minimum_donor_hydrogen_acceptor_angle_deg must lie between 0 and 180"
            )
        if not isinstance(self.require_explicit_hydrogen, bool):
            raise TypeError("require_explicit_hydrogen must be bool")
        if not self.require_explicit_hydrogen:
            raise ValueError("intermolecular v1 requires explicit hydrogen atoms")
        object.__setattr__(self, "maximum_distance_angstrom", maximum_distance)
        object.__setattr__(self, "maximum_vdw_gap_angstrom", maximum_gap)
        object.__setattr__(self, "hydrogen_acceptor_gap_angstrom", hydrogen_gap)
        object.__setattr__(
            self,
            "minimum_donor_hydrogen_acceptor_angle_deg",
            minimum_angle,
        )
        object.__setattr__(
            self,
            "hydrogen_donor_elements",
            _element_set(self.hydrogen_donor_elements, "hydrogen_donor_elements"),
        )
        object.__setattr__(
            self,
            "hydrogen_acceptor_elements",
            _element_set(self.hydrogen_acceptor_elements, "hydrogen_acceptor_elements"),
        )

    def get_config(self) -> tuple[tuple[str, object], ...]:
        return (
            ("maximum_distance_angstrom", self.maximum_distance_angstrom),
            ("maximum_vdw_gap_angstrom", self.maximum_vdw_gap_angstrom),
            ("hydrogen_acceptor_gap_angstrom", self.hydrogen_acceptor_gap_angstrom),
            (
                "minimum_donor_hydrogen_acceptor_angle_deg",
                self.minimum_donor_hydrogen_acceptor_angle_deg,
            ),
            ("hydrogen_donor_elements", tuple(sorted(self.hydrogen_donor_elements))),
            ("hydrogen_acceptor_elements", tuple(sorted(self.hydrogen_acceptor_elements))),
            ("require_explicit_hydrogen", self.require_explicit_hydrogen),
        )

    def clone(self, **changes: object) -> "IntermolecularContactPolicy":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class VanDerWaalsDistanceEvidence:
    """Component-pair distance normalized by neutral-atom vdW radii."""

    first_element: str
    second_element: str
    first_occupancy: float
    second_occupancy: float
    radius_sum: float | None
    vdw_gap: float | None
    normalized_distance: float | None
    status: ResolutionStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_element", normalize_element(self.first_element))
        object.__setattr__(self, "second_element", normalize_element(self.second_element))
        for name in ("first_occupancy", "second_occupancy"):
            value = _finite_number(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError("component occupancies must be positive")
            object.__setattr__(self, name, value)
        metrics = (self.radius_sum, self.vdw_gap, self.normalized_distance)
        if all(value is None for value in metrics):
            if self.status is not ResolutionStatus.INCOMPLETE:
                raise ValueError("missing radius evidence must be incomplete")
            return
        if any(value is None for value in metrics):
            raise ValueError("complete radius evidence requires all metrics")
        radius_sum = _finite_number(self.radius_sum, "radius_sum")
        vdw_gap = _finite_number(self.vdw_gap, "vdw_gap")
        normalized_distance = _finite_number(
            self.normalized_distance,
            "normalized_distance",
        )
        if radius_sum <= 0.0 or normalized_distance <= 0.0:
            raise ValueError("radius sum and normalized distance must be positive")
        object.__setattr__(self, "radius_sum", radius_sum)
        object.__setattr__(self, "vdw_gap", vdw_gap)
        object.__setattr__(self, "normalized_distance", normalized_distance)


@dataclass(frozen=True, slots=True)
class HydrogenBondCandidate:
    """Geometrically supported D-H...A interpretation using an explicit H atom."""

    interpretation_id: str
    donor_independent_site_id: str
    hydrogen_independent_site_id: str
    acceptor_independent_site_id: str
    donor_hydrogen_bond_orbit_id: str
    hydrogen_acceptor_distance: float
    donor_acceptor_distance: float
    donor_hydrogen_acceptor_angle_deg: float
    donor_hydrogen_relation: PeriodicSymmetryRelation
    hydrogen_acceptor_relation: PeriodicSymmetryRelation
    accepted_donor_acceptor_pairs: tuple[tuple[str, str], ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.interpretation_id,
            self.donor_independent_site_id,
            self.hydrogen_independent_site_id,
            self.acceptor_independent_site_id,
            self.donor_hydrogen_bond_orbit_id,
        )
        if not all(isinstance(value, str) and value for value in identities):
            raise ValueError("hydrogen-bond candidate identities must not be empty")
        for name in ("hydrogen_acceptor_distance", "donor_acceptor_distance"):
            value = _finite_number(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError("hydrogen-bond distances must be positive")
            object.__setattr__(self, name, value)
        angle = _finite_number(
            self.donor_hydrogen_acceptor_angle_deg,
            "donor_hydrogen_acceptor_angle_deg",
        )
        if not 0.0 <= angle <= 180.0:
            raise ValueError("donor-hydrogen-acceptor angle must lie between 0 and 180")
        object.__setattr__(self, "donor_hydrogen_acceptor_angle_deg", angle)
        pairs = tuple(
            (normalize_element(donor), normalize_element(acceptor))
            for donor, acceptor in self.accepted_donor_acceptor_pairs
        )
        if not pairs:
            raise ValueError("hydrogen-bond candidate requires a donor-acceptor pair")
        if pairs != tuple(sorted(set(pairs))):
            raise ValueError("donor-acceptor pairs must be unique and sorted")
        object.__setattr__(self, "accepted_donor_acceptor_pairs", pairs)


def _evidence_key(evidence: VanDerWaalsDistanceEvidence) -> tuple[object, ...]:
    return (
        evidence.first_element,
        evidence.second_element,
        evidence.first_occupancy,
        evidence.second_occupancy,
    )


def _validated_range(
    observed: tuple[float, float] | None,
    values: tuple[float, ...],
    name: str,
) -> tuple[float, float] | None:
    if not values:
        if observed is not None:
            raise ValueError(f"{name} cannot be reported without complete evidence")
        return None
    if observed is None or len(observed) != 2:
        raise ValueError(f"{name} is required for complete evidence")
    normalized = tuple(_finite_number(value, name) for value in observed)
    expected = (min(values), max(values))
    if normalized[0] > normalized[1] or not all(
        math.isclose(actual, wanted, abs_tol=1e-12)
        for actual, wanted in zip(normalized, expected, strict=True)
    ):
        raise ValueError(f"{name} does not cover component evidence")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class IntermolecularContactOrbit:
    """One symmetry-equivalent geometric contact between discrete molecules."""

    intermolecular_contact_orbit_id: str
    geometry_orbit_id: str
    first_molecule_orbit_id: str
    second_molecule_orbit_id: str
    canonical_molecule_relation: PeriodicSymmetryRelation
    first_independent_site_id: str
    second_independent_site_id: str
    representative_distance: float
    component_pair_evidence: tuple[VanDerWaalsDistanceEvidence, ...]
    vdw_gap_range: tuple[float, float] | None
    normalized_vdw_distance_range: tuple[float, float] | None
    multiplicity_in_reference_cell: int
    interpretations: tuple[HydrogenBondCandidate, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.intermolecular_contact_orbit_id,
            self.geometry_orbit_id,
            self.first_molecule_orbit_id,
            self.second_molecule_orbit_id,
            self.first_independent_site_id,
            self.second_independent_site_id,
        )
        if not all(isinstance(value, str) and value for value in identities):
            raise ValueError("intermolecular contact-orbit identities must not be empty")
        distance = _finite_number(self.representative_distance, "representative_distance")
        if distance <= 0.0:
            raise ValueError("representative distance must be positive")
        object.__setattr__(self, "representative_distance", distance)
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ):
            raise TypeError("intermolecular contact multiplicity must be an integer")
        if self.multiplicity_in_reference_cell <= 0:
            raise ValueError("intermolecular contact multiplicity must be positive")
        evidence_keys = tuple(_evidence_key(item) for item in self.component_pair_evidence)
        if evidence_keys != tuple(sorted(set(evidence_keys))):
            raise ValueError("component-pair evidence must be unique and sorted")
        interpretation_ids = tuple(item.interpretation_id for item in self.interpretations)
        if interpretation_ids != tuple(sorted(set(interpretation_ids))):
            raise ValueError("intermolecular interpretations must be unique and sorted")
        complete = tuple(
            item for item in self.component_pair_evidence if item.radius_sum is not None
        )
        object.__setattr__(
            self,
            "vdw_gap_range",
            _validated_range(
                self.vdw_gap_range,
                tuple(item.vdw_gap for item in complete),  # type: ignore[misc]
                "vdw_gap_range",
            ),
        )
        object.__setattr__(
            self,
            "normalized_vdw_distance_range",
            _validated_range(
                self.normalized_vdw_distance_range,
                tuple(item.normalized_distance for item in complete),  # type: ignore[misc]
                "normalized_vdw_distance_range",
            ),
        )
        has_incomplete_evidence = any(
            item.status is ResolutionStatus.INCOMPLETE
            for item in self.component_pair_evidence
        )
        if has_incomplete_evidence and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete vdW evidence requires an incomplete contact orbit")


@dataclass(frozen=True, slots=True)
class MaterializedIntermolecularContact:
    """One periodic contact instance produced only at the consumer boundary."""

    contact_id: str
    intermolecular_contact_orbit_id: str
    first_molecule_orbit_id: str
    second_molecule_orbit_id: str
    first_atom_ref: PeriodicAtomRef
    second_atom_ref: PeriodicAtomRef
    distance: float
    vector_cartesian: tuple[float, float, float]
    interpretations: tuple[HydrogenBondCandidate, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.contact_id,
            self.intermolecular_contact_orbit_id,
            self.first_molecule_orbit_id,
            self.second_molecule_orbit_id,
        )
        if not all(isinstance(value, str) and value for value in identities):
            raise ValueError("materialized contact identities must not be empty")
        distance = _finite_number(self.distance, "distance")
        if distance <= 0.0:
            raise ValueError("materialized contact distance must be positive")
        object.__setattr__(self, "distance", distance)
        if len(self.vector_cartesian) != 3:
            raise ValueError("materialized contact vector must contain three values")
        vector = tuple(_finite_number(value, "vector_cartesian") for value in self.vector_cartesian)
        object.__setattr__(self, "vector_cartesian", vector)
        interpretation_ids = tuple(item.interpretation_id for item in self.interpretations)
        if interpretation_ids != tuple(sorted(set(interpretation_ids))):
            raise ValueError("materialized contact interpretations must be unique and sorted")


@dataclass(frozen=True, slots=True)
class IntermolecularContactAnalysisResult:
    """Primary immutable result for contacts between discrete molecule orbits."""

    _contact_analysis: "ContactAnalysisResult" = field(repr=False, compare=False)
    _molecular_analysis: "MolecularAnalysisResult" = field(repr=False, compare=False)
    contact_orbits: tuple[IntermolecularContactOrbit, ...]
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...]
    configuration: tuple[tuple[str, object], ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        identifiers = tuple(
            item.intermolecular_contact_orbit_id for item in self.contact_orbits
        )
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("intermolecular contact-orbit IDs must be unique and sorted")
        if any(
            item.status is ResolutionStatus.INCOMPLETE
            for item in self.contact_orbits
        ) and self.status is not ResolutionStatus.INCOMPLETE:
            raise ValueError("incomplete contact orbit must propagate to result status")
        configuration_keys = tuple(key for key, _ in self.configuration)
        if configuration_keys != tuple(sorted(set(configuration_keys))):
            raise ValueError("intermolecular configuration keys must be unique and sorted")
        provenance_keys = tuple(key for key, _ in self.provenance)
        if provenance_keys != tuple(sorted(set(provenance_keys))):
            raise ValueError("intermolecular provenance keys must be unique and sorted")

    def materialize_contacts(self, region, **filters):
        """Project contact orbits outward into a requested periodic region."""

        from .materialization import IntermolecularContactMaterializer

        return IntermolecularContactMaterializer().materialize(self, region, **filters)


__all__ = [
    "HydrogenBondCandidate",
    "IntermolecularContactAnalysisResult",
    "IntermolecularContactOrbit",
    "IntermolecularContactPolicy",
    "MaterializedIntermolecularContact",
    "VanDerWaalsDistanceEvidence",
]
