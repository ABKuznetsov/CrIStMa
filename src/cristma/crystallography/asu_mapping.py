"""Deterministic asymmetric-unit site images and exact stabilizers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from cristma.structure import CrystalStructure, IndependentSite
from cristma.structure.identity import _expanded_atom_id
from cristma.symmetry.orbit import DEFAULT_FRACTIONAL_TOLERANCE

from .periodic_relation import PeriodicSymmetryRelation
from .symmetry_context import SymmetryContext, _cell_fingerprint, _digest


FractionalPosition = tuple[float, float, float]


class AsymmetricUnitMappingInvariantError(ValueError):
    """A structure cannot be mapped consistently under its symmetry context."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        evidence: tuple[tuple[str, object], ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.evidence = evidence


def _wrap_with_translation(
    raw: FractionalPosition,
    tolerance: float,
) -> tuple[FractionalPosition, tuple[int, int, int]]:
    wrapped: list[float] = []
    translations: list[int] = []
    for value in raw:
        nearest = round(value)
        normalized = float(nearest) if math.isclose(value, nearest, abs_tol=tolerance) else value
        translation = -math.floor(normalized)
        coordinate = normalized + translation
        if math.isclose(coordinate, 1.0, abs_tol=tolerance):
            coordinate = 0.0
            translation -= 1
        if math.isclose(coordinate, 0.0, abs_tol=tolerance):
            coordinate = 0.0
        wrapped.append(round(float(coordinate), 15))
        translations.append(int(translation))
    return tuple(wrapped), tuple(translations)


def _periodically_equal(
    left: FractionalPosition,
    right: FractionalPosition,
    tolerance: float,
) -> bool:
    return all(
        abs((first - second + 0.5) % 1.0 - 0.5) <= tolerance + 1e-12
        for first, second in zip(left, right, strict=True)
    )


def _raw_image(
    site_coordinates: FractionalPosition,
    context: SymmetryContext,
    operation_index: int,
) -> FractionalPosition:
    operation = context.operations[operation_index]
    return tuple(
        math.fsum(
            float(coefficient) * coordinate
            for coefficient, coordinate in zip(row, site_coordinates, strict=True)
        )
        + float(offset)
        for row, offset in zip(operation.rotation, operation.translation, strict=True)
    )


@dataclass(frozen=True, slots=True)
class SiteImage:
    """One unique reference-cell image with complete coset evidence."""

    image_id: str
    representative_relation: PeriodicSymmetryRelation
    equivalent_relations: tuple[PeriodicSymmetryRelation, ...]
    fractional_position: FractionalPosition
    normalization_translation: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("site image ID must not be empty")
        if not self.equivalent_relations:
            raise ValueError("site image must retain at least one relation")
        if tuple(sorted(set(self.equivalent_relations))) != self.equivalent_relations:
            raise ValueError("site image relations must be unique and sorted")
        if self.representative_relation != self.equivalent_relations[0]:
            raise ValueError("site image representative must be its first canonical relation")
        if self.normalization_translation != self.representative_relation.lattice_translation:
            raise ValueError("site image normalization and representative relation disagree")
        if len(self.fractional_position) != 3 or any(
            not math.isfinite(value) or not 0.0 <= value < 1.0
            for value in self.fractional_position
        ):
            raise ValueError("site image must have finite reference-cell coordinates")


@dataclass(frozen=True, slots=True)
class SiteOrbitMapping:
    """Reference-cell orbit and exact stabilizer of one independent site."""

    independent_site_id: str
    stabilizer_relations: tuple[PeriodicSymmetryRelation, ...]
    reference_cell_images: tuple[SiteImage, ...]

    def __post_init__(self) -> None:
        if not self.independent_site_id:
            raise ValueError("independent site ID must not be empty")
        if not self.stabilizer_relations or not self.reference_cell_images:
            raise ValueError("site orbit and stabilizer must not be empty")
        if tuple(sorted(set(self.stabilizer_relations))) != self.stabilizer_relations:
            raise ValueError("stabilizer relations must be unique and sorted")
        image_ids = tuple(image.image_id for image in self.reference_cell_images)
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("site orbit contains duplicate image IDs")


@dataclass(frozen=True, slots=True)
class AsymmetricUnitMapping:
    """Immutable site mappings for one structure and symmetry action."""

    site_orbits: tuple[SiteOrbitMapping, ...]
    symmetry_context_fingerprint: str
    fractional_tolerance: float
    fingerprint: str
    _site_lookup: Mapping[str, SiteOrbitMapping] = field(
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        ids = tuple(item.independent_site_id for item in self.site_orbits)
        if tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise ValueError("site orbit mappings must be unique and sorted by site ID")

    @property
    def by_site_id(self) -> Mapping[str, SiteOrbitMapping]:
        """Return the immutable independent-site lookup."""

        return self._site_lookup


class AsymmetricUnitMapper:
    """Build reference-cell site images from independent coordinates."""

    def __init__(self, fractional_tolerance: float = DEFAULT_FRACTIONAL_TOLERANCE) -> None:
        if isinstance(fractional_tolerance, bool) or not isinstance(
            fractional_tolerance,
            (int, float),
        ):
            raise TypeError("fractional_tolerance must be a real number")
        converted = float(fractional_tolerance)
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError("fractional_tolerance must be positive and finite")
        self.fractional_tolerance = converted

    def _map_site(
        self,
        site: IndependentSite,
        context: SymmetryContext,
        structure_id: str | None,
    ) -> tuple[SiteOrbitMapping, FractionalPosition]:
        reported = tuple(float(value.value) for value in site.fractional)
        source, _ = _wrap_with_translation(reported, self.fractional_tolerance)
        groups: list[tuple[FractionalPosition, list[PeriodicSymmetryRelation]]] = []

        for operation_index, operation_key in enumerate(context.operation_keys):
            raw = _raw_image(source, context, operation_index)
            fractional, translation = _wrap_with_translation(
                raw,
                self.fractional_tolerance,
            )
            relation = PeriodicSymmetryRelation(operation_key, translation)
            matching = next(
                (
                    relations
                    for position, relations in groups
                    if _periodically_equal(position, fractional, self.fractional_tolerance)
                ),
                None,
            )
            if matching is None:
                groups.append((fractional, [relation]))
            else:
                matching.append(relation)

        images: list[SiteImage] = []
        for fractional, raw_relations in groups:
            relations = tuple(sorted(set(raw_relations)))
            representative = relations[0]
            image_id = _expanded_atom_id(
                structure_id,
                site.id,
                fractional,
                self.fractional_tolerance,
            )
            images.append(
                SiteImage(
                    image_id=image_id,
                    representative_relation=representative,
                    equivalent_relations=relations,
                    fractional_position=fractional,
                    normalization_translation=representative.lattice_translation,
                )
            )
        images.sort(key=lambda image: image.representative_relation)

        source_image = next(
            (
                image
                for image in images
                if _periodically_equal(
                    image.fractional_position,
                    source,
                    self.fractional_tolerance,
                )
            ),
            None,
        )
        if source_image is None:
            raise AsymmetricUnitMappingInvariantError(
                "symmetry.asu.representative_missing",
                "site orbit does not contain its independent representative",
                evidence=(("independent_site_id", site.id),),
            )
        stabilizer = source_image.equivalent_relations
        if len(images) * len(stabilizer) != len(context.operations):
            raise AsymmetricUnitMappingInvariantError(
                "symmetry.asu.orbit_stabilizer_inconsistent",
                "site images and stabilizer violate the orbit-stabilizer invariant",
                evidence=(
                    ("independent_site_id", site.id),
                    ("image_count", len(images)),
                    ("stabilizer_order", len(stabilizer)),
                    ("group_order", len(context.operations)),
                ),
            )
        return (
            SiteOrbitMapping(
                independent_site_id=site.id,
                stabilizer_relations=stabilizer,
                reference_cell_images=tuple(images),
            ),
            source,
        )

    def build(
        self,
        structure: CrystalStructure,
        context: SymmetryContext,
    ) -> AsymmetricUnitMapping:
        """Map all independent sites under a validated symmetry context."""

        if not isinstance(structure, CrystalStructure):
            raise TypeError("structure must be CrystalStructure")
        if not isinstance(context, SymmetryContext):
            raise TypeError("context must be SymmetryContext")
        structure_cell_fingerprint = _cell_fingerprint(structure.cell)
        if structure_cell_fingerprint != context.cell_fingerprint:
            raise AsymmetricUnitMappingInvariantError(
                "symmetry.asu.cell_fingerprint_mismatch",
                "structure and symmetry context use different unit cells",
                evidence=(
                    ("structure_cell_fingerprint", structure_cell_fingerprint),
                    ("context_cell_fingerprint", context.cell_fingerprint),
                ),
            )
        site_ids = tuple(site.id for site in structure.sites)
        if len(set(site_ids)) != len(site_ids):
            raise AsymmetricUnitMappingInvariantError(
                "symmetry.asu.duplicate_site_id",
                "independent site IDs must be unique",
            )

        mapped = tuple(
            self._map_site(site, context, structure.id)
            for site in structure.sites
        )
        site_orbits = tuple(sorted((item[0] for item in mapped), key=lambda item: item.independent_site_id))
        normalized_by_id = {
            site.id: normalized
            for site, (_, normalized) in zip(structure.sites, mapped, strict=True)
        }
        sites_by_id = {site.id: site for site in structure.sites}
        fingerprint = _digest(
            {
                "symmetry_action": context.symmetry_action_fingerprint,
                "fractional_tolerance": self.fractional_tolerance,
                "sites": tuple(
                    {
                        "id": site_id,
                        "fractional": tuple(value.hex() for value in normalized_by_id[site_id]),
                        "components": tuple(
                            sorted(
                                (
                                    component.species.label,
                                    float(component.occupancy.value).hex(),
                                    None
                                    if component.oxidation_state is None
                                    or component.oxidation_state.value is None
                                    else float(component.oxidation_state.value).hex(),
                                )
                                for component in sites_by_id[site_id].components
                            )
                        ),
                    }
                    for site_id in sorted(site_ids)
                ),
            }
        )
        lookup = MappingProxyType(
            {item.independent_site_id: item for item in site_orbits}
        )
        return AsymmetricUnitMapping(
            site_orbits=site_orbits,
            symmetry_context_fingerprint=context.fingerprint,
            fractional_tolerance=self.fractional_tolerance,
            fingerprint=fingerprint,
            _site_lookup=lookup,
        )


__all__ = [
    "AsymmetricUnitMapper",
    "AsymmetricUnitMapping",
    "AsymmetricUnitMappingInvariantError",
    "FractionalPosition",
    "SiteImage",
    "SiteOrbitMapping",
]
