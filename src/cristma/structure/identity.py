"""Stable structure and symmetry-derived atom identity records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .crystal import DisplacementParameters
    from .occupation import SiteComponent


def _expanded_atom_id(
    structure_id: str | None,
    site_id: str,
    fractional: tuple[float, float, float],
    tolerance: float,
) -> str:
    """Return the shared identity of one reference-cell atom image."""

    position_key = tuple(round(value / tolerance) for value in fractional)
    return (
        f"expanded:{structure_id or 'unassigned'}:{site_id}:"
        f"{','.join(map(str, position_key))}"
    )


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Location of one scientific record in its source."""

    source_name: str | None = None
    format: str | None = None
    record_id: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass(frozen=True, slots=True)
class StructureProvenance:
    """Minimal origin chain shared by all canonical structures."""

    source: SourceReference | None = None
    parent_structure_id: str | None = None


@dataclass(frozen=True, slots=True)
class SymmetryImageProvenance:
    """How one symmetry operation was normalized into the reference cell."""

    operation_id: str
    normalization_translation: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ExpandedAtom:
    """One finite symmetry-expanded geometric position in the reference cell."""

    id: str
    structure_id: str | None
    source_site_id: str
    fractional: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    components: tuple[SiteComponent, ...]
    displacement: DisplacementParameters | None
    representative_image: SymmetryImageProvenance
    equivalent_images: tuple[SymmetryImageProvenance, ...]

    @property
    def independent_site_id(self) -> str:
        """Return the source asymmetric-unit site identity."""

        return self.source_site_id


@dataclass(frozen=True, slots=True)
class PeriodicAtomRef:
    """Reference to one lattice-translated canonical atom-image identity."""

    atom_id: str
    cell_translation: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.atom_id:
            raise ValueError("periodic atom reference ID must not be empty")
        if len(self.cell_translation) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.cell_translation
        ):
            raise ValueError("cell translation must contain three integers")


__all__ = [
    "ExpandedAtom",
    "PeriodicAtomRef",
    "SourceReference",
    "StructureProvenance",
    "SymmetryImageProvenance",
]
