"""Outward-only materialization of orbit-first contact results."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from cristma.crystallography import PeriodicSymmetryRelation, canonical_instance_owner
from cristma.crystallography.symmetry_context import _digest
from cristma.structure import PeriodicAtomRef
from .orbit_contacts import ContactInterpretation
from .shell_orbits import ShellRole


@dataclass(frozen=True, slots=True)
class ReferenceCell:
    """Canonical owners in cell (0, 0, 0)."""

    @property
    def owner_cells(self) -> tuple[tuple[int, int, int], ...]:
        return ((0, 0, 0),)


@dataclass(frozen=True, slots=True)
class CellRange:
    a_min: int
    a_max: int
    b_min: int
    b_max: int
    c_min: int
    c_max: int

    def __post_init__(self) -> None:
        values = (self.a_min, self.a_max, self.b_min, self.b_max, self.c_min, self.c_max)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("cell-range bounds must be integers")
        if self.a_min > self.a_max or self.b_min > self.b_max or self.c_min > self.c_max:
            raise ValueError("cell-range lower bounds must not exceed upper bounds")

    @property
    def owner_cells(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (a, b, c)
            for a in range(self.a_min, self.a_max + 1)
            for b in range(self.b_min, self.b_max + 1)
            for c in range(self.c_min, self.c_max + 1)
        )


@dataclass(frozen=True, slots=True, order=True)
class ShellMembership:
    shell_orbit_id: str
    alternative_id: str
    role: ShellRole


@dataclass(frozen=True, slots=True)
class ResolvedContact:
    contact_id: str
    resolved_contact_orbit_id: str
    first_atom_ref: PeriodicAtomRef
    second_atom_ref: PeriodicAtomRef
    distance: float
    vector_cartesian: tuple[float, float, float]
    interpretations: tuple[ContactInterpretation, ...]
    shell_memberships: tuple[ShellMembership, ...]
    provenance: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not self.contact_id or not self.resolved_contact_orbit_id or not self.interpretations:
            raise ValueError("materialized contact requires identities and interpretations")
        if not math.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("materialized contact distance must be positive and finite")
        if len(self.vector_cartesian) != 3 or not all(math.isfinite(x) for x in self.vector_cartesian):
            raise ValueError("materialized contact vector must contain three finite values")
        if tuple(sorted(set(self.shell_memberships))) != self.shell_memberships:
            raise ValueError("shell memberships must be unique and sorted")


def _owners(result, geometry):
    owners = set()
    context = result._symmetry_context
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        owners.add(canonical_instance_owner(
            geometry.first_independent_site_id,
            action,
            geometry.second_independent_site_id,
            action.compose(geometry.canonical_relation, context),
            result._asymmetric_unit_mapping,
        ))
    if len(owners) != geometry.multiplicity_in_reference_cell:
        raise ValueError("materialized owner count disagrees with pair-orbit multiplicity")
    return tuple(sorted(owners))


def _memberships(result):
    incidence_by_id = {x.incidence_orbit_id: x for x in result.contact_incidence_orbits}
    output: dict[str, set[ShellMembership]] = {}
    for shell in result.coordination_shell_orbits:
        for alternative in shell.alternatives:
            for role, incidence_ids in (
                (ShellRole.PRIMARY, alternative.primary_incidence_ids),
                (ShellRole.SECONDARY, alternative.secondary_incidence_ids),
            ):
                for incidence_id in incidence_ids:
                    contact_id = incidence_by_id[incidence_id].resolved_contact_orbit_id
                    output.setdefault(contact_id, set()).add(
                        ShellMembership(shell.shell_orbit_id, alternative.alternative_id, role)
                    )
    return {key: tuple(sorted(value)) for key, value in output.items()}


def _image_position(result, atom_ref: PeriodicAtomRef):
    image = next(
        image
        for site in result._asymmetric_unit_mapping.site_orbits
        for image in site.reference_cell_images
        if image.image_id == atom_ref.atom_id
    )
    return np.asarray(image.fractional_position, dtype=float) + np.asarray(atom_ref.cell_translation, dtype=float)


class ContactMaterializer:
    """Create consumer records without feeding them back into scientific analysis."""

    def materialize(
        self,
        result,
        region: ReferenceCell | CellRange,
        contact_orbit_ids: tuple[str, ...] | None = None,
        interpretation_ids: tuple[str, ...] | None = None,
        shell_alternative_ids: tuple[str, ...] | None = None,
    ) -> tuple[ResolvedContact, ...]:
        if not isinstance(region, (ReferenceCell, CellRange)):
            raise TypeError("region must be ReferenceCell or CellRange")
        selected_orbits = None if contact_orbit_ids is None else set(contact_orbit_ids)
        selected_interpretations = None if interpretation_ids is None else set(interpretation_ids)
        selected_alternatives = None if shell_alternative_ids is None else set(shell_alternative_ids)
        geometry_by_id = {x.geometry_orbit_id: x for x in result.pair_table.contact_orbits}
        memberships_by_contact = _memberships(result)
        contacts: list[ResolvedContact] = []
        for resolved in result.contact_orbits:
            if selected_orbits is not None and resolved.resolved_contact_orbit_id not in selected_orbits:
                continue
            interpretations = tuple(
                item for item in resolved.interpretations
                if selected_interpretations is None or item.interpretation_id in selected_interpretations
            )
            if not interpretations:
                continue
            memberships = tuple(
                item for item in memberships_by_contact.get(resolved.resolved_contact_orbit_id, ())
                if selected_alternatives is None or item.alternative_id in selected_alternatives
            )
            if selected_alternatives is not None and not memberships:
                continue
            geometry = geometry_by_id[resolved.geometry_orbit_id]
            for owner in _owners(result, geometry):
                for owner_cell in region.owner_cells:
                    first = PeriodicAtomRef(owner[0][1], tuple(owner[0][2][i] + owner_cell[i] for i in range(3)))
                    second = PeriodicAtomRef(owner[1][1], tuple(owner[1][2][i] + owner_cell[i] for i in range(3)))
                    vector = (_image_position(result, second) - _image_position(result, first)) @ result._structure.cell.matrix
                    contact_id = "contact:" + _digest({
                        "resolved_contact_orbit_id": resolved.resolved_contact_orbit_id,
                        "first": (first.atom_id, first.cell_translation),
                        "second": (second.atom_id, second.cell_translation),
                    })
                    contacts.append(ResolvedContact(
                        contact_id,
                        resolved.resolved_contact_orbit_id,
                        first,
                        second,
                        float(np.linalg.norm(vector)),
                        tuple(float(value) for value in vector),
                        interpretations,
                        memberships,
                        (*resolved.provenance, ("materialization", "canonical_owner")),
                    ))
        return tuple(sorted(contacts, key=lambda item: item.contact_id))


class IntermolecularContactMaterializer:
    """Create intermolecular instances without feeding them back into science."""

    def materialize(
        self,
        result,
        region: ReferenceCell | CellRange,
        contact_orbit_ids: tuple[str, ...] | None = None,
        interpretation_ids: tuple[str, ...] | None = None,
    ):
        from .intermolecular_models import MaterializedIntermolecularContact

        if not isinstance(region, (ReferenceCell, CellRange)):
            raise TypeError("region must be ReferenceCell or CellRange")
        selected_orbits = None if contact_orbit_ids is None else set(contact_orbit_ids)
        selected_interpretations = (
            None if interpretation_ids is None else set(interpretation_ids)
        )
        source = result._contact_analysis
        geometry_by_id = {
            item.geometry_orbit_id: item for item in source.pair_table.contact_orbits
        }
        contacts = []
        for orbit in result.contact_orbits:
            if (
                selected_orbits is not None
                and orbit.intermolecular_contact_orbit_id not in selected_orbits
            ):
                continue
            interpretations = tuple(
                item
                for item in orbit.interpretations
                if selected_interpretations is None
                or item.interpretation_id in selected_interpretations
            )
            if selected_interpretations is not None and not interpretations:
                continue
            geometry = geometry_by_id[orbit.geometry_orbit_id]
            for owner in _owners(source, geometry):
                for owner_cell in region.owner_cells:
                    first = PeriodicAtomRef(
                        owner[0][1],
                        tuple(
                            owner[0][2][index] + owner_cell[index]
                            for index in range(3)
                        ),
                    )
                    second = PeriodicAtomRef(
                        owner[1][1],
                        tuple(
                            owner[1][2][index] + owner_cell[index]
                            for index in range(3)
                        ),
                    )
                    vector = (
                        _image_position(source, second) - _image_position(source, first)
                    ) @ source._structure.cell.matrix
                    contact_id = "intermolecular-contact:" + _digest({
                        "intermolecular_contact_orbit_id": (
                            orbit.intermolecular_contact_orbit_id
                        ),
                        "first": (first.atom_id, first.cell_translation),
                        "second": (second.atom_id, second.cell_translation),
                    })
                    contacts.append(MaterializedIntermolecularContact(
                        contact_id,
                        orbit.intermolecular_contact_orbit_id,
                        orbit.first_molecule_orbit_id,
                        orbit.second_molecule_orbit_id,
                        first,
                        second,
                        float(np.linalg.norm(vector)),
                        tuple(float(value) for value in vector),
                        interpretations,
                        (*orbit.provenance, ("materialization", "canonical_owner")),
                    ))
        return tuple(sorted(contacts, key=lambda item: item.contact_id))


__all__ = [
    "CellRange",
    "ContactMaterializer",
    "IntermolecularContactMaterializer",
    "ReferenceCell",
    "ResolvedContact",
    "ShellMembership",
]
