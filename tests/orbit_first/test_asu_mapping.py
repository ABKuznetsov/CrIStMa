from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from cristma.core import MeasuredValue, UnitCell
from cristma.crystallography import (
    AsymmetricUnitMapper,
    SpaceGroupCatalog,
    SymmetryContext,
    canonical_operation_key,
)
from cristma.structure import CrystalStructure, IndependentSite, SiteComponent
from cristma.symmetry import AffineOperation


def _value(value: float, raw: str | None = None) -> MeasuredValue:
    return MeasuredValue(value, None, raw or str(value))


def _cell() -> UnitCell:
    return UnitCell.cubic(_value(5.0))


def _operation(
    rotation: tuple[tuple[int, int, int], ...],
) -> AffineOperation:
    return AffineOperation(
        tuple(tuple(Fraction(value) for value in row) for row in rotation),
        (Fraction(0), Fraction(0), Fraction(0)),
    )


IDENTITY = _operation(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
INVERSION = _operation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))


def _site(site_id: str, coordinates: tuple[float, float, float]) -> IndependentSite:
    return IndependentSite(
        id=site_id,
        label=site_id,
        components=(SiteComponent("C", _value(1.0)),),
        fractional=tuple(_value(value) for value in coordinates),
    )


def _structure(cell: UnitCell, *sites: IndependentSite) -> CrystalStructure:
    return CrystalStructure("fixture", cell, tuple(sites))


def test_general_position_has_one_image_per_operation() -> None:
    cell = _cell()
    context = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)
    site = _site("general", (0.13, 0.27, 0.39))

    mapping = AsymmetricUnitMapper().build(_structure(cell, site), context)
    orbit = mapping.by_site_id[site.id]

    assert len(orbit.reference_cell_images) == 2
    assert len(orbit.stabilizer_relations) == 1
    assert all(len(image.equivalent_relations) == 1 for image in orbit.reference_cell_images)


def test_special_position_merges_images_but_retains_coset_evidence() -> None:
    cell = _cell()
    context = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)
    site = _site("origin", (0.0, 0.0, 0.0))

    mapping = AsymmetricUnitMapper().build(_structure(cell, site), context)
    orbit = mapping.by_site_id[site.id]

    assert len(orbit.reference_cell_images) == 1
    assert len(orbit.stabilizer_relations) == 2
    assert len(orbit.reference_cell_images[0].equivalent_relations) == 2


def test_stabilizer_retains_nonzero_lattice_translation() -> None:
    cell = _cell()
    context = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)
    site = _site("boundary", (0.5, 0.0, 0.0))

    orbit = AsymmetricUnitMapper().build(_structure(cell, site), context).by_site_id[site.id]

    inversion_key = canonical_operation_key(INVERSION)
    inversion_relation = next(
        relation
        for relation in orbit.stabilizer_relations
        if relation.operation_key == inversion_key
    )
    assert inversion_relation.lattice_translation == (1, 0, 0)


def test_mapping_identity_is_independent_of_operation_order() -> None:
    cell = _cell()
    site = _site("site", (0.13, 0.27, 0.39))
    structure = _structure(cell, site)

    first = AsymmetricUnitMapper().build(
        structure,
        SymmetryContext.from_operations((IDENTITY, INVERSION), cell),
    )
    second = AsymmetricUnitMapper().build(
        structure,
        SymmetryContext.from_operations((INVERSION, IDENTITY), cell),
    )

    assert first == second
    assert first.fingerprint == second.fingerprint


def test_integer_shift_of_reported_coordinate_keeps_periodic_mapping_identity() -> None:
    cell = _cell()
    context = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)

    first = AsymmetricUnitMapper().build(
        _structure(cell, _site("same", (0.13, 0.27, 0.39))),
        context,
    )
    shifted = AsymmetricUnitMapper().build(
        _structure(cell, _site("same", (1.13, -0.73, 2.39))),
        context,
    )

    assert first.fingerprint == shifted.fingerprint
    assert first.site_orbits == shifted.site_orbits


def _invariant_cell(setting_operations: tuple[AffineOperation, ...]) -> UnitCell:
    metric = np.zeros((3, 3), dtype=float)
    for operation in setting_operations:
        rotation = np.asarray(operation.rotation, dtype=float)
        metric += rotation.T @ rotation
    metric *= 25.0 / float(np.max(np.diag(metric)))
    edges = np.sqrt(np.diag(metric))

    def angle(first: int, second: int) -> float:
        cosine = metric[first, second] / (edges[first] * edges[second])
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    return UnitCell(
        _value(float(edges[0])),
        _value(float(edges[1])),
        _value(float(edges[2])),
        _value(angle(1, 2)),
        _value(angle(0, 2)),
        _value(angle(0, 1)),
    )


def test_catalog_general_and_special_positions_obey_orbit_stabilizer_for_all_settings() -> None:
    parameters = (Fraction(1, 7), Fraction(2, 11), Fraction(3, 13))
    for setting in SpaceGroupCatalog.default().settings:
        cell = _invariant_cell(setting.symmetry_operations)
        context = SymmetryContext.from_setting(setting, cell)
        positions = tuple(
            {position.letter: position for position in (
                setting.wyckoff_positions[0],
                setting.wyckoff_positions[-1],
            )}.values()
        )
        sites = tuple(
            _site(
                f"setting-{setting.setting_id}-{position.letter}",
                tuple(
                    float(value % 1)
                    for value in position.coordinate_constraints[0].evaluate(parameters)
                ),
            )
            for position in positions
        )

        structure = CrystalStructure(
            "catalog-setting-fixture",
            cell,
            sites,
            id=f"catalog-setting:{setting.setting_id}",
            space_group=setting.definition(provenance="derived"),
        )
        mapping = AsymmetricUnitMapper().build(structure, context)
        expanded_ids = {atom.id for atom in structure.atomic_view().atoms}

        for site, position in zip(sites, positions, strict=True):
            orbit = mapping.by_site_id[site.id]
            assert len(orbit.reference_cell_images) == position.multiplicity, setting.setting_id
            assert (
                len(orbit.reference_cell_images) * len(orbit.stabilizer_relations)
                == len(setting.symmetry_operations)
            ), setting.setting_id
            assert {
                image.image_id for image in orbit.reference_cell_images
            } <= expanded_ids, setting.setting_id
