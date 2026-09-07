from __future__ import annotations

import pytest

import cristma
from cristma.crystallography import AsymmetricUnitMapper, SymmetryContext


R32_CA2_CIF = """\
data_ca2
_space_group_IT_number 155
_space_group_name_H-M_alt 'R 3 2 :H'
_cell_length_a 13.2516
_cell_length_b 13.2516
_cell_length_c 14.9679
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120

loop_
_space_group_symop_operation_xyz
'x, y, z'
'-y, x-y, z'
'-x+y, -x, z'
'y, x, -z'
'x-y, -y, -z'
'-x, -x+y, -z'
'x+2/3, y+1/3, z+1/3'
'-y+2/3, x-y+1/3, z+1/3'
'-x+y+2/3, -x+1/3, z+1/3'
'y+2/3, x+1/3, -z+1/3'
'x-y+2/3, -y+1/3, -z+1/3'
'-x+2/3, -x+y+1/3, -z+1/3'
'x+1/3, y+2/3, z+2/3'
'-y+1/3, x-y+2/3, z+2/3'
'-x+y+1/3, -x+2/3, z+2/3'
'y+1/3, x+2/3, -z+2/3'
'x-y+1/3, -y+2/3, -z+2/3'
'-x+1/3, -x+y+2/3, -z+2/3'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Ca2 Ca 0.66667 0.33333 -0.66667 1
"""


@pytest.mark.parametrize(
    "reported_coordinates",
    (
        "0.667 0.333 -0.667",
        "0.66667 0.33333 -0.66667",
    ),
)
def test_special_position_is_projected_within_reported_decimal_precision(
    reported_coordinates: str,
) -> None:
    result = cristma.read_text(
        R32_CA2_CIF.replace(
            "0.66667 0.33333 -0.66667",
            reported_coordinates,
        ),
        format="cif",
        source_name="rounded-r32-ca2.cif",
    )

    assert result.ok
    structure = tuple(result.structures)[0]
    assert tuple(value.value for value in structure.sites[0].fractional) == pytest.approx(
        (2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0),
        abs=1e-14,
    )
    assert any(
        diagnostic.code == "cif.map.special_position_symmetrized"
        for diagnostic in result.diagnostics
    )

    context = SymmetryContext.from_definition(structure.space_group, structure.cell)
    orbit = AsymmetricUnitMapper().build(structure, context).site_orbits[0]

    assert len(orbit.reference_cell_images) == 3
    assert len(orbit.stabilizer_relations) == 6
