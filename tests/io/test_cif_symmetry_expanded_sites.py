from __future__ import annotations

import pytest

import cristma


HEADER = """\
data_expanded
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 2
_space_group_name_H-M_alt 'P -1'
loop_
_space_group_symop_operation_xyz
{operations}
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
_atom_site_U_iso_or_equiv
{rows}
"""


def _read(
    rows: tuple[str, ...],
    *,
    operations: tuple[str, ...] = ("x,y,z", "-x,-y,-z"),
):
    result = cristma.read_text(
        HEADER.format(operations="\n".join(operations), rows="\n".join(rows)),
        format="cif",
        source_name="expanded.cif",
    )
    assert result.ok
    return result, result.structures[0]


def _diagnostic_codes(result) -> tuple[str, ...]:
    return tuple(item.code for item in result.diagnostics)


def test_symmetry_expanded_rows_collapse_to_one_independent_site() -> None:
    result, structure = _read(
        (
            "C1 C 0.1 0.2 0.3 1 . . .",
            "C2 C -0.1 -0.2 -0.3 1 . . .",
        )
    )

    assert len(structure.sites) == 1
    assert len(structure.atomic_view().atoms) == 2
    assert structure.sites[0].metadata["symmetry_source_rows"] == (0, 1)
    assert structure.sites[0].metadata["symmetry_source_labels"] == ("C1", "C2")
    assert "cif.map.symmetry_expanded_sites_collapsed" in _diagnostic_codes(result)


@pytest.mark.parametrize(
    "second_row",
    (
        "N2 N -0.1 -0.2 -0.3 1 . . .",
        "C2 C -0.1 -0.2 -0.3 0.5 . . .",
        "C2 C -0.1 -0.2 -0.3 1 A 2 .",
        "C2 C -0.1 -0.2 -0.3 1 . . 0.02",
    ),
)
def test_incompatible_symmetry_related_rows_are_not_merged(
    second_row: str,
) -> None:
    first_u_iso = "0.01" if second_row.endswith("0.02") else "."
    result, structure = _read(
        (
            f"C1 C 0.1 0.2 0.3 1 . . {first_u_iso}",
            second_row,
        )
    )

    assert len(structure.sites) == 2
    assert "cif.map.symmetry_equivalent_sites_unmerged" in _diagnostic_codes(result)


def test_identity_symmetry_does_not_collapse_distinct_rows() -> None:
    _result, structure = _read(
        (
            "C1 C 0.1 0.2 0.3 1 . . .",
            "C2 C -0.1 -0.2 -0.3 1 . . .",
        ),
        operations=("x,y,z",),
    )

    assert len(structure.sites) == 2


def test_integer_shifts_and_operation_order_do_not_change_grouping() -> None:
    rows = (
        "C1 C 0.1 0.2 0.3 1 . . .",
        "C2 C 0.9 0.8 0.7 1 . . .",
    )

    _first_result, first = _read(rows)
    _second_result, second = _read(
        rows,
        operations=("-x,-y,-z", "x,y,z"),
    )

    assert len(first.sites) == len(second.sites) == 1
    assert first.sites[0].metadata == second.sites[0].metadata
