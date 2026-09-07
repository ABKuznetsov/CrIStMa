from __future__ import annotations

import pytest

from cristma.core import MeasuredValue, UnitCell
from cristma.crystallography import SpaceGroupCatalog, SpaceGroupSetting
from cristma.diffraction import (
    ExtinctionAnalyzer,
    MillerIndex,
    ReflectionGenerator,
    StructureFactorCalculator,
    XRayScatteringContext,
)
from cristma.structure import CrystalStructure, IndependentSite, SiteComponent


def _value(value: float) -> MeasuredValue:
    return MeasuredValue(value, None, str(value))


def _site(
    site_id: str,
    element: str,
    fractional: tuple[float, float, float],
) -> IndependentSite:
    return IndependentSite(
        id=site_id,
        label=site_id,
        components=(SiteComponent(element, _value(1.0)),),
        fractional=tuple(_value(value) for value in fractional),
    )


def _setting(number: int) -> SpaceGroupSetting:
    matches = SpaceGroupCatalog.default().by_number(number)
    assert len(matches) == 1
    return matches[0]


def _catalog_setting(setting_id: int) -> SpaceGroupSetting:
    return SpaceGroupCatalog.default().by_setting(setting_id)


def _structure_factors(
    setting: SpaceGroupSetting,
    *sites: IndependentSite,
):
    cell = UnitCell.cubic(_value(4.0))
    structure = CrystalStructure(
        "analytical-reference",
        cell,
        tuple(sites),
        space_group=setting.definition(provenance="derived"),
    )
    reflections = ReflectionGenerator().generate(cell, setting, 2.0)
    factors = StructureFactorCalculator().calculate(
        structure,
        setting,
        reflections,
        XRayScatteringContext.default(),
    )
    return reflections, factors


def test_p1_atom_at_origin_has_atomic_form_factor_amplitude() -> None:
    """Catch omission, duplication, or a wrong phase for a single atom."""

    setting = _setting(1)
    reflections, factors = _structure_factors(
        setting,
        _site("C1", "C", (0.0, 0.0, 0.0)),
    )
    hkl = MillerIndex(1, 0, 0)
    reflection = next(item for item in reflections.reflections if item.representative_hkl == hkl)
    factor = next(item for item in factors.structure_factors if item.representative_hkl == hkl)
    s = 1.0 / (2.0 * reflection.d_spacing)
    expected_f0 = factors.context.form_factors.evaluate(6, s)

    assert factor.f_complex.real == pytest.approx(expected_f0, rel=1e-13)
    assert factor.f_complex.imag == pytest.approx(0.0, abs=1e-13)
    assert factor.f_squared == pytest.approx(expected_f0**2, rel=1e-13)


def test_p1_quarter_coordinate_has_positive_imaginary_phase() -> None:
    """Catch reversal of the documented exp(+2 pi i h.x) phase convention."""

    setting = _setting(1)
    reflections, factors = _structure_factors(
        setting,
        _site("C1", "C", (0.25, 0.0, 0.0)),
    )
    hkl = MillerIndex(1, 0, 0)
    reflection = next(item for item in reflections.reflections if item.representative_hkl == hkl)
    factor = next(item for item in factors.structure_factors if item.representative_hkl == hkl)
    expected_f0 = factors.context.form_factors.evaluate(
        6,
        1.0 / (2.0 * reflection.d_spacing),
    )

    assert factor.f_complex.real == pytest.approx(0.0, abs=1e-13)
    assert factor.f_complex.imag == pytest.approx(expected_f0, rel=1e-13)


def test_p1_structural_cancellation_is_not_systematic_extinction() -> None:
    """Catch conflation of an atomic-basis zero with a space-group absence."""

    setting = _setting(1)
    reflections, factors = _structure_factors(
        setting,
        _site("C1", "C", (0.0, 0.0, 0.0)),
        _site("C2", "C", (0.5, 0.5, 0.5)),
    )
    odd = MillerIndex(1, 0, 0)
    even = MillerIndex(1, 1, 0)
    odd_reflection = next(
        item for item in reflections.reflections if item.representative_hkl == odd
    )
    odd_factor = next(
        item for item in factors.structure_factors if item.representative_hkl == odd
    )
    even_reflection = next(
        item for item in reflections.reflections if item.representative_hkl == even
    )
    even_factor = next(
        item for item in factors.structure_factors if item.representative_hkl == even
    )
    expected_even_f0 = factors.context.form_factors.evaluate(
        6,
        1.0 / (2.0 * even_reflection.d_spacing),
    )

    assert odd_reflection.extinction.absent is False
    assert abs(odd_factor.f_complex) <= 1e-12
    assert odd_factor.provenance.normalized_to_zero is False
    assert even_factor.f_complex.real == pytest.approx(2.0 * expected_even_f0, rel=1e-13)


def test_cscl_matches_two_atom_phase_sum_for_odd_and_even_parity() -> None:
    """Catch a wrong h.x phase sign or double-counting of special positions."""

    setting = _setting(221)
    reflections, factors = _structure_factors(
        setting,
        _site("Cs1", "Cs", (0.0, 0.0, 0.0)),
        _site("Cl1", "Cl", (0.5, 0.5, 0.5)),
    )

    for member in (MillerIndex(1, 0, 0), MillerIndex(1, 1, 0)):
        reflection = next(
            item for item in reflections.reflections if member in item.equivalent_hkls
        )
        factor = next(
            item
            for item in factors.structure_factors
            if item.reflection_id == reflection.reflection_id
        )
        s = 1.0 / (2.0 * reflection.d_spacing)
        f_cs = factors.context.form_factors.evaluate(55, s)
        f_cl = factors.context.form_factors.evaluate(17, s)
        parity = sum(reflection.representative_hkl.as_tuple()) % 2
        expected = f_cs + (-1 if parity else 1) * f_cl

        assert factor.f_complex.real == pytest.approx(expected, rel=1e-12)
        assert factor.f_complex.imag == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("setting_id", "hkl", "expected_absent"),
    (
        (6, (0, 1, 0), True),
        (6, (0, 2, 0), False),
        (6, (1, 0, 0), False),
        (523, (1, 0, 0), True),
        (523, (1, 1, 1), False),
        (523, (2, 0, 0), False),
        (523, (2, 1, 0), True),
        (529, (1, 0, 0), True),
        (529, (1, 1, 0), False),
        (529, (1, 1, 1), True),
    ),
)
def test_exact_phase_extinctions_match_analytical_parity_conditions(
    setting_id: int,
    hkl: tuple[int, int, int],
    expected_absent: bool,
) -> None:
    """Catch broken screw/centering phase sums without implementation tables."""

    result = ExtinctionAnalyzer().analyze(
        MillerIndex(*hkl),
        _catalog_setting(setting_id),
    )

    assert result.absent is expected_absent
    assert all(
        isinstance(phase.numerator, int) and isinstance(phase.denominator, int)
        for bucket in result.evidence
        for phase in bucket.exact_phases + bucket.relative_phases
    )
