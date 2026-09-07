from __future__ import annotations

from dataclasses import replace
import math

import pytest

from cristma.core import MeasuredValue, UnitCell
from cristma.crystallography import SpaceGroupCatalog
from cristma.diffraction import (
    BraggBrentanoGeometry,
    ConstantWidthProfile,
    IsotropicSampleBroadening,
    PowderLineCalculator,
    PowderCorrectionCalculator,
    PowderProfileCalculator,
    RadiationSpectrum,
    ReflectionGenerator,
    StructureFactorCalculator,
    UniformTwoThetaGrid,
    XRayScatteringContext,
)
from cristma.structure import CrystalStructure, IndependentSite, SiteComponent


def _value(value: float) -> MeasuredValue:
    return MeasuredValue(value, None, str(value))


def _single_family_lines():
    cell = UnitCell.cubic(_value(4.0))
    setting = SpaceGroupCatalog.default().by_number(1)[0]
    site = IndependentSite(
        "C1",
        "C1",
        (SiteComponent("C", _value(1.0)),),
        (_value(0.0), _value(0.0), _value(0.0)),
    )
    structure = CrystalStructure(
        "profile-fixture",
        cell,
        (site,),
        space_group=setting.definition(provenance="derived"),
    )
    reflections = ReflectionGenerator().generate(cell, setting, 3.9)
    factors = StructureFactorCalculator().calculate(
        structure,
        setting,
        reflections,
        XRayScatteringContext.default(),
    )
    lines = PowderLineCalculator().calculate(
        factors,
        RadiationSpectrum.synchrotron(
            wavelength_angstrom=1.5406,
            source_id="profile-fixture",
        ),
    )
    return replace(
        lines,
        families=(lines.families[0],),
        provenance=replace(lines.provenance, families_emitted=1),
    )


def test_isotropic_sample_widths_follow_explicit_size_and_strain_formulas() -> None:
    model = IsotropicSampleBroadening(
        crystallite_size_nm=80.0,
        microstrain=8.0e-4,
        scherrer_constant=0.9,
    )

    gaussian, lorentzian = model.component_fwhm_deg_at(
        two_theta_deg=60.0,
        wavelength_angstrom=1.5406,
    )

    theta = math.radians(30.0)
    assert gaussian == pytest.approx(math.degrees(4.0 * 8.0e-4 * math.tan(theta)))
    assert lorentzian == pytest.approx(
        math.degrees(0.9 * 1.5406 / (800.0 * math.cos(theta)))
    )


def test_each_isotropic_sample_contribution_is_optional() -> None:
    size_only = IsotropicSampleBroadening(crystallite_size_nm=100.0)
    strain_only = IsotropicSampleBroadening(microstrain=1.0e-3)

    size_gaussian, size_lorentzian = size_only.component_fwhm_deg_at(40.0, 1.0)
    strain_gaussian, strain_lorentzian = strain_only.component_fwhm_deg_at(40.0, 1.0)

    assert size_gaussian == 0.0
    assert size_lorentzian > 0.0
    assert strain_gaussian > 0.0
    assert strain_lorentzian == 0.0


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"crystallite_size_nm": 0.0},
        {"crystallite_size_nm": True},
        {"microstrain": -1.0e-3},
        {"microstrain": True},
        {"scherrer_constant": 0.0, "microstrain": 1.0e-3},
    ),
)
def test_isotropic_sample_broadening_rejects_invalid_inputs(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        IsotropicSampleBroadening(**arguments)


def test_profile_applies_phase_local_sample_broadening_and_records_it() -> None:
    lines = _single_family_lines()
    center = lines.lines[0].two_theta_deg
    grid = UniformTwoThetaGrid(center - 1.0, center + 1.0, 0.002)
    instrument = ConstantWidthProfile(0.05)
    sample = IsotropicSampleBroadening(
        crystallite_size_nm=80.0,
        microstrain=8.0e-4,
    )
    calculator = PowderProfileCalculator()

    instrument_only = calculator.calculate(lines, grid, instrument)
    broadened = calculator.calculate(
        lines,
        grid,
        instrument,
        sample_broadening=sample,
    )

    assert max(broadened.intensity) < max(instrument_only.intensity)
    assert broadened.provenance.broadening_scope == "instrument_and_isotropic_sample"
    assert broadened.provenance.sample_broadening == sample


def test_sample_broadening_uses_source_wavelength_for_corrected_lines() -> None:
    lines = _single_family_lines()
    corrected = PowderCorrectionCalculator().calculate(
        lines,
        BraggBrentanoGeometry(),
    )
    center = lines.lines[0].two_theta_deg

    profile = PowderProfileCalculator().calculate(
        corrected,
        UniformTwoThetaGrid(center - 1.0, center + 1.0, 0.002),
        ConstantWidthProfile(0.05),
        sample_broadening=IsotropicSampleBroadening(crystallite_size_nm=80.0),
    )

    assert profile.source_lines is corrected
    assert profile.provenance.lines_contributed == 1
