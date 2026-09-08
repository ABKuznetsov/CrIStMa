"""One-call orchestration of the existing X-ray powder calculation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real

from cristma.crystallography import SpaceGroupSetting
from cristma.structure import CrystalStructure

from .pattern_models import PowderPatternProvenance, PowderPatternResult
from .powder import PowderLineCalculator
from .powder_corrections import PowderCorrectionCalculator
from .powder_correction_models import BraggBrentanoGeometry
from .powder_models import RadiationProbe, RadiationSpectrum
from .profile_models import (
    ConstantWidthProfile,
    IsotropicSampleBroadening,
    TchProfile,
    UniformTwoThetaGrid,
)
from .profiles import PowderProfileCalculator
from .reflections import ReflectionGenerator
from .structure_factor_models import XRayScatteringContext
from .structure_factors import StructureFactorCalculator


def _zero_shift(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("zero_shift_deg must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("zero_shift_deg must be finite")
    return normalized


def _raw_stop_two_theta_deg(
    grid: UniformTwoThetaGrid,
    zero_shift_deg: float,
) -> float:
    raw_stop = grid.stop_deg - min(zero_shift_deg, 0.0)
    if not 0.0 < raw_stop < 180.0:
        raise ValueError(
            "effective raw two-theta stop must lie within (0, 180) degrees"
        )
    return raw_stop


def _d_min_for_coverage(
    radiation: RadiationSpectrum,
    raw_stop_two_theta_deg: float,
) -> float:
    wavelength = min(
        component.wavelength_angstrom for component in radiation.components
    )
    theta = math.radians(raw_stop_two_theta_deg / 2.0)
    return wavelength / (2.0 * math.sin(theta))


@dataclass(frozen=True, slots=True)
class PowderPatternCalculator:
    """Compose a complete, traceable X-ray powder-pattern calculation."""

    reflection_generator: ReflectionGenerator = field(
        default_factory=ReflectionGenerator
    )
    structure_factor_calculator: StructureFactorCalculator = field(
        default_factory=StructureFactorCalculator
    )
    profile_calculator: PowderProfileCalculator = field(
        default_factory=PowderProfileCalculator
    )

    def calculate(
        self,
        structure: CrystalStructure,
        space_group: SpaceGroupSetting,
        radiation: RadiationSpectrum,
        grid: UniformTwoThetaGrid,
        instrument_profile: ConstantWidthProfile | TchProfile,
        *,
        geometry: BraggBrentanoGeometry | None = None,
        sample_broadening: IsotropicSampleBroadening | None = None,
        zero_shift_deg: float = 0.0,
        scattering_context: XRayScatteringContext | None = None,
    ) -> PowderPatternResult:
        if not isinstance(radiation, RadiationSpectrum):
            raise TypeError("radiation must be RadiationSpectrum")
        if radiation.probe is not RadiationProbe.XRAY:
            raise TypeError("powder pattern facade requires X-ray radiation")
        if not isinstance(grid, UniformTwoThetaGrid):
            raise TypeError("grid must be UniformTwoThetaGrid")
        shift = _zero_shift(zero_shift_deg)
        context = (
            XRayScatteringContext.default()
            if scattering_context is None
            else scattering_context
        )
        raw_stop = _raw_stop_two_theta_deg(grid, shift)
        d_min = _d_min_for_coverage(radiation, raw_stop)
        reflections = self.reflection_generator.generate(
            structure.cell,
            space_group,
            d_min,
        )
        factors = self.structure_factor_calculator.calculate(
            structure,
            space_group,
            reflections,
            context,
        )
        intrinsic = PowderLineCalculator().calculate(factors, radiation)
        corrected = (
            None
            if geometry is None
            else PowderCorrectionCalculator().calculate(intrinsic, geometry)
        )
        profile_source = intrinsic if corrected is None else corrected
        profile = self.profile_calculator.calculate(
            profile_source,
            grid,
            instrument_profile,
            zero_shift_deg=shift,
            sample_broadening=sample_broadening,
        )
        provenance = PowderPatternProvenance(
            method="cristma.diffraction.PowderPatternCalculator",
            version="1",
            derived_d_min_angstrom=d_min,
            raw_stop_two_theta_deg=raw_stop,
            radiation_source_id=radiation.source_id,
            radiation_probe=radiation.probe,
            scattering_model=context.model,
            correction_applied=corrected is not None,
            instrument_profile=instrument_profile,
            sample_broadening=sample_broadening,
            zero_shift_deg=shift,
            reflection_generator_config=tuple(
                sorted(self.reflection_generator.get_config().items())
            ),
            structure_factor_calculator_config=tuple(
                sorted(self.structure_factor_calculator.get_config().items())
            ),
            profile_calculator_config=tuple(
                sorted(self.profile_calculator.get_config().items())
            ),
        )
        return PowderPatternResult(
            reflection_set=reflections,
            structure_factor_set=factors,
            intrinsic_lines=intrinsic,
            corrected_lines=corrected,
            profile=profile,
            status=reflections.status,
            provenance=provenance,
        )


__all__ = ["PowderPatternCalculator"]
