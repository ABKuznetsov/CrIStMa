"""Immutable aggregate results for complete powder-pattern calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .models import ReflectionSet, ReflectionSetStatus
from .powder_correction_models import CorrectedPowderLineSet
from .powder_models import PowderLineSet, RadiationProbe
from .profile_models import (
    CalculatedProfile,
    ConstantWidthProfile,
    IsotropicSampleBroadening,
    TchProfile,
)
from .structure_factor_models import StructureFactorSet


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _configuration(
    value: tuple[tuple[str, object], ...],
    name: str,
) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and bool(item[0].strip())
        for item in value
    ):
        raise TypeError(f"{name} configuration must contain named tuple entries")
    keys = tuple(item[0] for item in value)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{name} configuration keys must be unique and sorted")


@dataclass(frozen=True, slots=True)
class PowderPatternProvenance:
    """Inputs and configured delegates used by the powder facade."""

    method: str
    version: str
    derived_d_min_angstrom: float
    raw_stop_two_theta_deg: float
    radiation_source_id: str
    radiation_probe: RadiationProbe
    scattering_model: str
    correction_applied: bool
    instrument_profile: ConstantWidthProfile | TchProfile
    sample_broadening: IsotropicSampleBroadening | None
    zero_shift_deg: float
    reflection_generator_config: tuple[tuple[str, object], ...]
    structure_factor_calculator_config: tuple[tuple[str, object], ...]
    profile_calculator_config: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        for value, name in (
            (self.method, "pattern method"),
            (self.version, "pattern method version"),
            (self.radiation_source_id, "radiation source ID"),
            (self.scattering_model, "scattering model"),
        ):
            _nonempty(value, name)
        d_min = _finite(self.derived_d_min_angstrom, "derived d_min")
        if d_min <= 0.0:
            raise ValueError("derived d_min must be positive")
        object.__setattr__(self, "derived_d_min_angstrom", d_min)
        raw_stop = _finite(self.raw_stop_two_theta_deg, "raw stop two-theta")
        if not 0.0 < raw_stop < 180.0:
            raise ValueError("raw stop two-theta must lie within (0, 180) degrees")
        object.__setattr__(self, "raw_stop_two_theta_deg", raw_stop)
        if not isinstance(self.radiation_probe, RadiationProbe):
            raise TypeError("radiation probe must be RadiationProbe")
        if not isinstance(self.correction_applied, bool):
            raise TypeError("correction flag must be bool")
        if not isinstance(self.instrument_profile, (ConstantWidthProfile, TchProfile)):
            raise TypeError("instrument profile must be a supported profile model")
        if self.sample_broadening is not None and not isinstance(
            self.sample_broadening, IsotropicSampleBroadening
        ):
            raise TypeError("sample broadening must be IsotropicSampleBroadening or None")
        object.__setattr__(
            self,
            "zero_shift_deg",
            _finite(self.zero_shift_deg, "zero shift"),
        )
        for value, name in (
            (self.reflection_generator_config, "reflection generator"),
            (
                self.structure_factor_calculator_config,
                "structure-factor calculator",
            ),
            (self.profile_calculator_config, "profile calculator"),
        ):
            _configuration(value, name)


@dataclass(frozen=True, slots=True)
class PowderPatternResult:
    """One traceable calculated powder profile and all of its source results."""

    reflection_set: ReflectionSet
    structure_factor_set: StructureFactorSet
    intrinsic_lines: PowderLineSet
    corrected_lines: CorrectedPowderLineSet | None
    profile: CalculatedProfile
    status: ReflectionSetStatus
    provenance: PowderPatternProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.reflection_set, ReflectionSet):
            raise TypeError("reflection_set must be ReflectionSet")
        if not isinstance(self.structure_factor_set, StructureFactorSet):
            raise TypeError("structure_factor_set must be StructureFactorSet")
        if self.structure_factor_set.reflection_set is not self.reflection_set:
            raise ValueError("structure-factor result belongs to another reflection set")
        if not isinstance(self.intrinsic_lines, PowderLineSet):
            raise TypeError("intrinsic_lines must be PowderLineSet")
        if self.intrinsic_lines.structure_factors is not self.structure_factor_set:
            raise ValueError("intrinsic lines belong to another structure-factor result")
        if not isinstance(self.profile, CalculatedProfile):
            raise TypeError("profile must be CalculatedProfile")
        if not isinstance(self.provenance, PowderPatternProvenance):
            raise TypeError("provenance must be PowderPatternProvenance")
        if self.corrected_lines is None:
            if self.profile.source_lines is not self.intrinsic_lines:
                raise ValueError("profile source must be the intrinsic line result")
            if self.provenance.correction_applied:
                raise ValueError("correction provenance requires corrected lines")
        else:
            if not isinstance(self.corrected_lines, CorrectedPowderLineSet):
                raise TypeError("corrected_lines must be CorrectedPowderLineSet or None")
            if self.corrected_lines.powder_lines is not self.intrinsic_lines:
                raise ValueError("corrected lines belong to another intrinsic line result")
            if self.profile.source_lines is not self.corrected_lines:
                raise ValueError("profile source must be the corrected line result")
            if not self.provenance.correction_applied:
                raise ValueError("corrected lines require correction provenance")
        if not isinstance(self.status, ReflectionSetStatus):
            raise TypeError("status must be ReflectionSetStatus")
        if self.status is not self.reflection_set.status:
            raise ValueError("pattern status must match the reflection-set status")
        spectrum = self.intrinsic_lines.spectrum
        if (
            self.provenance.radiation_source_id != spectrum.source_id
            or self.provenance.radiation_probe is not spectrum.probe
        ):
            raise ValueError("pattern provenance belongs to another radiation spectrum")
        if self.provenance.derived_d_min_angstrom != self.reflection_set.provenance.d_min:
            raise ValueError("pattern provenance disagrees with reflection d_min")
        profile_provenance = self.profile.provenance
        if (
            profile_provenance.instrument_profile is not self.provenance.instrument_profile
            or profile_provenance.sample_broadening
            is not self.provenance.sample_broadening
            or profile_provenance.zero_shift_deg != self.provenance.zero_shift_deg
        ):
            raise ValueError("pattern provenance disagrees with profile inputs")


__all__ = ["PowderPatternProvenance", "PowderPatternResult"]
