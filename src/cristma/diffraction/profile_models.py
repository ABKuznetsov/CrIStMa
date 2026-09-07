"""Immutable inputs and results for calculated powder profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from .powder_correction_models import CorrectedPowderLineSet
from .powder_models import PowderLineSet


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_finite(value: float, name: str) -> float:
    normalized = _finite(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True, slots=True)
class UniformTwoThetaGrid:
    """Uniform two-theta grid including every point not beyond ``stop_deg``."""

    start_deg: float
    stop_deg: float
    step_deg: float

    def __post_init__(self) -> None:
        start = _finite(self.start_deg, "grid start")
        stop = _finite(self.stop_deg, "grid stop")
        step = _positive_finite(self.step_deg, "grid step")
        if start < 0.0 or stop > 180.0 or start >= stop:
            raise ValueError("grid must satisfy 0 <= start < stop <= 180 degrees")
        object.__setattr__(self, "start_deg", start)
        object.__setattr__(self, "stop_deg", stop)
        object.__setattr__(self, "step_deg", step)

    @property
    def point_count(self) -> int:
        return math.floor(
            (self.stop_deg - self.start_deg) / self.step_deg + 1.0e-12
        ) + 1

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(
            self.start_deg + index * self.step_deg
            for index in range(self.point_count)
        )


@dataclass(frozen=True, slots=True)
class ConstantWidthProfile:
    """Instrument-only Gaussian broadening with one explicit FWHM."""

    fwhm_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fwhm_deg",
            _positive_finite(self.fwhm_deg, "profile FWHM"),
        )


@dataclass(frozen=True, slots=True)
class TchProfile:
    """Instrument-only TCH pseudo-Voigt parameters in GSAS CW units.

    ``U``, ``V`` and ``W`` define Gaussian variance in centidegrees squared;
    ``X`` and ``Y`` define Lorentzian FWHM in centidegrees. No instrument
    defaults are implied by this value object.
    """

    u: float
    v: float
    w: float
    x: float
    y: float

    def __post_init__(self) -> None:
        for field_name in ("u", "v", "w", "x", "y"):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field_name.upper()),
            )

    def _component_widths_at(self, two_theta_deg: float) -> tuple[float, float]:
        angle = _finite(two_theta_deg, "two-theta")
        if not 0.0 <= angle < 180.0:
            raise ValueError("two-theta must be within [0, 180) degrees")
        theta = math.radians(angle / 2.0)
        tangent = math.tan(theta)
        cosine = math.cos(theta)
        gaussian_variance = self.u * tangent**2 + self.v * tangent + self.w
        if gaussian_variance <= 0.0:
            raise ValueError(
                "TCH Gaussian width is not positive at the evaluated angle"
            )
        gaussian_fwhm = (
            2.0 * math.sqrt(2.0 * math.log(2.0) * gaussian_variance) / 100.0
        )
        lorentzian_fwhm = (self.x / cosine + self.y * tangent) / 100.0
        if lorentzian_fwhm < 0.0:
            raise ValueError(
                "TCH Lorentzian width is negative at the evaluated angle"
            )
        return gaussian_fwhm, lorentzian_fwhm

    def _width_and_mixing_at(self, two_theta_deg: float) -> tuple[float, float]:
        return _tch_width_and_mixing(*self._component_widths_at(two_theta_deg))

    def fwhm_deg_at(self, two_theta_deg: float) -> float:
        """Return the TCH total FWHM at an explicit two-theta angle."""

        return self._width_and_mixing_at(two_theta_deg)[0]


@dataclass(frozen=True, slots=True)
class IsotropicSampleBroadening:
    """Phase-local isotropic crystallite-size and microstrain broadening.

    Size contributes a Lorentzian Scherrer FWHM. Microstrain contributes a
    Gaussian FWHM proportional to ``tan(theta)``. Both returned widths are in
    two-theta degrees.
    """

    crystallite_size_nm: float | None = None
    microstrain: float | None = None
    scherrer_constant: float = 0.9

    def __post_init__(self) -> None:
        size = self.crystallite_size_nm
        if size is not None:
            size = _positive_finite(size, "crystallite size")
            object.__setattr__(self, "crystallite_size_nm", size)
        strain = self.microstrain
        if strain is not None:
            strain = _finite(strain, "microstrain")
            if strain < 0.0:
                raise ValueError("microstrain must be non-negative")
            object.__setattr__(self, "microstrain", strain)
        object.__setattr__(
            self,
            "scherrer_constant",
            _positive_finite(self.scherrer_constant, "Scherrer constant"),
        )
        if size is None and (strain is None or strain == 0.0):
            raise ValueError("at least one sample-broadening contribution is required")

    def component_fwhm_deg_at(
        self,
        two_theta_deg: float,
        wavelength_angstrom: float,
    ) -> tuple[float, float]:
        """Return ``(Gaussian strain, Lorentzian size)`` FWHM in degrees."""

        angle = _finite(two_theta_deg, "two-theta")
        if not 0.0 <= angle < 180.0:
            raise ValueError("two-theta must be within [0, 180) degrees")
        wavelength = _positive_finite(wavelength_angstrom, "wavelength")
        theta = math.radians(angle / 2.0)
        gaussian = (
            0.0
            if self.microstrain is None
            else math.degrees(4.0 * self.microstrain * math.tan(theta))
        )
        lorentzian = (
            0.0
            if self.crystallite_size_nm is None
            else math.degrees(
                self.scherrer_constant
                * wavelength
                / (10.0 * self.crystallite_size_nm * math.cos(theta))
            )
        )
        return gaussian, lorentzian


def _tch_width_and_mixing(
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
) -> tuple[float, float]:
    hg = gaussian_fwhm
    hl = lorentzian_fwhm
    total = (
        hg**5
        + 2.69269 * hg**4 * hl
        + 2.42843 * hg**3 * hl**2
        + 4.47163 * hg**2 * hl**3
        + 0.07842 * hg * hl**4
        + hl**5
    ) ** 0.2
    ratio = hl / total
    mixing = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3
    return total, min(1.0, max(0.0, mixing))


class ProfileIntensityBasis(str, Enum):
    """Line intensity consumed by a profile calculation."""

    INTRINSIC = "intrinsic_line_intensity"
    CORRECTED = "corrected_line_intensity"


@dataclass(frozen=True, slots=True)
class PowderProfileProvenance:
    """Exact instrument and optional sample inputs of a sampled profile."""

    method: str
    version: str
    broadening_model: str
    broadening_scope: str
    instrument_profile: ConstantWidthProfile | TchProfile
    intensity_basis: ProfileIntensityBasis
    zero_shift_deg: float
    local_window_fwhm: float
    max_points: int | None
    lines_considered: int
    lines_contributed: int
    sample_broadening: IsotropicSampleBroadening | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.method, "profile method"),
            (self.version, "profile method version"),
            (self.broadening_model, "broadening model"),
            (self.broadening_scope, "broadening scope"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.intensity_basis, ProfileIntensityBasis):
            raise TypeError("intensity_basis must be ProfileIntensityBasis")
        expected_scope = (
            "instrument_only"
            if self.sample_broadening is None
            else "instrument_and_isotropic_sample"
        )
        if self.broadening_scope != expected_scope:
            raise ValueError("profile broadening scope disagrees with its inputs")
        if not isinstance(
            self.instrument_profile, (ConstantWidthProfile, TchProfile)
        ):
            raise TypeError("instrument_profile must be a supported profile model")
        if self.sample_broadening is not None and not isinstance(
            self.sample_broadening, IsotropicSampleBroadening
        ):
            raise TypeError("sample_broadening must be IsotropicSampleBroadening or None")
        object.__setattr__(
            self, "zero_shift_deg", _finite(self.zero_shift_deg, "zero shift")
        )
        object.__setattr__(
            self,
            "local_window_fwhm",
            _positive_finite(self.local_window_fwhm, "local profile window"),
        )
        if self.max_points is not None and (
            isinstance(self.max_points, bool)
            or not isinstance(self.max_points, int)
            or self.max_points <= 0
        ):
            raise ValueError("max_points must be a positive integer or None")
        for value, name in (
            (self.lines_considered, "lines_considered"),
            (self.lines_contributed, "lines_contributed"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.lines_contributed > self.lines_considered:
            raise ValueError("contributing lines cannot exceed considered lines")


@dataclass(frozen=True, slots=True)
class CalculatedProfile:
    """A sampled calculated profile retaining its line source and provenance."""

    two_theta_deg: tuple[float, ...]
    intensity: tuple[float, ...]
    intensity_basis: ProfileIntensityBasis
    source_lines: PowderLineSet | CorrectedPowderLineSet
    provenance: PowderProfileProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.two_theta_deg, tuple) or not isinstance(
            self.intensity, tuple
        ):
            raise TypeError("profile coordinates and intensity must be tuples")
        if not self.two_theta_deg or len(self.two_theta_deg) != len(self.intensity):
            raise ValueError("profile coordinates and intensity must have equal length")
        if not all(math.isfinite(value) for value in self.two_theta_deg):
            raise ValueError("profile coordinates must be finite")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.intensity):
            raise ValueError("profile intensity must be finite and non-negative")
        if not isinstance(self.intensity_basis, ProfileIntensityBasis):
            raise TypeError("intensity_basis must be ProfileIntensityBasis")
        if not isinstance(self.source_lines, (PowderLineSet, CorrectedPowderLineSet)):
            raise TypeError("source_lines must be a powder line set")
        if not isinstance(self.provenance, PowderProfileProvenance):
            raise TypeError("provenance must be PowderProfileProvenance")
        if self.provenance.intensity_basis is not self.intensity_basis:
            raise ValueError("profile provenance uses another intensity basis")


__all__ = [
    "CalculatedProfile",
    "ConstantWidthProfile",
    "IsotropicSampleBroadening",
    "PowderProfileProvenance",
    "ProfileIntensityBasis",
    "TchProfile",
    "UniformTwoThetaGrid",
]
