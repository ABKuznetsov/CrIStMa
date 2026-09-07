"""Resource-bounded calculation of sampled powder profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .powder_correction_models import CorrectedPowderLineSet
from .powder_models import PowderLineSet
from .profile_models import (
    CalculatedProfile,
    ConstantWidthProfile,
    IsotropicSampleBroadening,
    PowderProfileProvenance,
    ProfileIntensityBasis,
    TchProfile,
    UniformTwoThetaGrid,
    _tch_width_and_mixing,
)


_LOCAL_WINDOW_FWHM = 25.0
_GAUSSIAN_NORMALIZATION = math.sqrt(math.pi / (4.0 * math.log(2.0)))


class PowderProfileLimitError(RuntimeError):
    """The requested output grid exceeds an explicit resource limit."""

    def __init__(self, requested_points: int, max_points: int) -> None:
        super().__init__(
            f"profile grid has {requested_points} points; limit is {max_points}"
        )
        self.requested_points = requested_points
        self.max_points = max_points


@dataclass(frozen=True, slots=True)
class PowderProfileCalculator:
    """Apply explicit instrument and optional sample broadening on a grid."""

    max_points: int | None = 1_000_000

    def __post_init__(self) -> None:
        if self.max_points is not None and (
            isinstance(self.max_points, bool)
            or not isinstance(self.max_points, int)
            or self.max_points <= 0
        ):
            raise ValueError("max_points must be a positive integer or None")

    def get_config(self) -> dict[str, int | None]:
        return {"max_points": self.max_points}

    def clone(self, **changes: Any) -> "PowderProfileCalculator":
        unknown = set(changes) - {"max_points"}
        if unknown:
            raise TypeError(f"unknown configuration field: {sorted(unknown)[0]}")
        return replace(self, **changes)

    def calculate(
        self,
        lines: PowderLineSet | CorrectedPowderLineSet,
        grid: UniformTwoThetaGrid,
        broadening: ConstantWidthProfile | TchProfile,
        *,
        zero_shift_deg: float = 0.0,
        sample_broadening: IsotropicSampleBroadening | None = None,
    ) -> CalculatedProfile:
        if not isinstance(lines, (PowderLineSet, CorrectedPowderLineSet)):
            raise TypeError("lines must be PowderLineSet or CorrectedPowderLineSet")
        if not isinstance(grid, UniformTwoThetaGrid):
            raise TypeError("grid must be UniformTwoThetaGrid")
        if not isinstance(broadening, (ConstantWidthProfile, TchProfile)):
            raise TypeError("broadening must be ConstantWidthProfile or TchProfile")
        if sample_broadening is not None and not isinstance(
            sample_broadening, IsotropicSampleBroadening
        ):
            raise TypeError(
                "sample_broadening must be IsotropicSampleBroadening or None"
            )
        if isinstance(zero_shift_deg, bool) or not isinstance(
            zero_shift_deg, (int, float)
        ):
            raise TypeError("zero_shift_deg must be a real number")
        shift = float(zero_shift_deg)
        if not math.isfinite(shift):
            raise ValueError("zero_shift_deg must be finite")

        requested_points = grid.point_count
        if self.max_points is not None and requested_points > self.max_points:
            raise PowderProfileLimitError(requested_points, self.max_points)

        angles = np.asarray(grid.values, dtype=float)
        profile = np.zeros(angles.size, dtype=float)
        if isinstance(lines, CorrectedPowderLineSet):
            source_lines = lines.lines
            powder_lines = lines.powder_lines.lines
            basis = ProfileIntensityBasis.CORRECTED
            intensities = tuple(item.corrected_line_intensity for item in source_lines)
        else:
            source_lines = lines.lines
            powder_lines = source_lines
            basis = ProfileIntensityBasis.INTRINSIC
            intensities = tuple(item.intrinsic_line_intensity for item in source_lines)

        contributed = 0
        for line, powder_line, line_intensity in zip(
            source_lines,
            powder_lines,
            intensities,
            strict=True,
        ):
            if line_intensity == 0.0:
                continue
            original_center = line.two_theta_deg
            center = original_center + shift
            if isinstance(broadening, ConstantWidthProfile):
                fwhm = broadening.fwhm_deg
                mixing = 0.0
            else:
                fwhm, mixing = broadening._width_and_mixing_at(original_center)
            if sample_broadening is not None:
                if isinstance(broadening, ConstantWidthProfile):
                    instrument_gaussian = broadening.fwhm_deg
                    instrument_lorentzian = 0.0
                else:
                    instrument_gaussian, instrument_lorentzian = (
                        broadening._component_widths_at(original_center)
                    )
                sample_gaussian, sample_lorentzian = (
                    sample_broadening.component_fwhm_deg_at(
                        original_center,
                        powder_line.wavelength_angstrom,
                    )
                )
                fwhm, mixing = _tch_width_and_mixing(
                    math.hypot(instrument_gaussian, sample_gaussian),
                    instrument_lorentzian + sample_lorentzian,
                )
            radius = _LOCAL_WINDOW_FWHM * fwhm
            raw_left = math.ceil(
                (center - radius - grid.start_deg) / grid.step_deg
            )
            raw_right = (
                math.floor((center + radius - grid.start_deg) / grid.step_deg) + 1
            )
            left = max(0, raw_left)
            right = min(angles.size, raw_right)
            if left >= right:
                continue
            offsets = angles[left:right] - center
            peak = _pseudo_voigt_density(offsets, fwhm, mixing)
            full_offsets = (
                grid.start_deg
                + np.arange(raw_left, raw_right, dtype=float) * grid.step_deg
                - center
            )
            sampled_window_area = float(
                np.sum(_pseudo_voigt_density(full_offsets, fwhm, mixing))
                * grid.step_deg
            )
            if sampled_window_area <= 0.0 or not math.isfinite(sampled_window_area):
                raise RuntimeError("profile kernel has no finite sampled area")
            profile[left:right] += peak * (line_intensity / sampled_window_area)
            contributed += 1

        model_id = (
            "constant_fwhm_gaussian"
            if isinstance(broadening, ConstantWidthProfile)
            else "tch_pseudo_voigt"
        )
        if sample_broadening is not None:
            model_id += "+isotropic_sample_tch"
        provenance = PowderProfileProvenance(
            method="cristma.diffraction.PowderProfileCalculator",
            version="2",
            broadening_model=model_id,
            broadening_scope=(
                "instrument_only"
                if sample_broadening is None
                else "instrument_and_isotropic_sample"
            ),
            instrument_profile=broadening,
            intensity_basis=basis,
            zero_shift_deg=shift,
            local_window_fwhm=_LOCAL_WINDOW_FWHM,
            max_points=self.max_points,
            lines_considered=len(source_lines),
            lines_contributed=contributed,
            sample_broadening=sample_broadening,
        )
        return CalculatedProfile(
            two_theta_deg=tuple(float(value) for value in angles),
            intensity=tuple(float(value) for value in profile),
            intensity_basis=basis,
            source_lines=lines,
            provenance=provenance,
        )


def _pseudo_voigt_density(
    offsets: np.ndarray,
    fwhm: float,
    mixing: float,
) -> np.ndarray:
    gaussian = np.exp(-4.0 * math.log(2.0) * (offsets / fwhm) ** 2)
    gaussian /= fwhm * _GAUSSIAN_NORMALIZATION
    half_width = fwhm / 2.0
    lorentzian = half_width / (math.pi * (offsets**2 + half_width**2))
    return (1.0 - mixing) * gaussian + mixing * lorentzian


__all__ = ["PowderProfileCalculator", "PowderProfileLimitError"]
