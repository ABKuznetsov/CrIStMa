# Isotropic Sample Broadening Design

## Scope

Add one optional, phase-local sample contribution to the existing calculated
powder profile. CrIStMa performs only a forward calculation. It does not fit
parameters, inspect experimental data, read instrument projects, identify
phases, or know about consuming applications.

## Public contract

```python
sample = IsotropicSampleBroadening(
    crystallite_size_nm=80.0,
    microstrain=8.0e-4,
    scherrer_constant=0.9,
)

profile = PowderProfileCalculator().calculate(
    phase_lines,
    grid,
    instrument_profile,
    sample_broadening=sample,
)
```

One `PowderLineSet` represents one phase calculation. Consumers calculate each
phase separately, so no phase mapping or multi-phase state belongs in CrIStMa.
The existing call without `sample_broadening` remains instrument-only.

## Numerical convention

For each line, with Bragg angle `theta`, wavelength `lambda` in angstrom and
coherent-domain size `L` in angstrom:

```text
H_L,size = K lambda / (L cos(theta))
H_G,strain = 4 microstrain tan(theta)
```

Both equations produce a two-theta FWHM in radians and are converted to
degrees. Isotropic size is treated as Lorentzian and isotropic microstrain as
Gaussian. Contributions combine with the selected instrument profile as:

```text
H_G,total = sqrt(H_G,instrument^2 + H_G,strain^2)
H_L,total = H_L,instrument + H_L,size
```

The existing Thompson-Cox-Hastings approximation converts these components to
one normalized pseudo-Voigt kernel. A constant-width instrument is a Gaussian
instrument component. No default sample broadening is implied.

## Validation and provenance

- `crystallite_size_nm` is absent or finite and positive.
- `microstrain` is absent or finite and non-negative.
- `scherrer_constant` is finite and positive.
- At least one effective sample contribution must be present.
- Every wavelength comes from its own `PowderLine`, so multi-component spectra
  receive component-specific size broadening.
- Profile provenance stores the immutable sample model and distinguishes
  `instrument_only` from `instrument_and_isotropic_sample`.

Preferred orientation, anisotropic size/strain, distributions, fitting,
background, absorption, and experimental matching remain out of scope.

