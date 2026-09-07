# Isotropic Sample Broadening Implementation Plan

**Goal:** Add optional phase-local Scherrer size and isotropic microstrain
broadening to the existing resource-bounded powder profile calculator.

**Spec:** `docs/superpowers/specs/2026-09-07-isotropic-sample-broadening-design.md`

### Task 1: Immutable model and analytical widths

- [ ] Add failing validation and analytical-formula tests.
- [ ] Add `IsotropicSampleBroadening` to `profile_models.py`.
- [ ] Return separate Gaussian-strain and Lorentzian-size FWHM values in degrees.
- [ ] Export the model from `cristma.diffraction`.

### Task 2: Combine instrument and sample contributions

- [ ] Add failing profile tests for wider phase peaks and provenance.
- [ ] Refactor TCH component-width combination into one private implementation.
- [ ] Add optional keyword-only `sample_broadening` to `calculate`.
- [ ] Preserve byte-for-byte numerical behavior when it is omitted.
- [ ] Use each source powder line's wavelength for corrected and intrinsic sets.

### Task 3: Documentation and beta6 gate

- [ ] Document the phase-local call and explicit assumptions in `README.md`.
- [ ] Run diffraction tests, the full suite, compilation, and `git diff --check`.
- [ ] Build wheel and sdist; verify metadata and absence of tests/docs.
- [ ] Smoke-install the wheel and exercise the new public API.

