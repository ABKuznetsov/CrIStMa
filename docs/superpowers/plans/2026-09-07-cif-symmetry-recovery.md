# CIF Symmetry Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover symmetry-constrained rounded cells and CIF atom lists that redundantly contain symmetry-expanded positions, while retaining strict diagnostics and one physical atom image per downstream expansion.

**Architecture:** Centralize reported-number precision in a private core helper, use it to derive a conservative metric tolerance inside `SymmetryContext`, and canonicalize symmetry-equivalent CIF rows after existing site normalization but before constructing `CrystalStructure`. Keep polyhedron geometry defensive so a genuinely unusable hull becomes an incomplete calculated result rather than an exception.

**Tech Stack:** Python 3.11+, immutable dataclasses, exact `Fraction` symmetry operations, NumPy numerical metrics, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-cif-symmetry-recovery-design.md`

## Global Constraints

- CrIStMa remains independent of CRAFT and Finder.
- Symmetry operations and periodic relations remain exact; reported precision affects only numerical coordinate and metric comparisons.
- Identity fallback never triggers symmetry-expanded-row collapse.
- Chemically different, mixed, split, disordered or metadata-conflicting sites are never merged.
- Recoverable source defects produce warnings and provenance; irrecoverable group or metric contradictions remain exceptions.
- Tests and documentation remain excluded from wheel and sdist artifacts.

---

### Task 1: Shared reported-number precision

**Files:**
- Create: `src/cristma/core/precision.py`
- Modify: `src/cristma/io/cif/mapper.py`
- Test: `tests/core/test_precision.py`

**Interfaces:**
- Produces: `reported_numeric_error(value: MeasuredValue, *, sigma_multiplier: float = 3.0) -> float`
- Consumes: `MeasuredValue.raw`, `MeasuredValue.uncertainty`

- [ ] **Step 1: Write failing tests for uncertainty and decimal precision**

```python
from cristma.core.precision import reported_numeric_error
from cristma.core import MeasuredValue


def test_reported_numeric_error_uses_three_sigma_when_available():
    value = MeasuredValue(9.10888, 0.00003, "9.10888(3)")
    assert reported_numeric_error(value) == pytest.approx(0.00009)


def test_reported_numeric_error_uses_half_last_decimal_without_uncertainty():
    value = MeasuredValue(9.108879, None, "9.108879")
    assert reported_numeric_error(value) == pytest.approx(0.0000005)


def test_reported_numeric_error_preserves_exponent_scale():
    value = MeasuredValue(1.25e-3, None, "1.25e-3")
    assert reported_numeric_error(value) == pytest.approx(0.005e-3)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/core/test_precision.py
```

Expected: collection fails because `cristma.core.precision` does not exist.

- [ ] **Step 3: Implement the private precision helper**

Implement strict validation for `sigma_multiplier`, return `sigma_multiplier * uncertainty` when present, otherwise parse the decimal exponent with `Decimal` and return half the last written unit. Missing or nonnumeric raw text returns `0.0`.

- [ ] **Step 4: Replace `_reported_coordinate_error` internals**

Keep `_reported_coordinate_error` as a local compatibility wrapper in `mapper.py`, but delegate to `reported_numeric_error`. This preserves the already verified special-position behavior while removing duplicate precision logic.

- [ ] **Step 5: Verify GREEN and existing coordinate regressions**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/core/test_precision.py tests/io/test_cif_special_positions.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/cristma/core/precision.py src/cristma/io/cif/mapper.py tests/core/test_precision.py
git commit -m "Centralize reported numeric precision"
```

---

### Task 2: Metric compatibility within reported cell precision

**Files:**
- Modify: `src/cristma/crystallography/symmetry_context.py`
- Modify: `tests/orbit_first/test_symmetry_context.py`

**Interfaces:**
- Produces: private `_reported_metric_tolerance(cell: UnitCell) -> float`
- Changes: `SymmetryContext.metric_tolerance` stores the effective tolerance; provenance records configured and reported contributions.
- Produces diagnostic: `symmetry.context.metric_within_reported_precision`

- [ ] **Step 1: Write a failing rounded-cubic-cell test**

Build a cubic-symmetry context using catalog setting `526` (`Fd-3m:2`) and this cell:

```python
UnitCell(
    MeasuredValue(9.10888, 0.00003, "9.10888(3)"),
    MeasuredValue(9.108879, None, "9.108879"),
    MeasuredValue(9.108879, None, "9.108879"),
    MeasuredValue(90.0, None, "90.0"),
    MeasuredValue(90.0, None, "90.0"),
    MeasuredValue(90.0, None, "90.0"),
)
```

Assert that context construction succeeds, its effective tolerance exceeds the configured `1e-8` floor, and its diagnostics contain `symmetry.context.metric_within_reported_precision`.

- [ ] **Step 2: Write a failing outside-precision test**

Use the same raw precision but change one edge to `9.107879`. Assert `SymmetryContextInvariantError.code == "symmetry.context.metric_incompatible"`.

- [ ] **Step 3: Run both tests and verify RED**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/orbit_first/test_symmetry_context.py -k metric
```

Expected: the rounded-cell case fails with `symmetry.context.metric_incompatible`; the outside-precision case already remains strict.

- [ ] **Step 4: Implement effective tolerance and diagnostic**

Calculate:

```python
edge_terms = tuple(reported_numeric_error(v) / float(v.value) for v in edges)
angle_terms = tuple(math.radians(reported_numeric_error(v)) for v in angles)
reported = 4.0 * max((*edge_terms, *angle_terms), default=0.0)
effective = max(configured, reported)
```

Validate the operation group exactly as before and validate metric residuals against `effective`. Add the warning only when the maximum residual exceeds `configured` but not `effective`. Record configured tolerance, reported tolerance, effective tolerance and maximum normalized residual in diagnostic text and context provenance.

- [ ] **Step 5: Verify GREEN and strict group invariants**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/orbit_first/test_symmetry_context.py tests/orbit_first/test_asu_mapping.py
```

Expected: all tests pass, including all 530 catalog-setting ASU invariants.

- [ ] **Step 6: Commit**

```bash
git add src/cristma/crystallography/symmetry_context.py tests/orbit_first/test_symmetry_context.py
git commit -m "Respect reported precision in metric validation"
```

---

### Task 3: Conservative symmetry-expanded site grouping

**Files:**
- Modify: `src/cristma/io/cif/mapper.py`
- Create: `tests/io/test_cif_symmetry_expanded_sites.py`

**Interfaces:**
- Produces: private `_collapse_symmetry_expanded_sites(sites, operations, diagnostics) -> tuple[IndependentSite, ...]`
- Produces diagnostics: `cif.map.symmetry_expanded_sites_collapsed`, `cif.map.symmetry_equivalent_sites_unmerged`
- Stores metadata keys: `symmetry_source_rows`, `symmetry_source_labels`, `symmetry_source_site_ids`

- [ ] **Step 1: Write a failing full-cell-to-ASU test**

Create a minimal CIF with inversion operations and two carbon rows at
`(0.1, 0.2, 0.3)` and `(-0.1, -0.2, -0.3)`. Assert:

```python
assert len(structure.sites) == 1
assert len(structure.atomic_view().atoms) == 2
assert structure.sites[0].metadata["symmetry_source_labels"] == ("C1", "C2")
assert "cif.map.symmetry_expanded_sites_collapsed" in diagnostic_codes
```

- [ ] **Step 2: Write non-merge tests**

Use the same related coordinates but separately vary:

- species `C` versus `N`;
- occupancy `1.0` versus `0.5`;
- explicit disorder groups `1` versus `2`;
- incompatible displacement values.

Assert that both independent sites remain and the unmerged diagnostic is present.

- [ ] **Step 3: Write identity and ordering tests**

Assert that identity-only symmetry never collapses distinct rows, integer-shifted coordinates group identically, and reordering the operation rows does not change grouping or alias order.

- [ ] **Step 4: Run the new file and verify RED**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/io/test_cif_symmetry_expanded_sites.py
```

Expected: the first and ordering tests fail because every input atom row is currently retained as independent.

- [ ] **Step 5: Implement symmetry relation matching**

For every operation, transform the representative coordinates and compare to a candidate modulo integers. Propagate coordinate bounds row-wise:

```python
allowed[row] = candidate_error[row] + sum(
    abs(rotation[row][column]) * representative_error[column]
    for column in range(3)
)
```

Use `max(1e-12, nextafter(allowed[row], inf))`. Return the first matching normalized operation relation for provenance. Keep the first source row as representative.

- [ ] **Step 6: Implement conservative chemical eligibility**

Compare ordered component tuples by species identity and normalized occupancy value. Require both disorder fields to be `None`, non-conflicting Wyckoff/multiplicity values and equal normalized displacement objects. On conflict, retain both rows and emit the unmerged diagnostic.

- [ ] **Step 7: Integrate after site normalization**

Run collapse after special-position and ADP normalization has completed for every row, and before `CrystalStructure` construction. Update the representative metadata with source aliases without changing its scientific occupancy or coordinates.

- [ ] **Step 8: Verify GREEN and ordinary CIF stability**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/io/test_cif_symmetry_expanded_sites.py tests/io tests/integration/test_structure_core.py tests/integration/test_inorganic_crystal_chemistry.py
```

Expected: all tests pass and existing ordinary CIF IDs remain unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/cristma/io/cif/mapper.py tests/io/test_cif_symmetry_expanded_sites.py
git commit -m "Collapse redundant CIF symmetry images"
```

---

### Task 4: Preserve incomplete polyhedra after hull failure

**Files:**
- Modify: `src/cristma/crystal_chemistry/polyhedron_orbits.py`
- Modify: `tests/orbit_first/test_orbit_polyhedra.py`

**Interfaces:**
- Changes: a caught hull/signature `ValueError` clears all dependent geometry before constructing `CoordinationPolyhedron`.
- Preserves: `ResolutionStatus.INCOMPLETE`, `GEOMETRY_DEGENERATE`, `DESCRIPTOR_UNAVAILABLE`.

- [ ] **Step 1: Write a failing duplicate-geometry regression**

Extend the existing `_contact_result` fixture with an identity-symmetry case whose
centre is at `(0.5, 0.5, 0.5)` and whose ligand sites lie at the eight cube
corners around the centre, with a ninth ligand site duplicating the first
corner under a different site ID. Use the existing centre–ligand grammar and a
shell policy that selects all nine equal-distance incidences. Assert
`PolyhedronOrbitBuilder().build(result)` returns an incomplete polyhedron with
empty faces, `face_signature is None`, and `GEOMETRY_DEGENERATE` rather than
raising `ValueError`. The nine-point local array is therefore equivalent to:

```python
cube = tuple(product((-1.0, 1.0), repeat=3))
local_vertices = (*cube, cube[0])
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/orbit_first/test_orbit_polyhedra.py -k hull_failure
```

Expected: `ValueError: polyhedron face graph must be a connected closed manifold` escapes during model construction.

- [ ] **Step 3: Reset dependent geometry in the existing exception path**

Inside `_representative_polyhedron`, set:

```python
faces = ()
face_signature = None
angle_dispersion = None
volume = None
centroid = None
center_offset = None
```

before recording the incomplete status and diagnostics. Do not catch errors outside the established hull/descriptor calculation boundary.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q tests/orbit_first/test_orbit_polyhedra.py
```

Expected: all polyhedron-orbit tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/cristma/crystal_chemistry/polyhedron_orbits.py tests/orbit_first/test_orbit_polyhedra.py
git commit -m "Preserve incomplete polyhedra after hull failure"
```

---

### Task 5: Real-CIF acceptance and release gate

**Files:**
- Modify: `README.md`
- No production API additions beyond Tasks 1–4.

**Interfaces:**
- Verifies: installed public `cristma.read`, `SymmetryContext`, `ContactAnalyzer`, `PolyhedronOrbitBuilder`.

- [ ] **Step 1: Run the three real COD files**

Use the local Finder cache paths and execute the public pipeline for
`2240655.cif`, `5910202.cif`, and `5910213.cif`. Record site counts,
expanded-atom counts, diagnostic codes, contact/shell counts and polyhedron
statuses. Expected: no unhandled exception; `2240655` reports metric recovery;
the two historical expanded lists report site collapse.

- [ ] **Step 2: Run the full CrIStMa suite**

```bash
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pytest -q
```

Expected: zero failures.

- [ ] **Step 3: Run the full Finder/CRAFT corpus externally**

Install the local checkout or wheel as `cristma==0.1.0b6` in the existing CRAFT
acceptance environment and repeat the 346-file report. Expected: the three
CrIStMa errors are eliminated; timeout reporting remains separate.

- [ ] **Step 4: Document recoverable CIF behavior**

Add concise README text explaining that complete symmetry-expanded historical
atom lists are conservatively reduced to an ASU with warnings, and that metric
rounding is accepted only within reported precision.

- [ ] **Step 5: Verify package contents**

Build wheel and sdist in a clean temporary directory. Run `twine check`, install
the wheel into an empty environment, import `cristma`, and verify neither
artifact contains `tests/` or `docs/`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Document recoverable CIF symmetry normalization"
```
