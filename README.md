# CrIStMa

**Crystallographic Infrastructure for Structures and Materials.**

A compact, physics-first Python library for crystallography, crystal chemistry,
and periodic structure analysis.

[![Python](https://img.shields.io/badge/Python-3.11%2B-1479b8)](https://www.python.org/)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-777777)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-55a630)](https://github.com/ABKuznetsov/CrIStMa/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-Beta-e6a700)](https://pypi.org/project/cristma/)

## Overview

CrIStMa provides a common scientific foundation for programs that work with
crystal and molecular structures. It reads widely used structural formats,
maps them into one canonical model, and offers independent tools for symmetry,
geometry, crystal chemistry, and periodic topology.

The project exists because scientific logic is often coupled to a particular
file parser, graphical application, or large external framework. That makes
calculations difficult to reuse, compare, and audit. CrIStMa keeps these layers
separate: formats end at the I/O boundary, scientific operations receive
explicit inputs, and results retain diagnostics and provenance.

CrIStMa is not an end-user application and does not prescribe a workflow. It
is a reusable scientific library for scripts, notebooks, research software,
desktop applications, and automated data-processing systems.

## What it can do

- read CIF, SHELX RES/INS, VASP, PDB, XYZ, and extXYZ structures through one
  content-aware API;
- preserve source documents where supported or write a normalized structure;
- represent periodic crystals and non-periodic molecules as distinct physical
  models;
- expand crystallographic sites using exact symmetry operations;
- use a bundled catalog of all 530 Hall settings and their Wyckoff positions;
- build symmetry orbits and assign Wyckoff positions;
- generate reciprocal reflections down to a physical `d_min`, including exact
  systematic absences, crystallographic multiplicity, and Friedel relations;
- calculate neutral-atom X-ray structure factors `F`, `|F|`, and `|F|²` from
  independent crystallographic sites and an explicit Hall setting;
- calculate intrinsic powder-line angles and multiplicity-weighted strengths
  for explicit single- or multi-component radiation, including selectable Cr,
  Fe, Co, Cu, Mo, and Ag Kα1/Kα2 sources and monochromatic synchrotron X-rays;
- apply explicit Bragg–Brentano Lorentz and X-ray polarization corrections;
- calculate finite and periodic neighbour graphs and coordination
  environments;
- analyze composition, oxidation-state evidence, coordination shells, and
  coordination polyhedra;
- return an explicit crystal-chemistry resolution status and stable
  symmetry-equivalent contact orbits for downstream applications;
- assemble structural units and classify periodic blocks as finite units,
  chains, layers, or frameworks;
- find translation-aware finite rings in periodic structural
  representations;
- report recoverable problems as structured diagnostics instead of hiding
  assumptions or silently changing the input.

## Scientific model

All supported formats converge on the same native structures:

```text
CIF / RES / INS / POSCAR / XDATCAR / OUTCAR / vasprun.xml / PDB / XYZ / extXYZ
                                |
                                v
             CrystalStructure | MolecularStructure
                                |
                                v
 symmetry / geometry / chemistry / periodic topology / reciprocal reflections
```

The canonical structure is the source of truth for calculations. Parsed
documents remain available for source preservation and provenance, but
file-specific details do not control downstream scientific semantics.

Calculated objects are immutable results rather than hidden application state.
The caller decides calculation order, caching, storage, presentation, and user
interaction.

## Installation

CrIStMa requires Python 3.11 or newer.

Install the public beta from PyPI:

```bash
python -m pip install --pre cristma
```

To install the current source checkout:

```bash
git clone https://github.com/ABKuznetsov/CrIStMa.git
cd CrIStMa
python -m pip install -e .
```

NumPy is the only runtime dependency. Optional development and reference-data
dependencies are kept outside the scientific runtime.

## Quick start

```python
import cristma
from cristma.geometry import CoordinationAnalyzer, NeighborFinder
from cristma.symmetry import expand_structure

result = cristma.read("sample.cif")

for diagnostic in result.diagnostics:
    print(diagnostic.severity.value, diagnostic.code, diagnostic.message)

if not result.ok or not result.structures:
    raise RuntimeError("The structure could not be read")

crystal = result.structures.primary or result.structures[0]
view = expand_structure(crystal)
neighbors = NeighborFinder(cutoff=3.0).find(view)
coordination = CoordinationAnalyzer().analyze(view, neighbors)

print(crystal.cell.volume)
print(len(view.atoms), len(neighbors.edges))
print(len(coordination.environments))
```

The same entry point reads other supported formats:

```python
structure = cristma.read("POSCAR").structures[0]
trajectory = cristma.read("XDATCAR").structures
model = cristma.read("molecule.pdb").structures[0]
```

## Native structure I/O

| Format | Reading | Writing | Notes |
| --- | --- | --- | --- |
| CIF 1.1 | Yes | Preserve and canonical | Source order, comments, unknown tags, and numeric text can be retained |
| SHELX RES/INS | Yes | Preserve and canonical | Canonical output requires an explicit wavelength |
| VASP POSCAR/CONTCAR | Yes | — | Selective Dynamics and reported velocities are retained |
| VASP XDATCAR | Yes | — | Frames are indexed and loaded lazily |
| VASP OUTCAR | Structural frames | — | Per-atom forces and units are retained |
| `vasprun.xml` | Structural frames | — | Trajectory-oriented structural parsing |
| PDB | Yes | — | Crystal and molecular coordinate models |
| XYZ/extXYZ | Yes | — | Typed properties and lazy trajectories |

CrIStMa implements these readers natively. Gemmi, pymatgen, PyXtal, CrysPy,
GSAS-II, SHELX, and graphical frameworks are not required at runtime.

Recoverable source-property problems do not discard otherwise usable
coordinates. CIF occupancies outside their physical `[0, 1]` interval are
retained in their reported raw form, normalized to the nearest bound for
calculation, and reported by a warning diagnostic. Overfilled coincident mixed
positions are normalized proportionally. Fractional coordinates are periodic
and are therefore not clamped to `[0, 1]`; reported ADP values likewise keep
their own physical convention and are not treated as occupancies. Atom-site
loops are mapped by tag name rather than column position. A non-standard
trailing `t` attached to a fractional symmetry translation (for example
`z+1/2t`) is removed only after strict parsing fails, and every repaired
operation produces a warning with the exact recovered expression.

## Reflection generation

The first diffraction layer generates complete reciprocal-space reflection
orbits without requiring an atomic structure. It accepts an explicit unit cell,
one unambiguous catalog `SpaceGroupSetting`, and a resolution limit:

```python
from cristma.crystallography import SpaceGroupCatalog
from cristma.diffraction import ReflectionGenerator

setting = SpaceGroupCatalog.default().by_setting(523)
reflection_set = ReflectionGenerator().generate(
    cell=crystal.cell,
    space_group=setting,
    d_min=0.8,
)

allowed = reflection_set.allowed
absent = reflection_set.systematically_absent
```

Systematic absences are derived from exact symmetry-operation phases, not from
group-name heuristics or expected-reflection tables.

## Orbit-first crystal chemistry

Direct-space crystal chemistry starts from independent sites and an explicit,
validated symmetry context. Symmetry orbits are the scientific results;
expanded contact instances are created only for a consumer-requested region.

```python
from cristma.chemistry import ChemistryAnalyzer, Composition
from cristma.crystallography import SymmetryContext
from cristma.crystal_chemistry import (
    ContactAnalyzer,
    PolyhedronOrbitBuilder,
    ReferenceCell,
    ShellResolutionPolicy,
)

context = SymmetryContext.from_definition(crystal.space_group, crystal.cell)
chemistry = ChemistryAnalyzer().analyze(Composition.from_structure(crystal))
result = ContactAnalyzer(
    ShellResolutionPolicy(1.60, 0.01, 0.08, 0.01, 2.0)
).analyze(crystal, context, chemistry.grammar)
polyhedra = PolyhedronOrbitBuilder().build(result)

# Outward-only compatibility/materialization boundary:
reference_cell_contacts = result.materialize_contacts(ReferenceCell())
assert result.contacts == reference_cell_contacts
```

`ContactAnalysisResult` exposes pair orbits, chemically resolved contact
orbits, oriented incidence orbits, coordination-shell alternatives, explicit
status, diagnostics, configuration and provenance. Polyhedron orbits retain
oriented periodic atom references, coordination number, occupancy-weighted
ligand composition, convex-hull faces, bond-length statistics, the Baur
distortion index, edge-angle population dispersion, volume, geometric
centroid and centre offset.

The canonical face signature represents the complete vertex-edge-face
incidence graph. Shape descriptors are not guessed for open or degenerate
shells: unavailable values remain `None` and the reason is diagnostic. CrIStMa
does not triangulate display meshes or define colours, visibility, selection,
tables or motif-comparison policy; those are responsibilities of consuming
applications.

## Structural hierarchy

The hierarchy remains on the same finite symmetry quotient graph. No stage
below consumes materialized contacts or an expanded atomic view:

```python
units = StructuralUnitBuilder().build(result, polyhedra)
graph = StructuralGraphBuilder().build(result, polyhedra)
representation = StructuralRepresentationBuilder(selection_policy).build(graph)
connectivity = PeriodicConnectivityAnalyzer().analyze(representation)
blocks = StructuralBlockFinder().find(representation, connectivity)
rings = RingFinder().find(representation, blocks)
```

The results expose stable unit, connection, block and ring orbit identities.
Periodic rank comes from exact affine cycle translations, not geometric
heuristics. `StructuralUnitGeometry` records only calculated points, linear
groups, planar polygons, or closed polyhedra. Rendering meshes, colours,
visibility, tree grouping, matching and interactive comparison remain outside
CrIStMa.

## X-ray structure factors

The first scattering layer calculates forward neutral-atom X-ray amplitudes
from a `CrystalStructure` and its generated `ReflectionSet`:

```python
from cristma.crystallography import (
    SpaceGroupCatalog,
    SpaceGroupSettingResolutionStatus,
    resolve_space_group_setting,
)
from cristma.diffraction import StructureFactorCalculator, XRayScatteringContext

resolution = resolve_space_group_setting(
    crystal.space_group,
    SpaceGroupCatalog.default(),
)
if resolution.status is not SpaceGroupSettingResolutionStatus.RESOLVED:
    raise ValueError("the structure has no unique catalog setting")
setting = resolution.setting

factors = StructureFactorCalculator().calculate(
    structure=crystal,
    space_group=setting,
    reflections=reflection_set,
    context=XRayScatteringContext.default(),
)
```

Setting resolution compares the complete exact operation set in the reported
fractional basis. It returns `AMBIGUOUS` or `UNRESOLVED` instead of selecting a
setting from a Hermann–Mauguin symbol or space-group number alone. The
diffraction calculators themselves still accept only a uniquely resolved
`SpaceGroupSetting`.

The calculator expands independent sites with the supplied exact symmetry,
deduplicates special positions, applies occupancies and isotropic displacement
parameters, and returns `F`, `|F|`, and `|F|²` without multiplying by reflection
multiplicity. The bundled `f0(s)` table covers neutral atoms H through Cf and
does not require xraylib at runtime. Anisotropic displacement parameters and
anomalous scattering are not evaluated directly in v1. When a source reports
`U_aniso` together with `U_iso_or_equiv`, the calculator uses the reported
equivalent isotropic value and returns an explicit warning. Without that value,
the position is calculated with `T=1` and the result carries an error diagnostic
plus the affected site IDs instead of aborting the complete calculation.

This is forward crystallographic physics only. Experimental peak matching,
similarity measures, R-factors, powder corrections, and phase identification
belong to consuming applications or later independent layers.

## Powder diffraction lines

The next layer groups Friedel mates and calculates a separate Bragg angle for
every component of an explicit radiation spectrum:

```python
from cristma.diffraction import (
    BraggBrentanoGeometry,
    PowderCorrectionCalculator,
    PowderLineCalculator,
    RadiationSpectrum,
)

powder_lines = PowderLineCalculator().calculate(
    structure_factors=factors,
    spectrum=RadiationSpectrum.lab_k_alpha("Cu"),
)

corrected = PowderCorrectionCalculator().calculate(
    powder_lines,
    BraggBrentanoGeometry(),
)

for line in corrected.lines_by_angle:
    print(line.two_theta_deg, line.corrected_line_intensity)
```

Packaged laboratory sources are selectable for Cr, Fe, Co, Cu, Mo, and Ag.
Every preset retains Kα1 and Kα2 as separate lines, so their angular separation
remains visible at high angles. A synchrotron wavelength is supplied explicitly:

```python
spectrum = RadiationSpectrum.synchrotron(
    wavelength_angstrom=0.41328,
    source_id="beamline:my-experiment",
)
geometry = BraggBrentanoGeometry(
    perpendicular_polarization_fraction=0.98,
)
```

`intrinsic_line_intensity` contains only normalized radiation weight,
crystallographic multiplicity, Friedel grouping, and `|F|²`. The separate
correction layer returns the Lorentz factor, polarization factor, and
`corrected_line_intensity` for symmetric Bragg–Brentano reflection geometry.
The default polarization fraction is `0.5`, corresponding to unpolarized
laboratory X-rays; synchrotron polarization must be provided explicitly.
These are relative calculated intensities, not an absolute detector signal.

Preferred orientation, absorption, and comparison with experiment are not
part of this line result. `RadiationProbe`
already distinguishes X-rays from neutrons, but neutron powder intensities
will only be enabled together with a separate nuclear scattering-length
context; CrIStMa never substitutes X-ray atomic form factors for neutrons.

## Calculated powder profile

A minimal final forward layer can place intrinsic or corrected powder lines
on an explicit uniform grid. If no instrument calibration is available, the
caller supplies one constant Gaussian FWHM:

```python
from cristma.diffraction import (
    ConstantWidthProfile,
    PowderProfileCalculator,
    UniformTwoThetaGrid,
)

profile = PowderProfileCalculator(max_points=1_000_000).calculate(
    corrected,
    UniformTwoThetaGrid(5.0, 120.0, 0.01),
    ConstantWidthProfile(fwhm_deg=0.10),
)
```

An instrument profile stored by a consuming application can instead provide
explicit GSAS-style continuous-wave `U`, `V`, `W`, `X`, and `Y` values:

```python
from cristma.diffraction import TchProfile

instrument = TchProfile(u=U, v=V, w=W, x=X, y=Y)
profile = PowderProfileCalculator().calculate(
    corrected,
    grid,
    instrument,
    zero_shift_deg=zero_shift,
)
```

CrIStMa has no built-in instrument preset and does not read refinement project
files. Every peak kernel is area-normalized and evaluated only in a local
window. The configurable `max_points` limit is checked before allocating the
output array; an oversized request raises `PowderProfileLimitError` rather than
returning a truncated scientific result. Profile v1 represents **instrument
broadening only**. Crystallite size,
microstrain, preferred orientation, absorption, background, asymmetry,
experimental matching, and refinement remain separate future layers.

## Design principles

- **Physics before interface.** Scientific meaning is not determined by a GUI
  or storage format.
- **One canonical model.** Every reader produces the same structure types for
  downstream calculations.
- **Explicit assumptions.** Policies, tolerances, limits, and incomplete
  searches are visible in inputs and results.
- **Traceable results.** Symmetry images, reference data, transformations, and
  diagnostics retain provenance.
- **Composable tools.** Calculators are independent and do not rely on a
  hidden current structure or global workflow.
- **Small runtime.** The core depends only on Python and NumPy.

## Beta status

`0.1.0b1` was the first public beta. `0.1.0b2` added the first diffraction
milestone, explicit crystal-chemistry result statuses, and stable
symmetry-equivalent contact orbits. `0.1.0b3` keeps coordinate structures
available when a reported anisotropic displacement tensor conflicts with site
symmetry, and extends composition grammar to hydrogen-omitted organic and
metal-organic structures. The local beta3 development line also adds
neutral-atom X-ray structure factors with explicit scattering provenance. The
`0.1.0b4` keeps a usable coordinate structure when
an anisotropic ADP fails the final orbit-consistency check: it retains a
reported `U_iso_or_equiv` when available, otherwise omits only that ADP, and
returns a warning instead of rejecting the complete CIF. The published
`0.1.0b5` release introduces orbit-first direct-space crystal chemistry:
validated symmetry contexts, asymmetric-unit pair orbits, chemical contact
orbits, oriented incidences, and multiplicity-weighted coordination-shell
orbits without expanded contacts in the scientific pipeline. The local
`0.1.0b6` development line unifies reference-cell atom-image identities across
`atomic_view()`, contact materialization, polyhedra, and structural units. The
implemented scientific core is covered by automated tests and is ready for
evaluation and integration. Until the first stable release, public APIs may
still change when required to correct or clarify scientific contracts.

The current development version adds reciprocal metrics, bounded reflection
generation, exact systematic absences, reciprocal symmetry orbits,
crystallographic multiplicity, Friedel relations, neutral-atom structure
factors, intrinsic multi-component powder lines, selectable X-ray sources, and
Bragg–Brentano Lorentz–polarization corrections, plus minimal instrument-only
calculated profiles on explicit grids, to the published beta's
structural I/O, symmetry, geometry, crystal chemistry, and topology layers. It
does not yet calculate sample broadening or corrections, neutron structure
factors, experimental matching, or structure refinement.

## Roadmap

Planned scientific layers are developed as independent milestones:

1. energy-dependent and additional scattering contexts;
2. additional powder geometries, sample corrections, and explicit instrument
   models;
3. sample-broadening contributions and neutron scattering;
4. additional structural transforms, hierarchy and topology tools, and
   refinement built over the same forward calculations.

The roadmap describes direction, not a compatibility or release-date promise.
CrIStMa will remain independent of any particular consuming application.

## License and reference data

Original CrIStMa code is distributed under the permissive
[BSD-3-Clause license](https://github.com/ABKuznetsov/CrIStMa/blob/main/LICENSE).
It may be used in open-source, commercial, and closed-source software subject
to the license notice requirements.

Bundled reference resources retain their own attribution and provenance:

- space-group and Wyckoff data normalized from pinned spglib 2.7.0 resources
  under BSD-3-Clause;
- Cordero covalent radii compiled from QCElemental resources under
  BSD-3-Clause;
- Shannon radii compiled from a pinned pymatgen artifact under MIT;
- neutral-atom X-ray form factors normalized from pinned xraylib 4.3.0 data
  under its BSD-style license, with EPDL97 recorded as the scientific source;
- selected Crystallography Open Database fixtures under CC0/public-domain
  terms;
- curated chemical-reference rules with their scientific literature recorded
  in the versioned resources.

Versions, commits, hashes, known provenance limitations, and redistribution
requirements are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/ABKuznetsov/CrIStMa/blob/main/THIRD_PARTY_NOTICES.md).

## Author

Artem B. Kuznetsov<br>
[GitHub](https://github.com/ABKuznetsov)
