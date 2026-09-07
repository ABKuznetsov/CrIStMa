# CIF symmetry-recovery design

## Purpose

Some valid historical CIF files report coordinates for several or all
symmetry-equivalent atom images while also reporting the complete space-group
operation set. Treating every row as an independent crystallographic site and
expanding every row again duplicates atoms, contacts and coordination-shell
vertices. Other CIF files report symmetry-constrained cell parameters with
slightly different decimal precision, so a fixed metric tolerance can reject a
cell whose differences are entirely explained by the reported precision.

CrIStMa shall recover both cases without hiding the source inconsistency. It
shall retain a usable calculated structure, emit explicit diagnostics and keep
the original reported values in provenance. The recovery belongs to structural
I/O and symmetry validation; no CRAFT or Finder behavior enters the library.

## Scope

This milestone includes:

- recognition of atom rows related by the CIF's validated exact symmetry;
- conservative collapse of redundant symmetry-expanded rows into one
  `IndependentSite`;
- preservation of source-row identities as aliases/provenance;
- metric compatibility bounded by the precision of reported cell parameters;
- non-fatal diagnostics for both recoveries;
- regressions based on COD entries `2240655`, `5910202` and `5910213`.

It does not include:

- merging chemically different, split or disordered positions;
- guessing a space group from coordinates;
- changing coordinates merely to improve a fit;
- repairing an incomplete or mathematically invalid operation group;
- suppressing genuinely incompatible cell metrics;
- changing contact, shell or polyhedron chemistry.

## Symmetry-expanded atom-row recovery

The recovery runs only after CIF syntax parsing, exact symmetry-operation
normalization, coincident mixed-site handling and special-position projection.
The complete operation set must already be mathematically valid. Identity
fallback structures do not use this recovery.

For every unconsumed site, the mapper tests later sites against its complete
periodic symmetry orbit. Two rows belong to the same recoverable group only
when all of the following hold:

1. an exact reported symmetry operation maps one fractional position to the
   other modulo an integer lattice translation within propagated reported
   coordinate precision;
2. their component species and normalized occupancies agree;
3. their disorder assembly and disorder group agree and do not describe
   distinct split alternatives;
4. their reported Wyckoff and multiplicity values do not conflict;
5. their displacement descriptions are absent or equal after the existing
   site-symmetry normalization.

Coordinate comparison uses the uncertainty when explicitly reported and the
half-unit of the last written decimal otherwise. The transformed uncertainty
is propagated through the exact integer rotation. A `1e-12` numerical floor is
the only non-source tolerance.

If any chemical or structural condition is not met, the rows remain separate.
An exact spatial relation with incompatible metadata produces a warning rather
than a merge.

The earliest source row remains the representative so existing source-order
identity remains predictable. Its scientific site data are retained. The
metadata contain every original source row, label and site ID in deterministic
source order. The merged site receives no additional occupancy and is expanded
exactly once by downstream symmetry code.

Successful recovery emits:

```text
cif.map.symmetry_expanded_sites_collapsed
```

with representative site ID, source aliases, operation relations and number of
removed redundant rows. A spatially equivalent but non-mergeable row emits:

```text
cif.map.symmetry_equivalent_sites_unmerged
```

with the conflicting fields.

## Reported-precision metric compatibility

`SymmetryContext` retains its explicit configured metric tolerance as a strict
numerical floor. In addition, it derives a source-bound tolerance from the six
`MeasuredValue` cell parameters:

```text
edge contribution  = reported_error / edge_length
angle contribution = radians(reported_error_deg)
reported_metric_tolerance = 4 * max(all contributions)
effective tolerance = max(configured floor, reported_metric_tolerance)
```

`reported_error` is three standard uncertainties when an uncertainty is
present, otherwise half a unit in the final reported decimal place. Missing raw
precision contributes zero. The factor four conservatively covers the
quadratic metric terms and combined angular contributions; it is part of the
versioned numerical convention.

When every operation preserves the metric only under the source-derived part
of the tolerance, the context is valid and emits:

```text
symmetry.context.metric_within_reported_precision
```

The diagnostic records the configured floor, reported tolerance, effective
tolerance and maximum observed normalized residual. The original cell values
are not averaged or replaced.

If any rotation exceeds the effective tolerance,
`SymmetryContextInvariantError` remains mandatory with code
`symmetry.context.metric_incompatible`. Closure, inverses, integer rotations
and exact translations are never relaxed by cell precision.

## Downstream behavior

Recovered sites are ordinary immutable `IndependentSite` objects. All
downstream calculations consume the canonical site tuple and must not repeat
the collapse. Atomic expansion, pair orbits, contacts, shell orbits,
polyhedra, structural units and rings therefore see every physical atom image
once.

A coordination geometry that is genuinely open or degenerate still produces
an incomplete polyhedron and diagnostics under the existing contract. The
polyhedron layer must not be used to conceal duplicate input atoms.

## Status and provenance

Both recoveries are warnings because the source is imperfect but the
scientific calculation remains complete after a uniquely justified repair.
They do not make contact analysis `INCOMPLETE`.

Canonical site metadata and `SymmetryContext.provenance` retain enough evidence
to reconstruct why recovery was accepted. Fingerprints include the canonical
sites or context diagnostics as they do for other normalized input state.

## Acceptance

- `2240655.cif` creates a valid symmetry context and reports metric recovery;
- a similar cubic cell outside reported precision remains a strict error;
- `5910202.cif` collapses the historical full-cell atom list to the correct
  asymmetric-unit site orbits and completes polyhedron construction;
- `5910213.cif` collapses its redundant rows and completes polyhedron
  construction;
- mixed occupancy, split/disorder and chemically incompatible symmetry-related
  rows are not merged;
- operation ordering and integer-shifted coordinates do not change grouping;
- ordinary asymmetric-unit CIF files retain the same sites and IDs;
- the complete CrIStMa test suite and Finder CIF corpus remain valid.

