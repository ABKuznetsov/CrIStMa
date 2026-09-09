"""Immutable models for bounded orbit-first molecular ring analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from cristma.crystallography import PeriodicSymmetryRelation, identity_relation
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic, Severity

from .models import ResolutionStatus


@dataclass(frozen=True, slots=True)
class MolecularRingSearchPolicy:
    """Explicit resource bounds for molecular shortest-return searches."""

    maximum_ring_size: int = 12
    maximum_states_per_bond: int = 50_000
    maximum_paths_per_bond: int = 4_096

    def __post_init__(self) -> None:
        for name in (
            "maximum_ring_size",
            "maximum_states_per_bond",
            "maximum_paths_per_bond",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_ring_size < 3:
            raise ValueError("maximum_ring_size must be at least three")

    def get_config(self) -> dict[str, int]:
        return {
            "maximum_paths_per_bond": self.maximum_paths_per_bond,
            "maximum_ring_size": self.maximum_ring_size,
            "maximum_states_per_bond": self.maximum_states_per_bond,
        }

    def clone(self, **changes: object) -> "MolecularRingSearchPolicy":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, order=True)
class MolecularAtomOrbitRef:
    """One physical molecular vertex in an exact lifted quotient graph."""

    independent_site_id: str
    periodic_relation: PeriodicSymmetryRelation

    def __post_init__(self) -> None:
        if not isinstance(self.independent_site_id, str) or not self.independent_site_id:
            raise ValueError("molecular atom-orbit reference ID must not be empty")


@dataclass(frozen=True, slots=True)
class MolecularRingOrbit:
    """One symmetry-unique chordless ring inside a discrete molecule orbit."""

    ring_orbit_id: str
    parent_component_orbit_id: str
    atom_refs: tuple[MolecularAtomOrbitRef, ...]
    bond_orbit_ids: tuple[str, ...]
    multiplicity_in_reference_cell: int
    centroid_cartesian: tuple[float, float, float]
    normal_cartesian: tuple[float, float, float] | None
    planarity_rms_angstrom: float | None
    status: ResolutionStatus
    diagnostics: tuple[Diagnostic, ...] = ()
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.ring_orbit_id or not self.parent_component_orbit_id:
            raise ValueError("molecular ring identities must not be empty")
        if len(self.atom_refs) < 3 or len(set(self.atom_refs)) < 3:
            raise ValueError("molecular ring requires three distinct physical vertices")
        if len(self.bond_orbit_ids) != len(self.atom_refs):
            raise ValueError("molecular ring requires one bond per atom reference")
        if any(not isinstance(value, str) or not value for value in self.bond_orbit_ids):
            raise ValueError("molecular ring bond IDs must not be empty")
        if isinstance(self.multiplicity_in_reference_cell, bool) or not isinstance(
            self.multiplicity_in_reference_cell, int
        ):
            raise TypeError("molecular ring multiplicity must be an integer")
        if self.multiplicity_in_reference_cell <= 0:
            raise ValueError("molecular ring multiplicity must be positive")
        if len(self.centroid_cartesian) != 3 or not all(
            math.isfinite(value) for value in self.centroid_cartesian
        ):
            raise ValueError("molecular ring centroid must contain three finite values")
        geometry_missing = (
            self.normal_cartesian is None or self.planarity_rms_angstrom is None
        )
        if (self.normal_cartesian is None) != (self.planarity_rms_angstrom is None):
            raise ValueError("molecular ring normal and planarity must be present together")
        if geometry_missing:
            if self.status is ResolutionStatus.RESOLVED:
                raise ValueError("resolved ring requires non-degenerate geometry")
            if not self.diagnostics:
                raise ValueError("ring without geometry requires a diagnostic")
            return
        normal = self.normal_cartesian
        if len(normal) != 3 or not all(math.isfinite(value) for value in normal):
            raise ValueError("molecular ring normal must contain three finite values")
        norm = math.sqrt(math.fsum(value * value for value in normal))
        if not math.isclose(norm, 1.0, abs_tol=1e-12):
            raise ValueError("molecular ring normal must be a unit vector")
        if not math.isfinite(self.planarity_rms_angstrom) or self.planarity_rms_angstrom < 0:
            raise ValueError("molecular ring planarity RMS must be finite and non-negative")

    @property
    def size(self) -> int:
        return len(self.atom_refs)


@dataclass(frozen=True, slots=True)
class _Step:
    bond_orbit_id: str
    source: MolecularAtomOrbitRef
    target: MolecularAtomOrbitRef


@dataclass(frozen=True, slots=True)
class _Path:
    states: tuple[MolecularAtomOrbitRef, ...]
    steps: tuple[_Step, ...]


def _steps_from(state, bonds, context) -> tuple[_Step, ...]:
    steps: list[_Step] = []
    for bond in bonds:
        if bond.first_independent_site_id == state.independent_site_id:
            target = MolecularAtomOrbitRef(
                bond.second_independent_site_id,
                state.periodic_relation.compose(bond.periodic_relation, context),
            )
            steps.append(_Step(bond.bond_orbit_id, state, target))
        if bond.second_independent_site_id == state.independent_site_id:
            target = MolecularAtomOrbitRef(
                bond.first_independent_site_id,
                state.periodic_relation.compose(
                    bond.periodic_relation.inverse(context), context
                ),
            )
            steps.append(_Step(bond.bond_orbit_id, state, target))
    return tuple(sorted(steps, key=lambda item: (item.bond_orbit_id, item.target)))


def _backtrack(predecessors, source, target, maximum_paths):
    paths: list[_Path] = []
    truncated = False

    def visit(state, reversed_steps):
        nonlocal truncated
        if len(paths) >= maximum_paths:
            truncated = True
            return
        if state == source:
            steps = tuple(reversed(reversed_steps))
            paths.append(_Path((source, *(item.target for item in steps)), steps))
            return
        for step in predecessors.get(state, ()):
            visit(step.source, (*reversed_steps, step))
            if truncated:
                return

    visit(target, ())
    paths.sort(key=lambda item: item.states)
    return tuple(paths), truncated


def _shortest_return_paths(bonds, removed, context, policy):
    source = MolecularAtomOrbitRef(
        removed.first_independent_site_id,
        identity_relation(context),
    )
    target = MolecularAtomOrbitRef(
        removed.second_independent_site_id,
        removed.periodic_relation,
    )
    distance = {source: 0}
    predecessors: dict[MolecularAtomOrbitRef, list[_Step]] = {}
    queue = deque((source,))
    target_depth = None
    while queue:
        current = queue.popleft()
        depth = distance[current]
        if target_depth is not None and depth >= target_depth:
            continue
        if depth >= policy.maximum_ring_size - 1:
            continue
        for step in _steps_from(current, bonds, context):
            if step.bond_orbit_id == removed.bond_orbit_id and (
                (step.source == source and step.target == target)
                or (step.source == target and step.target == source)
            ):
                continue
            candidate_depth = depth + 1
            known = distance.get(step.target)
            if known is None:
                if len(distance) >= policy.maximum_states_per_bond:
                    return (), False, "maximum_states_per_bond"
                distance[step.target] = candidate_depth
                predecessors[step.target] = [step]
                queue.append(step.target)
                if step.target == target:
                    target_depth = candidate_depth
            elif known == candidate_depth:
                predecessors.setdefault(step.target, []).append(step)
    if target_depth is None:
        return (), True, None
    paths, truncated = _backtrack(
        predecessors,
        source,
        target,
        policy.maximum_paths_per_bond,
    )
    return paths, not truncated, "maximum_paths_per_bond" if truncated else None


def _joins(bond, first, second, context) -> bool:
    return (
        bond.first_independent_site_id == first.independent_site_id
        and bond.second_independent_site_id == second.independent_site_id
        and first.periodic_relation.compose(bond.periodic_relation, context)
        == second.periodic_relation
    ) or (
        bond.second_independent_site_id == first.independent_site_id
        and bond.first_independent_site_id == second.independent_site_id
        and first.periodic_relation.compose(
            bond.periodic_relation.inverse(context), context
        ) == second.periodic_relation
    )


def _is_chordless(refs, bonds, context) -> bool:
    count = len(refs)
    for first in range(count):
        for second in range(first + 1, count):
            if second == first + 1 or (first == 0 and second == count - 1):
                continue
            if any(_joins(bond, refs[first], refs[second], context) for bond in bonds):
                return False
    return True


def _lattice_normalized_tokens(refs, bond_ids, context):
    shift = tuple(-value for value in refs[0].periodic_relation.lattice_translation)
    translation = PeriodicSymmetryRelation(context.identity_operation_key, shift)
    normalized = tuple(
        MolecularAtomOrbitRef(
            ref.independent_site_id,
            translation.compose(ref.periodic_relation, context),
        )
        for ref in refs
    )
    return tuple(
        (
            ref.independent_site_id,
            ref.periodic_relation.operation_key,
            ref.periodic_relation.lattice_translation,
            bond_id,
        )
        for ref, bond_id in zip(normalized, bond_ids, strict=True)
    )


def _orientation_keys(refs, bond_ids, context):
    count = len(refs)
    keys = []
    for direction in (1, -1):
        for start in range(count):
            indices = tuple((start + direction * step) % count for step in range(count))
            ordered_refs = tuple(refs[index] for index in indices)
            ordered_bonds = tuple(
                bond_ids[index if direction == 1 else (index - 1) % count]
                for index in indices
            )
            keys.append(_lattice_normalized_tokens(ordered_refs, ordered_bonds, context))
    return tuple(keys)


def _canonical_key(refs, bond_ids, component, context):
    from .intermolecular import molecular_component_stabilizer

    candidates = []
    for action in molecular_component_stabilizer(component, context):
        transformed = tuple(
            MolecularAtomOrbitRef(
                ref.independent_site_id,
                action.compose(ref.periodic_relation, context),
            )
            for ref in refs
        )
        candidates.extend(_orientation_keys(transformed, bond_ids, context))
    return min(candidates)


def _refs_from_key(key):
    return tuple(
        MolecularAtomOrbitRef(
            site_id,
            PeriodicSymmetryRelation(operation_key, translation),
        )
        for site_id, operation_key, translation, _ in key
    )


def _multiplicity(refs, bond_ids, component, context):
    images = set()
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        transformed = tuple(
            MolecularAtomOrbitRef(
                ref.independent_site_id,
                action.compose(ref.periodic_relation, context),
            )
            for ref in refs
        )
        images.add(min(_orientation_keys(transformed, bond_ids, context)))
    return len(images)


def _position(site, relation, context):
    operation = context.operation_by_key(relation.operation_key)
    coordinates = tuple(float(item.value) for item in site.fractional)
    return np.asarray(tuple(
        math.fsum(
            float(coefficient) * coordinate
            for coefficient, coordinate in zip(row, coordinates, strict=True)
        )
        + float(offset)
        + relation.lattice_translation[index]
        for index, (row, offset) in enumerate(
            zip(operation.rotation, operation.translation, strict=True)
        )
    ))


def _geometry(refs, structure, context):
    sites = {site.id: site for site in structure.sites}
    fractional = np.asarray(tuple(
        _position(sites[ref.independent_site_id], ref.periodic_relation, context)
        for ref in refs
    ))
    cartesian = fractional @ structure.cell.matrix
    centroid = cartesian.mean(axis=0)
    centered = cartesian - centroid
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= 1e-12:
        return tuple(float(value) for value in centroid), None, None
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-15:
        return tuple(float(value) for value in centroid), None, None
    normal = normal / norm
    for value in normal:
        if abs(float(value)) <= 1e-15:
            continue
        if value < 0:
            normal = -normal
        break
    distances = centered @ normal
    rms = float(np.sqrt(np.mean(distances * distances)))
    return (
        tuple(float(value) for value in centroid),
        tuple(float(value) for value in normal),
        rms,
    )


def find_molecular_ring_orbits(
    structure,
    context,
    bond_orbits,
    component_orbits,
    policy: MolecularRingSearchPolicy,
) -> tuple[tuple[MolecularRingOrbit, ...], tuple[Diagnostic, ...]]:
    """Return bounded chordless shortest-return rings for discrete molecules."""

    found = {}
    diagnostics: list[Diagnostic] = []
    bond_by_id = {item.bond_orbit_id: item for item in bond_orbits}
    for component in component_orbits:
        if component.kind.value != "discrete_molecule":
            continue
        bonds = tuple(bond_by_id[item] for item in component.bond_orbit_ids)
        for removed in bonds:
            paths, complete, limit = _shortest_return_paths(
                bonds, removed, context, policy
            )
            if not complete:
                diagnostics.append(Diagnostic(
                    Severity.WARNING,
                    "crystal_chemistry.molecular_rings.search_limit_reached",
                    f"Molecular ring search reached {limit} at {removed.bond_orbit_id}",
                    recovery="increase the explicit MolecularRingSearchPolicy limit",
                ))
            for path in paths:
                refs = path.states
                bond_ids = tuple(
                    item.bond_orbit_id for item in path.steps
                ) + (removed.bond_orbit_id,)
                if (
                    len(refs) < 3
                    or len(set(refs)) != len(refs)
                    or not _is_chordless(refs, bonds, context)
                ):
                    continue
                key = _canonical_key(refs, bond_ids, component, context)
                found.setdefault(key, component)

    rings: list[MolecularRingOrbit] = []
    for key, component in sorted(found.items()):
        refs = _refs_from_key(key)
        bond_ids = tuple(item[3] for item in key)
        centroid, normal, rms = _geometry(refs, structure, context)
        ring_diagnostics = ()
        status = ResolutionStatus.RESOLVED
        if normal is None:
            diagnostic = Diagnostic(
                Severity.WARNING,
                "crystal_chemistry.molecular_rings.degenerate_geometry",
                "Molecular graph ring has degenerate Cartesian geometry",
            )
            ring_diagnostics = (diagnostic,)
            diagnostics.append(diagnostic)
            status = ResolutionStatus.INCOMPLETE
        elif any(
            bond_by_id[item].status is ResolutionStatus.INCOMPLETE
            for item in bond_ids
        ):
            status = ResolutionStatus.INCOMPLETE
        elif any(
            bond_by_id[item].status is ResolutionStatus.AMBIGUOUS
            for item in bond_ids
        ):
            status = ResolutionStatus.AMBIGUOUS
        ring_id = "molecular-ring-orbit:" + _digest({
            "symmetry_action": context.symmetry_action_fingerprint,
            "parent_component_orbit_id": component.component_orbit_id,
            "cycle": key,
        })
        rings.append(MolecularRingOrbit(
            ring_id,
            component.component_orbit_id,
            refs,
            bond_ids,
            _multiplicity(refs, bond_ids, component, context),
            centroid,
            normal,
            rms,
            status,
            ring_diagnostics,
            (("method", "cristma.molecular_ring_orbits:1"),),
        ))
    return (
        tuple(sorted(rings, key=lambda item: item.ring_orbit_id)),
        tuple(dict.fromkeys(diagnostics)),
    )


__all__ = [
    "MolecularAtomOrbitRef",
    "MolecularRingOrbit",
    "MolecularRingSearchPolicy",
    "find_molecular_ring_orbits",
]
