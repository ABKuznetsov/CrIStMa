"""Space-group orbits of coordination polyhedra."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from cristma.crystallography import periodic_endpoint_instance
from cristma.crystallography.symmetry_context import _digest
from cristma.diagnostics import Diagnostic, Severity
from cristma.structure import PeriodicAtomRef

from .contact_analysis import ContactAnalysisResult
from .models import ResolutionStatus
from .polyhedra import (
    DESCRIPTOR_UNAVAILABLE,
    GEOMETRY_DEGENERATE,
    MIXED_POSITION,
    PARTIAL_OCCUPANCY,
    CoordinationPolyhedron,
    CoordinationPolyhedronOrbit,
    PolyhedronVertex,
    _bond_length_metrics,
    _convex_hull_faces,
    _edge_angle_dispersion_deg,
    _volume_and_centroid,
    canonical_face_signature,
    unique_hull_edges,
)


Translation = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class PolyhedronOrbitBuildResult:
    polyhedra: tuple[CoordinationPolyhedron, ...]
    polyhedron_orbits: tuple[CoordinationPolyhedronOrbit, ...]
    diagnostics: tuple[Diagnostic, ...]
    complete: bool

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.diagnostics)


def _source_image(contact_result: ContactAnalysisResult, site_id: str):
    site_mapping = contact_result._asymmetric_unit_mapping.by_site_id[site_id]
    matches = tuple(
        image
        for image in site_mapping.reference_cell_images
        if image.equivalent_relations == site_mapping.stabilizer_relations
    )
    if len(matches) != 1:
        raise ValueError("independent centre does not identify one source image")
    return matches[0]


def _image_by_id(contact_result: ContactAnalysisResult, site_id: str, image_id: str):
    return next(
        image
        for image in contact_result._asymmetric_unit_mapping.by_site_id[
            site_id
        ].reference_cell_images
        if image.image_id == image_id
    )


def _local_polyhedron_vertices(
    contact_result: ContactAnalysisResult,
    shell,
    alternative,
) -> tuple[PolyhedronVertex, ...]:
    incidence_by_id = {
        item.incidence_orbit_id: item
        for item in contact_result.contact_incidence_orbits
    }
    center_image = _source_image(contact_result, shell.center_independent_site_id)
    center_fractional = np.asarray(center_image.fractional_position, dtype=float)
    matrix = contact_result._structure.cell.matrix
    vertices: list[PolyhedronVertex] = []
    for incidence_id in alternative.primary_incidence_ids:
        incidence = incidence_by_id[incidence_id]
        for relation in incidence.equivalent_oriented_relations:
            _, image_id, cell_translation = periodic_endpoint_instance(
                incidence.ligand_independent_site_id,
                relation,
                contact_result._asymmetric_unit_mapping,
            )
            image = _image_by_id(
                contact_result,
                incidence.ligand_independent_site_id,
                image_id,
            )
            ligand_fractional = np.asarray(image.fractional_position, dtype=float)
            ligand_fractional += np.asarray(cell_translation, dtype=float)
            local = (ligand_fractional - center_fractional) @ matrix
            distance = float(np.linalg.norm(local))
            vertices.append(
                PolyhedronVertex(
                    PeriodicAtomRef(image_id, cell_translation),
                    incidence.incidence_orbit_id,
                    incidence.resolved_contact_orbit_id,
                    tuple(float(value) for value in local),
                    distance,
                    incidence.effective_neighbor_occupancy,
                )
            )
    ordered = tuple(
        sorted(
            vertices,
            key=lambda item: (
                item.atom_ref.atom_id,
                item.atom_ref.cell_translation,
                item.incidence_orbit_id,
            ),
        )
    )
    if len(ordered) != alternative.geometric_CN:
        raise ValueError("realized polyhedron vertex count disagrees with shell CN")
    if len({item.atom_ref for item in ordered}) != len(ordered):
        raise ValueError("shell incidences realize a duplicate periodic ligand")
    return ordered


def _orbit_ligand_composition(
    contact_result: ContactAnalysisResult,
    vertices: tuple[PolyhedronVertex, ...],
) -> tuple[tuple[str, float], ...]:
    incidence_by_id = {
        item.incidence_orbit_id: item
        for item in contact_result.contact_incidence_orbits
    }
    interpretation_by_id = {
        item.interpretation_id: item
        for orbit in contact_result.contact_orbits
        for item in orbit.interpretations
    }
    geometry_by_resolved = {
        resolved.resolved_contact_orbit_id: next(
            geometry
            for geometry in contact_result.pair_table.contact_orbits
            if geometry.geometry_orbit_id == resolved.geometry_orbit_id
        )
        for resolved in contact_result.contact_orbits
    }
    sites = {item.id: item for item in contact_result._structure.sites}
    totals: dict[str, list[float]] = {}
    for vertex in vertices:
        incidence = incidence_by_id[vertex.incidence_orbit_id]
        interpretation = interpretation_by_id[incidence.interpretation_id]
        geometry = geometry_by_resolved[incidence.resolved_contact_orbit_id]
        if (
            incidence.center_independent_site_id == geometry.first_independent_site_id
            and incidence.ligand_independent_site_id == geometry.second_independent_site_id
        ):
            participating = {
                item.second_species
                for item in interpretation.component_pair_interpretations
            }
        elif (
            incidence.center_independent_site_id == geometry.second_independent_site_id
            and incidence.ligand_independent_site_id == geometry.first_independent_site_id
        ):
            participating = {
                item.first_species
                for item in interpretation.component_pair_interpretations
            }
        else:
            participating = {
                species
                for item in interpretation.component_pair_interpretations
                for species in (item.first_species, item.second_species)
            }
        for component in sites[incidence.ligand_independent_site_id].components:
            if component.species not in participating:
                continue
            label = component.element or component.species.label
            totals.setdefault(label, []).append(float(component.occupancy.value))
    return tuple(
        (label, math.fsum(values)) for label, values in sorted(totals.items())
    )


def _representative_polyhedron(
    contact_result: ContactAnalysisResult,
    shell,
    tolerance: float,
) -> CoordinationPolyhedron:
    alternative = shell.selected
    if alternative is None:
        raise ValueError("resolved shell lacks a selected alternative")
    vertices = _local_polyhedron_vertices(contact_result, shell, alternative)
    local = np.asarray(
        tuple(item.local_cartesian for item in vertices), dtype=float
    ).reshape((-1, 3))
    distances = tuple(item.distance for item in vertices)
    mean, minimum, maximum, distortion = _bond_length_metrics(distances)
    diagnostics: list[Diagnostic] = []
    incidence_by_id = {
        item.incidence_orbit_id: item
        for item in contact_result.contact_incidence_orbits
    }
    sites = {item.id: item for item in contact_result._structure.sites}
    if any(item.occupancy < 1.0 - 1e-12 for item in vertices):
        diagnostics.append(Diagnostic(
            Severity.WARNING,
            PARTIAL_OCCUPANCY,
            "Polyhedron contains a partially occupied ligand vertex",
        ))
    if any(
        len(sites[incidence_by_id[item.incidence_orbit_id].ligand_independent_site_id].components) > 1
        for item in vertices
    ):
        diagnostics.append(Diagnostic(
            Severity.WARNING,
            MIXED_POSITION,
            "Polyhedron contains a mixed ligand position",
        ))

    faces: tuple[tuple[int, ...], ...] = ()
    face_signature = None
    angle_dispersion = None
    volume = None
    centroid = None
    center_offset = None
    status = ResolutionStatus.RESOLVED
    is_three_dimensional = (
        len(local) >= 4
        and np.linalg.matrix_rank(local[1:] - local[0], tol=tolerance) >= 3
    )
    if is_three_dimensional:
        try:
            faces = _convex_hull_faces(local, tolerance)
            face_signature = canonical_face_signature(len(vertices), faces)
            angle_dispersion = _edge_angle_dispersion_deg(
                local, unique_hull_edges(faces)
            )
            volume_value, centroid_value = _volume_and_centroid(local, faces)
            volume = volume_value
            centroid = tuple(float(value) for value in centroid_value)
            center_offset = float(np.linalg.norm(centroid_value))
        except ValueError as error:
            faces = ()
            face_signature = None
            angle_dispersion = None
            volume = None
            centroid = None
            center_offset = None
            status = ResolutionStatus.INCOMPLETE
            diagnostics.append(Diagnostic(Severity.WARNING, GEOMETRY_DEGENERATE, str(error)))
    else:
        status = ResolutionStatus.INCOMPLETE
        diagnostics.append(Diagnostic(
            Severity.WARNING,
            GEOMETRY_DEGENERATE,
            "Coordination shell is not a closed three-dimensional polyhedron",
        ))
    if distortion is None or angle_dispersion is None:
        diagnostics.append(Diagnostic(
            Severity.WARNING,
            DESCRIPTOR_UNAVAILABLE,
            "One or more polyhedron descriptors are unavailable",
        ))

    polyhedron_orbit_id = "polyhedron-orbit:" + _digest(
        {"shell_orbit_id": shell.shell_orbit_id}
    )
    polyhedron_id = "polyhedron:" + _digest(
        {
            "shell_orbit_id": shell.shell_orbit_id,
            "alternative_id": alternative.alternative_id,
            "vertices": tuple(
                (
                    item.atom_ref.atom_id,
                    item.atom_ref.cell_translation,
                    item.incidence_orbit_id,
                )
                for item in vertices
            ),
        }
    )
    center_image = _source_image(contact_result, shell.center_independent_site_id)
    return CoordinationPolyhedron(
        polyhedron_id=polyhedron_id,
        polyhedron_orbit_id=polyhedron_orbit_id,
        source_shell_orbit_id=shell.shell_orbit_id,
        source_site_id=shell.center_independent_site_id,
        center_atom_id=center_image.image_id,
        coordination_number=len(vertices),
        ligand_composition=_orbit_ligand_composition(contact_result, vertices),
        vertices=vertices,
        faces=faces,
        face_signature=face_signature,
        mean_bond_length=mean,
        min_bond_length=minimum,
        max_bond_length=maximum,
        bond_length_distortion=distortion,
        edge_angle_dispersion_deg=angle_dispersion,
        volume=volume,
        geometric_centroid=centroid,
        center_offset=center_offset,
        status=status,
        provenance=(
            *shell.provenance,
            ("method", "cristma.orbit_polyhedron_builder:1"),
            ("selected_alternative_id", alternative.alternative_id),
            ("tolerance", tolerance),
            ("bond_distortion", "Baur mean absolute relative deviation"),
            ("edge_angle_dispersion", "population standard deviation in degrees"),
        ),
        local_vertices=tuple(item.local_cartesian for item in vertices),
        shell_provenance=shell.provenance,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )


@dataclass(frozen=True, slots=True)
class PolyhedronOrbitBuilder:
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("polyhedron tolerance must be positive and finite")

    def build(self, contact_result: ContactAnalysisResult) -> PolyhedronOrbitBuildResult:
        if not isinstance(contact_result, ContactAnalysisResult):
            raise TypeError("contact_result must be ContactAnalysisResult")
        polyhedra: list[CoordinationPolyhedron] = []
        orbits: list[CoordinationPolyhedronOrbit] = []
        diagnostics: list[Diagnostic] = []
        skipped = False
        for shell in contact_result.coordination_shell_orbits:
            if shell.status is not ResolutionStatus.RESOLVED:
                skipped = True
                code = (
                    "crystal_chemistry.polyhedron.shell_ambiguous"
                    if shell.status is ResolutionStatus.AMBIGUOUS
                    else "crystal_chemistry.polyhedron.shell_incomplete"
                )
                diagnostic = Diagnostic(
                    Severity.WARNING,
                    code,
                    f"polyhedron is not selected from a {shell.status.value} shell",
                )
                diagnostics.append(diagnostic)
                continue
            polyhedron = _representative_polyhedron(
                contact_result, shell, self.tolerance
            )
            orbit = CoordinationPolyhedronOrbit(
                polyhedron.polyhedron_orbit_id,
                polyhedron.polyhedron_id,
                (polyhedron,),
                polyhedron.status,
                polyhedron.diagnostics,
            )
            polyhedra.append(polyhedron)
            orbits.append(orbit)
            diagnostics.extend(polyhedron.diagnostics)
        return PolyhedronOrbitBuildResult(
            tuple(sorted(polyhedra, key=lambda item: item.polyhedron_id)),
            tuple(sorted(orbits, key=lambda item: item.polyhedron_orbit_id)),
            tuple(dict.fromkeys(diagnostics)),
            not skipped and all(
                item.status is ResolutionStatus.RESOLVED for item in polyhedra
            ),
        )


__all__ = [
    "PolyhedronOrbitBuildResult",
    "PolyhedronOrbitBuilder",
]
