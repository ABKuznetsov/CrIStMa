from __future__ import annotations

from fractions import Fraction

from cristma.chemistry import (
    CandidateInteraction,
    ChemicalEvidence,
    CompositionGrammar,
    DecompositionMode,
    GrammarOperation,
    InteractionLayer,
    InteractionPriority,
)
from cristma.core import MeasuredValue, UnitCell
from cristma.crystal_chemistry import (
    ContactAnalyzer,
    PolyhedronOrbitBuilder,
    ResolutionStatus,
    ShellResolutionPolicy,
)
from cristma.crystallography import SymmetryContext
from cristma.structure import CrystalStructure, IndependentSite, SiteComponent
from cristma.symmetry import AffineOperation
import cristma.crystal_chemistry.polyhedra as polyhedra_module
import cristma.crystal_chemistry.polyhedron_orbits as polyhedron_orbits_module


def _value(value: float) -> MeasuredValue:
    return MeasuredValue(value, None, str(value))


def _site(site_id: str, fractional: tuple[float, float, float], symbol: str) -> IndependentSite:
    return IndependentSite(
        site_id,
        site_id,
        (SiteComponent(symbol, _value(1.0)),),
        tuple(_value(value) for value in fractional),
    )


IDENTITY = AffineOperation(
    (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    ),
    (Fraction(0), Fraction(0), Fraction(0)),
)
INVERSION = AffineOperation(
    (
        (Fraction(-1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(-1)),
    ),
    (Fraction(0), Fraction(0), Fraction(0)),
)


def _contact_result(*, ambiguous: bool = False):
    cell = UnitCell.cubic(_value(10.0))
    if ambiguous:
        sites = (
            _site("M", (0.0, 0.0, 0.0), "Ca"),
            _site("X1", (0.20, 0.0, 0.0), "O"),
            _site("X2", (0.30, 0.0, 0.0), "O"),
            _site("X3", (0.40, 0.0, 0.0), "O"),
        )
        policy = ShellResolutionPolicy(1.6, 0.01, 0.20, 1.0, 2.0)
    else:
        sites = (
            _site("M", (0.0, 0.0, 0.0), "Ca"),
            _site("X", (0.20, 0.0, 0.0), "O"),
            _site("Y", (0.0, 0.20, 0.0), "O"),
            _site("Z", (0.0, 0.0, 0.20), "O"),
            _site("outer", (0.40, 0.0, 0.0), "O"),
        )
        policy = ShellResolutionPolicy(1.6, 0.01, 0.20, 0.01, 2.0)
    structure = CrystalStructure("polyhedron", cell, sites)
    context = SymmetryContext.from_operations((IDENTITY, INVERSION), cell)
    request = CandidateInteraction(
        ("Ca",), ("O",), GrammarOperation.CENTRE_LIGAND_SHELL,
        InteractionLayer.COORDINATION, InteractionPriority.PRIMARY,
        ("Ca",), ("O",), (ChemicalEvidence("fixture", "coordination"),),
    )
    grammar = CompositionGrammar(
        DecompositionMode.COVALENT_NETWORK,
        (request,), 1.0, (ChemicalEvidence("fixture", "grammar"),), (), "fixture",
    )
    return ContactAnalyzer(policy).analyze(structure, context, grammar)


def test_polyhedron_realizes_local_vertices_without_global_contacts() -> None:
    contact_result = _contact_result()
    result = PolyhedronOrbitBuilder().build(contact_result)

    assert result.complete
    assert len(result.polyhedron_orbits) == 1
    polyhedron = result.polyhedron_orbits[0].representative
    assert polyhedron.coordination_number == 6
    assert {tuple(round(value, 12) for value in row) for row in polyhedron.local_vertices} == {
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, -2.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, -2.0),
        (0.0, 0.0, 2.0),
    }
    assert len({vertex.atom_ref for vertex in polyhedron.vertices}) == 6
    assert len({vertex.incidence_orbit_id for vertex in polyhedron.vertices}) == 3
    assert not hasattr(polyhedron, "vertex_contacts")
    assert polyhedron.face_signature is not None
    assert polyhedron.volume is not None


def test_ambiguous_shell_does_not_guess_one_polyhedron() -> None:
    contact_result = _contact_result(ambiguous=True)
    assert contact_result.status is ResolutionStatus.AMBIGUOUS

    result = PolyhedronOrbitBuilder().build(contact_result)

    assert result.polyhedron_orbits == ()
    assert "crystal_chemistry.polyhedron.shell_ambiguous" in result.diagnostic_codes


def test_hull_failure_returns_incomplete_polyhedron_instead_of_raising(
    monkeypatch,
) -> None:
    contact_result = _contact_result()

    def fail_signature(*_args, **_kwargs):
        raise ValueError("polyhedron face graph must be a connected closed manifold")

    monkeypatch.setattr(
        polyhedron_orbits_module, "canonical_face_signature", fail_signature
    )
    monkeypatch.setattr(polyhedra_module, "canonical_face_signature", fail_signature)

    result = PolyhedronOrbitBuilder().build(contact_result)

    assert len(result.polyhedron_orbits) == 1
    polyhedron = result.polyhedron_orbits[0].representative
    assert polyhedron.status is ResolutionStatus.INCOMPLETE
    assert polyhedron.faces == ()
    assert polyhedron.face_signature is None
    assert "crystal_chemistry.polyhedron.geometry_degenerate" in result.diagnostic_codes


def test_incidence_retains_every_exact_local_relation_needed_for_realization() -> None:
    contact_result = _contact_result()

    assert all(
        len(incidence.equivalent_oriented_relations)
        == incidence.incidence_multiplicity_per_center
        for incidence in contact_result.contact_incidence_orbits
    )
    assert all(
        incidence.oriented_periodic_relation
        == incidence.equivalent_oriented_relations[0]
        for incidence in contact_result.contact_incidence_orbits
    )
