"""Reusable orbit-first crystal-chemistry calculations and results."""

from .models import (
    ComponentPairInterpretation,
    EvidenceStatus,
    ResolutionStatus,
    SecondaryEvidence,
)
from .policy import ShellResolutionPolicy
from .orbit_contacts import (
    ContactInterpretation,
    ContactOrbitResolution,
    ContactOrbitResolver,
    EndpointRole,
    EndpointRoles,
    OrientationMode,
    ResolvedContactOrbit,
)
from .incidence_orbits import ContactIncidenceBuilder, ContactIncidenceOrbit
from .shell_orbits import (
    CoordinationShellAlternative,
    CoordinationShellOrbit,
    CoordinationShellOrbitResolver,
    ShellRole,
)
from .contact_analysis import (
    ContactAnalysisResult,
    ContactAnalyzer,
    aggregate_contact_analysis_status,
)
from .polyhedra import (
    CoordinationPolyhedron,
    CoordinationPolyhedronOrbit,
    FaceSignature,
    PolyhedronVertex,
    canonical_face_signature,
    polyhedron_face_signature,
    unique_hull_edges,
)
from .polyhedron_orbits import PolyhedronOrbitBuildResult, PolyhedronOrbitBuilder
from .materialization import (
    CellRange,
    ContactMaterializer,
    IntermolecularContactMaterializer,
    ReferenceCell,
    ResolvedContact,
    ShellMembership,
)
from .intermolecular import IntermolecularContactAnalyzer
from .intermolecular_models import (
    HydrogenBondCandidate,
    IntermolecularContactAnalysisResult,
    IntermolecularContactOrbit,
    IntermolecularContactPolicy,
    MaterializedIntermolecularContact,
    VanDerWaalsDistanceEvidence,
)
from .shannon_distance import ShannonDistanceCheck, ShannonDistanceValidator
from .structural_units import (
    StructuralUnitBuildResult,
    StructuralUnitBuilder,
    StructuralUnitGeometry,
    StructuralUnitGeometryKind,
    StructuralUnitKind,
    StructuralUnitOrbit,
)
from .structural_graph import (
    StructuralConnectionKind,
    StructuralConnectionOrbit,
    StructuralGraphBuilder,
    StructuralUnitGraph,
)
from .representation import (
    StructuralRepresentation,
    StructuralRepresentationBuilder,
    StructuralSelectionPolicy,
)
from .periodic_connectivity import (
    PeriodicComponent,
    PeriodicConnectivityAnalyzer,
    PeriodicConnectivityResult,
    integer_translation_lattice_basis,
)
from .structural_blocks import (
    StructuralBlock,
    StructuralBlockClassification,
    StructuralBlockFinder,
    StructuralBlockResult,
)
from .ring_finder import RingFinder
from .rings import (
    PeriodicUnitOrbitRef,
    RingAnalysisResult,
    RingAnalysisStatus,
    RingSearchPolicy,
    StructuralRingOrbit,
    StructuralRingScope,
)
from .molecular import (
    MolecularAnalysisPolicy,
    MolecularAnalysisResult,
    MolecularAnalyzer,
    MolecularBondOrbit,
    MolecularComponentKind,
    MolecularComponentOrbit,
)
from .molecular_rings import (
    MolecularAtomOrbitRef,
    MolecularRingOrbit,
    MolecularRingSearchPolicy,
)

__all__ = [
    "ComponentPairInterpretation",
    "CellRange",
    "ContactMaterializer",
    "HydrogenBondCandidate",
    "IntermolecularContactAnalysisResult",
    "IntermolecularContactAnalyzer",
    "IntermolecularContactMaterializer",
    "IntermolecularContactOrbit",
    "IntermolecularContactPolicy",
    "ContactInterpretation",
    "ContactIncidenceBuilder",
    "ContactIncidenceOrbit",
    "ContactAnalysisResult",
    "ContactAnalyzer",
    "ContactOrbitResolution",
    "ContactOrbitResolver",
    "MolecularAnalysisPolicy",
    "MolecularAnalysisResult",
    "MolecularAnalyzer",
    "MolecularAtomOrbitRef",
    "MolecularBondOrbit",
    "MolecularComponentKind",
    "MolecularComponentOrbit",
    "MolecularRingOrbit",
    "MolecularRingSearchPolicy",
    "MaterializedIntermolecularContact",
    "CoordinationShellAlternative",
    "CoordinationShellOrbit",
    "CoordinationShellOrbitResolver",
    "CoordinationPolyhedron",
    "CoordinationPolyhedronOrbit",
    "EvidenceStatus",
    "EndpointRole",
    "EndpointRoles",
    "OrientationMode",
    "ResolutionStatus",
    "ResolvedContact",
    "ResolvedContactOrbit",
    "PolyhedronOrbitBuildResult",
    "PolyhedronOrbitBuilder",
    "PolyhedronVertex",
    "FaceSignature",
    "SecondaryEvidence",
    "ShellMembership",
    "ShellRole",
    "ShellResolutionPolicy",
    "ShannonDistanceCheck",
    "ShannonDistanceValidator",
    "StructuralUnitBuildResult",
    "StructuralUnitBuilder",
    "StructuralUnitGeometry",
    "StructuralUnitGeometryKind",
    "StructuralUnitKind",
    "StructuralUnitOrbit",
    "StructuralConnectionKind",
    "StructuralConnectionOrbit",
    "StructuralGraphBuilder",
    "StructuralUnitGraph",
    "StructuralRepresentation",
    "StructuralRepresentationBuilder",
    "StructuralSelectionPolicy",
    "PeriodicComponent",
    "PeriodicConnectivityAnalyzer",
    "PeriodicConnectivityResult",
    "integer_translation_lattice_basis",
    "StructuralBlock",
    "StructuralBlockClassification",
    "StructuralBlockFinder",
    "StructuralBlockResult",
    "PeriodicUnitOrbitRef",
    "RingAnalysisResult",
    "RingAnalysisStatus",
    "RingFinder",
    "RingSearchPolicy",
    "StructuralRingOrbit",
    "StructuralRingScope",
    "ReferenceCell",
    "polyhedron_face_signature",
    "canonical_face_signature",
    "unique_hull_edges",
    "VanDerWaalsDistanceEvidence",
    "aggregate_contact_analysis_status",
]
