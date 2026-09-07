"""Catalog-backed structural crystallography tools."""

from .space_group import SpaceGroupSetting
from .setting_resolution import (
    SpaceGroupSettingResolution,
    SpaceGroupSettingResolutionProvenance,
    SpaceGroupSettingResolutionStatus,
    resolve_space_group_setting,
)
from .wyckoff import AffineCoordinateMap, WyckoffPosition
from .catalog import SpaceGroupCatalog
from .orbit import (
    CrystallographicOrbit,
    SiteSymmetry,
    WyckoffAssignment,
    assign_wyckoff,
    build_orbit,
)
from .local_geometry import GeometricContact, geometric_contacts
from .symmetry_context import (
    DEFAULT_METRIC_TOLERANCE,
    DirectBasisConvention,
    SymmetryContext,
    SymmetryContextInvariantError,
    SymmetrySourceKind,
    canonical_operation_key,
)
from .periodic_relation import (
    ExactFractionalPosition,
    LatticeTranslation,
    PeriodicSymmetryRelation,
    compose_periodic_relations,
    identity_relation,
    invert_periodic_relation,
)
from .asu_mapping import (
    AsymmetricUnitMapper,
    AsymmetricUnitMapping,
    AsymmetricUnitMappingInvariantError,
    FractionalPosition,
    SiteImage,
    SiteOrbitMapping,
)
from .symmetry_pairs import (
    PairCandidateResult,
    PairTableStatus,
    SymmetryContactOrbit,
    SymmetryPairCandidate,
    SymmetryPairFinder,
    SymmetryPairSearchPolicy,
    SymmetryPairTable,
)
from .pair_canonical import (
    CanonicalPairDescriptor,
    EndpointInstance,
    PairInstanceOwner,
    canonical_instance_owner,
    canonical_pair_relation,
    periodic_endpoint_instance,
)

__all__ = [
    "AffineCoordinateMap",
    "AsymmetricUnitMapper",
    "AsymmetricUnitMapping",
    "AsymmetricUnitMappingInvariantError",
    "CanonicalPairDescriptor",
    "CrystallographicOrbit",
    "DEFAULT_METRIC_TOLERANCE",
    "DirectBasisConvention",
    "EndpointInstance",
    "ExactFractionalPosition",
    "FractionalPosition",
    "GeometricContact",
    "LatticeTranslation",
    "PairCandidateResult",
    "PairInstanceOwner",
    "PairTableStatus",
    "PeriodicSymmetryRelation",
    "SiteImage",
    "SiteOrbitMapping",
    "SiteSymmetry",
    "SpaceGroupCatalog",
    "SpaceGroupSetting",
    "SpaceGroupSettingResolution",
    "SpaceGroupSettingResolutionProvenance",
    "SpaceGroupSettingResolutionStatus",
    "SymmetryContactOrbit",
    "SymmetryContext",
    "SymmetryContextInvariantError",
    "SymmetryPairCandidate",
    "SymmetryPairFinder",
    "SymmetryPairSearchPolicy",
    "SymmetryPairTable",
    "SymmetrySourceKind",
    "WyckoffAssignment",
    "WyckoffPosition",
    "assign_wyckoff",
    "build_orbit",
    "canonical_instance_owner",
    "canonical_operation_key",
    "canonical_pair_relation",
    "compose_periodic_relations",
    "geometric_contacts",
    "identity_relation",
    "invert_periodic_relation",
    "periodic_endpoint_instance",
    "resolve_space_group_setting",
]
