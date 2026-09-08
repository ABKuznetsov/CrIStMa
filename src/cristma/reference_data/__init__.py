"""Versioned scientific reference data used by CrIStMa tools."""

from .elements import ElementCatalog, ElementCategory, ElementRecord
from .chemical_reference import (
    ChemicalReference,
    ChemicalReferenceIntegrityReport,
    load_chemical_reference,
    validate_reference_integrity,
)
from .facade import ReferenceData
from .radii import CovalentRadii, CovalentRadiusRecord
from .shannon import ShannonRadii, ShannonRadiusRecord, ShannonSpinState
from .vdw import VanDerWaalsRadii, VanDerWaalsRadiusRecord
from .xray_f0 import NeutralAtomFormFactorTable, XRayFormFactorProvenance

__all__ = [
    "ChemicalReference",
    "ChemicalReferenceIntegrityReport",
    "CovalentRadii",
    "CovalentRadiusRecord",
    "ElementCatalog",
    "ElementCategory",
    "ElementRecord",
    "ReferenceData",
    "ShannonRadii",
    "ShannonRadiusRecord",
    "ShannonSpinState",
    "VanDerWaalsRadii",
    "VanDerWaalsRadiusRecord",
    "NeutralAtomFormFactorTable",
    "XRayFormFactorProvenance",
    "load_chemical_reference",
    "validate_reference_integrity",
]
