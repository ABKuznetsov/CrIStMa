"""Small immutable facade over the default CrIStMa reference catalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .chemical_reference import ChemicalReference, load_chemical_reference
from .elements import ElementCatalog
from .radii import CovalentRadii
from .shannon import ShannonRadii
from .vdw import VanDerWaalsRadii


@dataclass(frozen=True, slots=True)
class ReferenceData:
    elements: ElementCatalog
    covalent_radii: CovalentRadii
    shannon_radii: ShannonRadii
    chemical: ChemicalReference
    vdw_radii: VanDerWaalsRadii = field(default_factory=VanDerWaalsRadii.default)

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "ReferenceData":
        return cls(
            elements=ElementCatalog.default(),
            covalent_radii=CovalentRadii.default(),
            shannon_radii=ShannonRadii.default(),
            chemical=load_chemical_reference(),
            vdw_radii=VanDerWaalsRadii.default(),
        )


__all__ = ["ReferenceData"]
