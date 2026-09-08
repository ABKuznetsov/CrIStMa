"""Pinned van der Waals radii for intermolecular contact calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import math
from types import MappingProxyType
from typing import Mapping

from cristma.chemistry.elements import normalize_element


@dataclass(frozen=True, slots=True)
class VanDerWaalsRadiusRecord:
    symbol: str
    value_angstrom: float
    unit: str = "angstrom"
    dataset_id: str = "cristma.vdw_radii.pymatgen_crc_2024"
    dataset_version: str = "1"
    source_commit: str = "8495e941504cd5123701635b6572942c78d9589c"


class VanDerWaalsRadii:
    """Exact lookup with no guessed fallback for missing elements."""

    def __init__(self, records: tuple[VanDerWaalsRadiusRecord, ...]) -> None:
        normalized: dict[str, VanDerWaalsRadiusRecord] = {}
        for record in records:
            symbol = normalize_element(record.symbol)
            if symbol in normalized:
                raise ValueError("van der Waals radius symbols must be unique")
            if (
                isinstance(record.value_angstrom, bool)
                or not isinstance(record.value_angstrom, (int, float))
                or not math.isfinite(float(record.value_angstrom))
                or float(record.value_angstrom) <= 0.0
            ):
                raise ValueError("van der Waals radii must be positive and finite")
            if record.unit != "angstrom":
                raise ValueError("van der Waals radii must use angstrom units")
            normalized[symbol] = record
        self._records: Mapping[str, VanDerWaalsRadiusRecord] = MappingProxyType(
            dict(sorted(normalized.items()))
        )

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "VanDerWaalsRadii":
        resource = files("cristma.reference_data").joinpath(
            "resources", "vdw_radii.json"
        )
        payload = json.loads(resource.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        return cls(tuple(
            VanDerWaalsRadiusRecord(
                symbol=item["symbol"],
                value_angstrom=float(item["value_angstrom"]),
                unit=metadata["unit"],
                dataset_id=metadata["dataset_id"],
                dataset_version=metadata["version"],
                source_commit=metadata["source_commit"],
            )
            for item in payload["records"]
        ))

    def find(self, symbol: str) -> VanDerWaalsRadiusRecord:
        normalized = normalize_element(symbol)
        try:
            return self._records[normalized]
        except KeyError as exc:
            raise KeyError(
                f"No van der Waals radius for {normalized}"
            ) from exc

    @property
    def records(self) -> tuple[VanDerWaalsRadiusRecord, ...]:
        return tuple(self._records.values())


__all__ = ["VanDerWaalsRadii", "VanDerWaalsRadiusRecord"]
