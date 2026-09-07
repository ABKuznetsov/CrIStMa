"""Read-only facade over packaged crystallographic reference data."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from types import MappingProxyType
from typing import Mapping

from cristma.symmetry.affine import AffineOperation

from .space_group import SpaceGroupSetting
from .wyckoff import AffineCoordinateMap, WyckoffPosition


_SCHEMA_VERSION = "1.0.0"
_DATASET_ID = "cristma.crystallography.spglib"


def _operation_signature(operations: tuple[AffineOperation, ...]) -> tuple[object, ...]:
    return tuple(
        sorted(
            (
                operation.normalized().rotation,
                operation.normalized().translation,
            )
            for operation in operations
        )
    )


def _normalized_symbol(value: str) -> str:
    return " ".join(value.casefold().split())


def _fraction(value: object) -> Fraction:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"invalid rational pair: {value!r}")
    numerator, denominator = value
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise ValueError(f"invalid rational pair: {value!r}")
    return Fraction(int(numerator), int(denominator))


def _load_catalog(space_group_bytes: bytes, wyckoff_bytes: bytes) -> "SpaceGroupCatalog":
    group_document = json.loads(space_group_bytes)
    wyckoff_document = json.loads(wyckoff_bytes)
    group_metadata = group_document.get("metadata")
    wyckoff_metadata = wyckoff_document.get("metadata")
    if group_metadata != wyckoff_metadata:
        raise ValueError("space-group and Wyckoff resource metadata disagree")
    if not isinstance(group_metadata, dict):
        raise ValueError("crystallography resource metadata is missing")
    if group_metadata.get("dataset_id") != _DATASET_ID:
        raise ValueError("unsupported crystallography dataset")
    if group_metadata.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported crystallography schema")

    raw_wyckoffs = wyckoff_document.get("records")
    raw_groups = group_document.get("records")
    if not isinstance(raw_wyckoffs, dict) or not isinstance(raw_groups, list):
        raise ValueError("crystallography resource records are malformed")

    settings: list[SpaceGroupSetting] = []
    for raw_group in raw_groups:
        setting_id = int(raw_group["hall_number"])
        positions = []
        for raw_position in raw_wyckoffs.get(str(setting_id), ()):
            constraints = tuple(
                AffineCoordinateMap(
                    parameter_matrix=tuple(
                        tuple(Fraction(int(value)) for value in row)
                        for row in raw_constraint["parameter_matrix"]
                    ),
                    translation=tuple(
                        _fraction(value) for value in raw_constraint["translation"]
                    ),
                    source=raw_constraint.get("source"),
                )
                for raw_constraint in raw_position["representatives"]
            )
            positions.append(
                WyckoffPosition(
                    setting_id=setting_id,
                    letter=str(raw_position["letter"]),
                    multiplicity=int(raw_position["multiplicity"]),
                    site_symmetry_symbol=str(raw_position["site_symmetry"]),
                    coordinate_constraints=constraints,
                )
            )

        operations = tuple(
            AffineOperation(
                rotation=tuple(
                    tuple(Fraction(int(value)) for value in row)
                    for row in raw_operation["rotation"]
                ),
                translation=tuple(
                    _fraction(value) for value in raw_operation["translation"]
                ),
                id=f"hall:{setting_id}:op:{index}",
            )
            for index, raw_operation in enumerate(raw_group["operations"], start=1)
        )
        settings.append(
            SpaceGroupSetting(
                setting_id=setting_id,
                number=int(raw_group["number"]),
                hall_symbol=str(raw_group["hall_symbol"]),
                choice=str(raw_group["choice"]),
                hm_short=str(raw_group["hm_short"]),
                hm_full=str(raw_group["hm_full"]),
                point_group=str(raw_group["point_group"]),
                centering=str(raw_group["centering"]),
                crystal_system=str(raw_group["crystal_system"]),
                symmetry_operations=operations,
                wyckoff_positions=tuple(positions),
            )
        )

    catalog = SpaceGroupCatalog(
        settings=tuple(settings),
        dataset_id=str(group_metadata["dataset_id"]),
        schema_version=str(group_metadata["schema_version"]),
        resource_sha256=(
            hashlib.sha256(space_group_bytes).hexdigest(),
            hashlib.sha256(wyckoff_bytes).hexdigest(),
        ),
    )
    catalog._validate_complete()
    return catalog


@dataclass(frozen=True, slots=True)
class SpaceGroupCatalog:
    """Read-only lookup facade over all packaged Hall settings."""

    settings: tuple[SpaceGroupSetting, ...]
    dataset_id: str
    schema_version: str
    resource_sha256: tuple[str, str]
    _by_setting: Mapping[int, SpaceGroupSetting] = field(init=False, repr=False)
    _by_number: Mapping[int, tuple[SpaceGroupSetting, ...]] = field(
        init=False,
        repr=False,
    )
    _by_hall: Mapping[str, tuple[SpaceGroupSetting, ...]] = field(
        init=False,
        repr=False,
    )
    _by_operation_signature: Mapping[tuple[object, ...], tuple[SpaceGroupSetting, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        by_setting: dict[int, SpaceGroupSetting] = {}
        by_number: dict[int, list[SpaceGroupSetting]] = {}
        by_hall: dict[str, list[SpaceGroupSetting]] = {}
        by_operation_signature: dict[tuple[object, ...], list[SpaceGroupSetting]] = {}
        for setting in self.settings:
            if setting.setting_id in by_setting:
                raise ValueError(f"duplicate Hall setting {setting.setting_id}")
            by_setting[setting.setting_id] = setting
            by_number.setdefault(setting.number, []).append(setting)
            by_hall.setdefault(_normalized_symbol(setting.hall_symbol), []).append(setting)
            by_operation_signature.setdefault(
                _operation_signature(setting.symmetry_operations),
                [],
            ).append(setting)
        object.__setattr__(self, "_by_setting", MappingProxyType(by_setting))
        object.__setattr__(
            self,
            "_by_number",
            MappingProxyType({key: tuple(value) for key, value in by_number.items()}),
        )
        object.__setattr__(
            self,
            "_by_hall",
            MappingProxyType({key: tuple(value) for key, value in by_hall.items()}),
        )
        object.__setattr__(
            self,
            "_by_operation_signature",
            MappingProxyType(
                {
                    key: tuple(value)
                    for key, value in by_operation_signature.items()
                }
            ),
        )

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "SpaceGroupCatalog":
        root = files("cristma.reference_data").joinpath("resources/crystallography")
        return _load_catalog(
            root.joinpath("space_groups.json").read_bytes(),
            root.joinpath("wyckoff_positions.json").read_bytes(),
        )

    def __len__(self) -> int:
        return len(self.settings)

    def by_setting(self, setting_id: int) -> SpaceGroupSetting:
        return self._by_setting[setting_id]

    def by_number(self, number: int) -> tuple[SpaceGroupSetting, ...]:
        return self._by_number.get(number, ())

    def by_hall(self, hall_symbol: str) -> SpaceGroupSetting:
        matches = self._by_hall.get(_normalized_symbol(hall_symbol), ())
        if not matches:
            raise KeyError(hall_symbol)
        if len(matches) != 1:
            raise LookupError(
                f"ambiguous Hall symbol {hall_symbol!r}; use by_setting()"
            )
        return matches[0]

    def wyckoff_positions(self, setting_id: int) -> tuple[WyckoffPosition, ...]:
        return self.by_setting(setting_id).wyckoff_positions

    def _validate_complete(self) -> None:
        if set(self._by_setting) != set(range(1, 531)):
            raise ValueError("catalog must contain Hall settings 1..530")
        if set(self._by_number) != set(range(1, 231)):
            raise ValueError("catalog must contain space-group numbers 1..230")
        for setting in self.settings:
            positions = setting.wyckoff_positions
            if not positions:
                raise ValueError(f"Hall setting {setting.setting_id} has no Wyckoff positions")
            letters = tuple(position.letter for position in positions)
            if len(set(letters)) != len(letters):
                raise ValueError(f"Hall setting {setting.setting_id} repeats a Wyckoff letter")
            multiplicities = tuple(position.multiplicity for position in positions)
            if tuple(sorted(multiplicities, reverse=True)) != multiplicities:
                raise ValueError(
                    f"Hall setting {setting.setting_id} has unordered multiplicities"
                )
            if multiplicities[0] != len(setting.symmetry_operations):
                raise ValueError(
                    f"Hall setting {setting.setting_id} general multiplicity disagrees "
                    "with operation count"
                )
            identity = (
                ((Fraction(1), Fraction(0), Fraction(0)),
                 (Fraction(0), Fraction(1), Fraction(0)),
                 (Fraction(0), Fraction(0), Fraction(1))),
                (Fraction(0), Fraction(0), Fraction(0)),
            )
            normalized = {
                (operation.normalized().rotation, operation.normalized().translation)
                for operation in setting.symmetry_operations
            }
            if identity not in normalized:
                raise ValueError(f"Hall setting {setting.setting_id} lacks identity")


__all__ = ["SpaceGroupCatalog"]
