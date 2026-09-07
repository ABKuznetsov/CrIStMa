"""Exact resolution of symmetry definitions to catalog settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cristma.diagnostics import Diagnostic, Severity
from cristma.symmetry import SpaceGroupDefinition

from .catalog import SpaceGroupCatalog, _operation_signature
from .space_group import SpaceGroupSetting


_METHOD = "exact_normalized_operation_set"
_VERSION = "1"


class SpaceGroupSettingResolutionStatus(StrEnum):
    """Outcome of matching one exact symmetry definition to the catalog."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class SpaceGroupSettingResolutionProvenance:
    """Catalog and method identity used for one resolution."""

    method: str
    version: str
    catalog_dataset_id: str
    catalog_schema_version: str
    catalog_resource_sha256: tuple[str, str]
    operation_count: int
    choice_used: str | None


@dataclass(frozen=True, slots=True)
class SpaceGroupSettingResolution:
    """Immutable exact-match result without a guessed setting."""

    status: SpaceGroupSettingResolutionStatus
    setting: SpaceGroupSetting | None
    candidates: tuple[SpaceGroupSetting, ...]
    diagnostics: tuple[Diagnostic, ...]
    provenance: SpaceGroupSettingResolutionProvenance

    def __post_init__(self) -> None:
        candidate_ids = tuple(item.setting_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("setting candidates must be unique and sorted")
        if self.status is SpaceGroupSettingResolutionStatus.RESOLVED:
            if len(self.candidates) != 1 or self.setting is not self.candidates[0]:
                raise ValueError("resolved setting requires one selected candidate")
        elif self.status is SpaceGroupSettingResolutionStatus.AMBIGUOUS:
            if len(self.candidates) < 2 or self.setting is not None:
                raise ValueError("ambiguous resolution requires multiple candidates")
        elif self.status is SpaceGroupSettingResolutionStatus.UNRESOLVED:
            if self.candidates or self.setting is not None:
                raise ValueError("unresolved setting must not contain candidates")


def _choice(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return " ".join(value.casefold().split())


def resolve_space_group_setting(
    definition: SpaceGroupDefinition,
    catalog: SpaceGroupCatalog,
) -> SpaceGroupSettingResolution:
    """Resolve exact operations in their current basis to a catalog setting."""

    if not isinstance(definition, SpaceGroupDefinition):
        raise TypeError("definition must be SpaceGroupDefinition")
    if not isinstance(catalog, SpaceGroupCatalog):
        raise TypeError("catalog must be SpaceGroupCatalog")

    signature = _operation_signature(definition.operations)
    candidates = catalog._by_operation_signature.get(
        signature,
        (),
    )
    reported_choice = _choice(definition.setting) or _choice(
        definition.origin_choice
    )
    choice_used: str | None = None
    if len(candidates) > 1 and reported_choice is not None:
        filtered = tuple(
            setting
            for setting in candidates
            if _choice(setting.choice) == reported_choice
        )
        if len(filtered) == 1:
            candidates = filtered
            choice_used = reported_choice

    provenance = SpaceGroupSettingResolutionProvenance(
        method=_METHOD,
        version=_VERSION,
        catalog_dataset_id=catalog.dataset_id,
        catalog_schema_version=catalog.schema_version,
        catalog_resource_sha256=catalog.resource_sha256,
        operation_count=len(definition.operations),
        choice_used=choice_used,
    )
    if len(candidates) == 1:
        return SpaceGroupSettingResolution(
            SpaceGroupSettingResolutionStatus.RESOLVED,
            candidates[0],
            candidates,
            (),
            provenance,
        )
    if candidates:
        diagnostic = Diagnostic(
            Severity.WARNING,
            "symmetry.setting_resolution.ambiguous",
            "Exact symmetry operations match multiple catalog settings; "
            "an explicit setting choice is required.",
        )
        return SpaceGroupSettingResolution(
            SpaceGroupSettingResolutionStatus.AMBIGUOUS,
            None,
            candidates,
            (diagnostic,),
            provenance,
        )
    diagnostic = Diagnostic(
        Severity.WARNING,
        "symmetry.setting_resolution.unresolved",
        "Exact symmetry operations do not match a catalog setting in this basis.",
    )
    return SpaceGroupSettingResolution(
        SpaceGroupSettingResolutionStatus.UNRESOLVED,
        None,
        (),
        (diagnostic,),
        provenance,
    )


__all__ = [
    "SpaceGroupSettingResolution",
    "SpaceGroupSettingResolutionProvenance",
    "SpaceGroupSettingResolutionStatus",
    "resolve_space_group_setting",
]
