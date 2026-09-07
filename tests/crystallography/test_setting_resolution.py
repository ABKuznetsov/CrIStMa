from __future__ import annotations

from dataclasses import replace

import pytest

from cristma.crystallography import (
    SpaceGroupCatalog,
    SpaceGroupSettingResolutionStatus,
    resolve_space_group_setting,
)
from cristma.symmetry import SpaceGroupDefinition


def test_every_catalog_definition_resolves_to_its_exact_setting() -> None:
    catalog = SpaceGroupCatalog.default()

    for setting in catalog.settings:
        resolution = resolve_space_group_setting(
            setting.definition(provenance="derived"),
            catalog,
        )

        assert resolution.status is SpaceGroupSettingResolutionStatus.RESOLVED
        assert resolution.setting is not None
        assert resolution.setting.setting_id == setting.setting_id
        assert tuple(item.setting_id for item in resolution.candidates) == (
            setting.setting_id,
        )
        assert resolution.provenance.operation_count == len(
            setting.symmetry_operations
        )
        assert resolution.provenance.catalog_dataset_id == (
            "cristma.crystallography.spglib"
        )


def test_resolution_is_independent_of_reported_operation_order() -> None:
    catalog = SpaceGroupCatalog.default()
    setting = catalog.by_setting(439)
    definition = replace(
        setting.definition(provenance="reported"),
        operations=tuple(reversed(setting.symmetry_operations)),
    )

    resolution = resolve_space_group_setting(definition, catalog)

    assert resolution.status is SpaceGroupSettingResolutionStatus.RESOLVED
    assert resolution.setting is not None
    assert resolution.setting.setting_id == 439


@pytest.mark.parametrize(
    "setting_ids",
    ((322, 324), (326, 328), (330, 332)),
)
def test_operation_identical_settings_remain_ambiguous_without_choice(
    setting_ids: tuple[int, int],
) -> None:
    catalog = SpaceGroupCatalog.default()
    setting = catalog.by_setting(setting_ids[0])
    definition = replace(
        setting.definition(provenance="reported"),
        setting=None,
    )

    resolution = resolve_space_group_setting(definition, catalog)

    assert resolution.status is SpaceGroupSettingResolutionStatus.AMBIGUOUS
    assert resolution.setting is None
    assert tuple(item.setting_id for item in resolution.candidates) == setting_ids
    assert tuple(item.code for item in resolution.diagnostics) == (
        "symmetry.setting_resolution.ambiguous",
    )


def test_incomplete_operation_set_is_unresolved_even_with_catalog_metadata() -> None:
    catalog = SpaceGroupCatalog.default()
    setting = catalog.by_setting(439)
    definition = SpaceGroupDefinition(
        operations=setting.symmetry_operations[:-1],
        provenance="reported",
        number=setting.number,
        hm_symbol=setting.hm_full,
        hall_symbol=setting.hall_symbol,
        setting=setting.choice,
    )

    resolution = resolve_space_group_setting(definition, catalog)

    assert resolution.status is SpaceGroupSettingResolutionStatus.UNRESOLVED
    assert resolution.setting is None
    assert resolution.candidates == ()
    assert tuple(item.code for item in resolution.diagnostics) == (
        "symmetry.setting_resolution.unresolved",
    )
