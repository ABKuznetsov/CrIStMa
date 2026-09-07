"""Map loss-preserving CIF documents to canonical CrIStMa crystals."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import math
import re

import numpy as np

from cristma.chemistry.elements import normalize_element
from cristma.chemistry.species import IsotopeSpecies
from cristma.core.cell import UnitCell
from cristma.crystallography.catalog import SpaceGroupCatalog
from cristma.crystallography.orbit import assign_wyckoff, build_orbit
from cristma.crystallography.space_group import SpaceGroupSetting
from cristma.structure import (
    CrystalStructure,
    DisplacementParameters,
    IndependentSite,
    SiteComponent,
)
from cristma.core.values import MeasuredValue, parse_measured_value
from cristma.io.diagnostics import Diagnostic, Severity
from cristma.symmetry.affine import AffineOperation, parse_xyz_operation
from cristma.symmetry.displacement import (
    SymmetryConsistencyError,
    symmetrize_displacement,
)
from cristma.symmetry.orbit import SpaceGroupDefinition, expand_orbit

from . import names
from .document import CifBlock, CifDocument, CifLoop, CifScalar
from .tokens import CifToken


_STRAY_TRANSLATION_SUFFIX = re.compile(r"(?<=\d)[tT](?=\s*(?:,|$))")


def _cif_species(value: str) -> str | IsotopeSpecies:
    """Map CIF's conventional deuterium symbol without confusing element labels."""

    match = re.match(r"[A-Za-z]+", value.strip())
    if match is not None and match.group(0).casefold() == "d":
        return IsotopeSpecies("H", 2)
    return normalize_element(value)


def _scalar(block: CifBlock, aliases: tuple[str, ...]) -> CifScalar | None:
    for alias in aliases:
        value = block.scalar(alias)
        if value is not None:
            return value
    return None


def _scalar_text(block: CifBlock, aliases: tuple[str, ...]) -> str | None:
    scalar = _scalar(block, aliases)
    if scalar is None or scalar.value in {"?", "."}:
        return None
    return scalar.value


def _find_loop(block: CifBlock, required: tuple[str, ...]) -> CifLoop | None:
    required_names = {item.casefold() for item in required}
    for loop in block.loops:
        present = {item.casefold() for item in loop.tags}
        if required_names <= present:
            return loop
    return None


def _column(loop: CifLoop, aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        index = loop.column_index(alias)
        if index is not None:
            return index
    return None


def _token(
    row: tuple[CifToken, ...],
    loop: CifLoop,
    aliases: tuple[str, ...],
) -> CifToken | None:
    index = _column(loop, aliases)
    return row[index] if index is not None else None


def _looks_structural(block: CifBlock) -> bool:
    if any(_scalar(block, alias) is not None for alias in (names.CELL_A, names.ATOM_LABEL)):
        return True
    return any(
        loop.column_index(names.ATOM_LABEL[0]) is not None
        for loop in block.loops
    )


def _cell(
    block: CifBlock,
    diagnostics: list[Diagnostic],
) -> UnitCell | None:
    fields = (
        (names.CELL_A, "angstrom"),
        (names.CELL_B, "angstrom"),
        (names.CELL_C, "angstrom"),
        (names.CELL_ALPHA, "degree"),
        (names.CELL_BETA, "degree"),
        (names.CELL_GAMMA, "degree"),
    )
    scalars = [_scalar(block, aliases) for aliases, _unit in fields]
    if any(item is None for item in scalars):
        diagnostics.append(
            Diagnostic(
                Severity.ERROR,
                "cif.map.cell_missing",
                f"Block {block.name!r} does not report all six unit-cell parameters",
                block.data_token.span,
            )
        )
        return None
    try:
        values = tuple(
            parse_measured_value(scalar.raw_value, unit=unit)
            for scalar, (_aliases, unit) in zip(scalars, fields, strict=True)
        )
        if any(value.value is None for value in values):
            raise ValueError("missing cell value")
        return UnitCell(*values)
    except ValueError as exc:
        diagnostics.append(
            Diagnostic(
                Severity.ERROR,
                "cif.map.cell_invalid",
                f"Invalid unit cell in block {block.name!r}: {exc}",
                block.data_token.span,
            )
        )
        return None


def _normalized_space_group_symbol(value: str) -> str:
    return "".join(value.casefold().replace("_", "").split())


def _catalog_setting(
    block: CifBlock,
    catalog: SpaceGroupCatalog,
) -> tuple[SpaceGroupSetting | None, bool]:
    hall_symbol = _scalar_text(block, names.HALL_SYMBOL)
    if hall_symbol is not None:
        try:
            return catalog.by_hall(hall_symbol), False
        except KeyError:
            pass
        except LookupError:
            return None, True

    number_text = _scalar_text(block, names.IT_NUMBER)
    choice = _scalar_text(block, names.SETTING) or _scalar_text(block, names.ORIGIN_CHOICE)
    if number_text is not None and choice is not None:
        try:
            number = int(float(number_text))
        except ValueError:
            number = None
        if number is not None:
            matches = tuple(
                setting
                for setting in catalog.by_number(number)
                if setting.choice.casefold() == choice.casefold()
            )
            if len(matches) == 1:
                return matches[0], False
            if len(matches) > 1:
                return None, True

    hm_symbol = _scalar_text(block, names.HM_SYMBOL)
    if hm_symbol is not None:
        normalized = _normalized_space_group_symbol(hm_symbol)
        matches = tuple(
            setting
            for setting in catalog.settings
            if normalized
            in {
                _normalized_space_group_symbol(setting.hm_short),
                _normalized_space_group_symbol(setting.hm_full),
            }
        )
        if len(matches) == 1:
            return matches[0], False
        if len(matches) > 1:
            return None, True
    return None, False


def _operation_key(operation: AffineOperation) -> object:
    normalized = operation.normalized()
    return normalized.rotation, normalized.translation


def _same_operation_set(
    left: tuple[AffineOperation, ...],
    right: tuple[AffineOperation, ...],
) -> bool:
    return {_operation_key(operation) for operation in left} == {
        _operation_key(operation) for operation in right
    }


def _parse_symmetry_operation(
    token: CifToken,
    operation_id: str,
    diagnostics: list[Diagnostic],
) -> AffineOperation | None:
    try:
        return parse_xyz_operation(token.value, operation_id=operation_id)
    except ValueError as original_error:
        repaired = _STRAY_TRANSLATION_SUFFIX.sub("", token.value)
        if repaired != token.value:
            try:
                operation = parse_xyz_operation(repaired, operation_id=operation_id)
            except ValueError:
                pass
            else:
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.symmetry_operation_repaired",
                        (
                            f"Removed a non-standard trailing 't' from fractional "
                            f"translations in {token.value!r}"
                        ),
                        token.span,
                        recovery=repaired,
                    )
                )
                return operation
        diagnostics.append(
            Diagnostic(
                Severity.ERROR,
                "cif.map.symmetry_operation_invalid",
                str(original_error),
                token.span,
            )
        )
        return None


def _symmetry(
    block: CifBlock,
    diagnostics: list[Diagnostic],
    catalog: SpaceGroupCatalog,
) -> tuple[SpaceGroupDefinition, SpaceGroupSetting | None] | None:
    operation_tokens: list[CifToken] = []
    for loop in block.loops:
        index = _column(loop, names.SYMMETRY_OPERATION)
        if index is not None:
            operation_tokens.extend(row[index] for row in loop.row_tokens)
    scalar_operation = _scalar(block, names.SYMMETRY_OPERATION)
    if scalar_operation is not None:
        operation_tokens.append(scalar_operation.value_token)

    catalog_setting, lookup_ambiguous = _catalog_setting(block, catalog)
    provenance = "reported"
    if not operation_tokens:
        if catalog_setting is not None:
            diagnostics.append(
                Diagnostic(
                    Severity.INFO,
                    "cif.map.symmetry_operations_derived",
                    f"Symmetry operations derived from Hall setting "
                    f"{catalog_setting.setting_id} ({catalog_setting.hall_symbol}).",
                    block.data_token.span,
                )
            )
            return catalog_setting.definition(provenance="derived"), catalog_setting
        provenance = "identity_fallback"
        if lookup_ambiguous:
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.space_group_lookup_ambiguous",
                    "Reported space-group metadata identifies more than one setting.",
                    block.data_token.span,
                )
            )
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                "cif.map.symmetry_operations_missing",
                "No symmetry operations reported; using identity only",
                block.data_token.span,
                recovery="x,y,z",
            )
        )
        operations = (parse_xyz_operation("x,y,z", operation_id="op:1"),)
    else:
        parsed = []
        for index, token in enumerate(operation_tokens, start=1):
            operation = _parse_symmetry_operation(
                token,
                f"op:{index}",
                diagnostics,
            )
            if operation is not None:
                parsed.append(operation)
        if len(parsed) != len(operation_tokens):
            return None
        operations = tuple(parsed)

    number = None
    number_text = _scalar_text(block, names.IT_NUMBER)
    if number_text is not None:
        try:
            number = int(float(number_text))
        except ValueError:
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.space_group_number_invalid",
                    f"Invalid reported space-group number: {number_text!r}",
                    _scalar(block, names.IT_NUMBER).value_token.span,
                )
            )

    definition = SpaceGroupDefinition(
        operations=operations,
        provenance=provenance,
        number=number,
        hm_symbol=_scalar_text(block, names.HM_SYMBOL),
        hall_symbol=_scalar_text(block, names.HALL_SYMBOL),
        setting=_scalar_text(block, names.SETTING),
        origin_choice=_scalar_text(block, names.ORIGIN_CHOICE),
    )
    if catalog_setting is not None and not _same_operation_set(
        operations,
        catalog_setting.symmetry_operations,
    ):
        diagnostics.append(
            Diagnostic(
                Severity.WARNING,
                "cif.map.space_group_operations_mismatch",
                "Reported symmetry operations disagree with the identified catalog setting; "
                "the explicit source operations were retained.",
                block.data_token.span,
            )
        )
        catalog_setting = None
    return definition, catalog_setting


def _optional_measured(
    token: CifToken | None,
    *,
    unit: str | None = None,
) -> MeasuredValue | None:
    if token is None or token.value in {"?", "."}:
        return None
    return parse_measured_value(token.raw, unit=unit)


def _optional_text(token: CifToken | None) -> str | None:
    if token is None or token.value in {"?", "."}:
        return None
    return token.value


def _optional_integer(token: CifToken | None) -> int | None:
    if token is None or token.value in {"?", "."}:
        return None
    value = float(token.value)
    if not value.is_integer():
        raise ValueError(f"Expected integer, got {token.value!r}")
    return int(value)


def _label_identity(site: IndependentSite) -> str:
    identity = site.label
    for component in site.components:
        if identity.casefold().startswith(component.element.casefold()):
            identity = identity[len(component.element) :]
            break
    return identity.casefold()


def _coincident(
    left: IndependentSite,
    right: IndependentSite,
    tolerance: float = 1e-8,
) -> bool:
    return all(
        abs((float(a.value) - float(b.value) + 0.5) % 1.0 - 0.5) <= tolerance
        for a, b in zip(left.fractional, right.fractional, strict=True)
    )


def _can_merge_mixed(group: list[IndependentSite], candidate: IndependentSite) -> bool:
    sites = [*group, candidate]
    components = [component for site in sites for component in site.components]
    if len({component.element for component in components}) != len(components):
        return False
    if any(
        component.occupancy.raw is None
        or component.occupancy.value is None
        or not 0 <= component.occupancy.value < 1
        for component in components
    ):
        return False
    if len({site.displacement for site in sites}) > 1:
        return False

    assemblies = {site.disorder_assembly for site in sites}
    groups = {site.disorder_group for site in sites}
    explicit_disorder = None not in assemblies and len(assemblies) == 1 and len(groups) == 1
    matching_identity = len({_label_identity(site) for site in sites}) == 1
    return explicit_disorder or matching_identity


def _merge_coincident_sites(
    sites: tuple[IndependentSite, ...],
    diagnostics: list[Diagnostic],
) -> tuple[IndependentSite, ...] | None:
    consumed: set[int] = set()
    merged: list[IndependentSite] = []
    for index, site in enumerate(sites):
        if index in consumed:
            continue
        group = [site]
        for candidate_index in range(index + 1, len(sites)):
            if candidate_index in consumed:
                continue
            candidate = sites[candidate_index]
            if not _coincident(site, candidate):
                continue
            if _can_merge_mixed(group, candidate):
                group.append(candidate)
                consumed.add(candidate_index)
            else:
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.coincident_sites_unmerged",
                        f"Coincident sites {site.label} and {candidate.label} were not merged",
                    )
                )

        if len(group) == 1:
            merged.append(site)
            continue
        wyckoff_values = {item.wyckoff for item in group}
        multiplicities = {item.reported_multiplicity for item in group}
        source_rows = tuple(item.metadata["source_row"] for item in group)
        components = tuple(
            component
            for item in group
            for component in item.components
        )
        total_occupancy = math.fsum(
            float(component.occupancy.value) for component in components
        )
        if total_occupancy > 1.0 + 1e-12:
            components = tuple(
                replace(
                    component,
                    occupancy=replace(
                        component.occupancy,
                        value=float(component.occupancy.value) / total_occupancy,
                    ),
                )
                for component in components
            )
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.occupancy_total_normalized",
                    (
                        f"Coincident site components total {total_occupancy:g}; "
                        "occupancies were normalized proportionally"
                    ),
                    recovery="proportional normalization to total occupancy 1.0",
                )
            )
        merged.append(
            replace(
                site,
                id=f"{site.id.rsplit(':', 1)[0]}:mixed:{','.join(map(str, source_rows))}",
                label="/".join(item.label for item in group),
                components=components,
                wyckoff=next(iter(wyckoff_values)) if len(wyckoff_values) == 1 else None,
                reported_multiplicity=(
                    next(iter(multiplicities)) if len(multiplicities) == 1 else None
                ),
                metadata={"source_rows": source_rows},
            )
        )
    return tuple(merged)


def _reported_coordinate_error(value: MeasuredValue) -> float:
    if value.uncertainty is not None:
        return 3.0 * float(value.uncertainty)
    if value.raw is None:
        return 0.0
    numeric = value.raw.partition("(")[0]
    try:
        exponent = Decimal(numeric).as_tuple().exponent
    except InvalidOperation:
        return 0.0
    if exponent >= 0:
        return 0.0
    return 0.5 * float(Decimal(10) ** exponent)


def _periodic_delta(left: float, right: float) -> float:
    return (left - right + 0.5) % 1.0 - 0.5


def _operation_position(
    operation: AffineOperation,
    coordinates: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(operation.rotation, dtype=float)
    translation = np.asarray(operation.translation, dtype=float)
    return rotation @ coordinates + translation


def _normalize_special_position(
    site: IndependentSite,
    operations: tuple[AffineOperation, ...],
) -> tuple[IndependentSite, bool, float]:
    errors = tuple(_reported_coordinate_error(value) for value in site.fractional)
    observed = np.asarray([float(value.value) for value in site.fractional])
    stabilizer = []
    identity = np.eye(3)
    for operation in operations:
        rotation = np.asarray(operation.rotation, dtype=float)
        transformed = _operation_position(operation, observed)
        if all(
            abs(_periodic_delta(transformed[row], observed[row]))
            <= max(
                1e-12,
                math.nextafter(
                    math.fsum(
                        abs(rotation[row, column] - identity[row, column])
                        * errors[column]
                        for column in range(3)
                    ),
                    math.inf,
                ),
            )
            for row in range(3)
        ):
            stabilizer.append(operation)
    if len(stabilizer) <= 1:
        return site, False, 0.0

    rows: list[np.ndarray] = []
    targets: list[float] = []
    for operation in stabilizer:
        rotation = np.asarray(operation.rotation, dtype=float)
        translation = np.asarray(operation.translation, dtype=float)
        lattice = np.rint(rotation @ observed + translation - observed)
        for row, target in zip(
            rotation - identity,
            lattice - translation,
            strict=True,
        ):
            if not np.allclose(row, 0.0, rtol=0.0, atol=1e-15):
                rows.append(row)
                targets.append(float(target))
    if not rows:
        return site, False, 0.0
    constraints = np.stack(rows)
    target_vector = np.asarray(targets)
    residual = constraints @ observed - target_vector
    correction = constraints.T @ np.linalg.pinv(
        constraints @ constraints.T
    ) @ residual
    normalized = observed - correction
    if float(np.max(np.abs(constraints @ normalized - target_vector))) > 1e-10:
        return site, False, 0.0
    adjustment = float(np.max(np.abs(normalized - observed)))
    if any(
        abs(float(after - before)) > max(1e-12, error)
        for after, before, error in zip(normalized, observed, errors, strict=True)
    ):
        return site, False, 0.0
    fractional = tuple(
        value
        if math.isclose(float(value.value), float(coordinate), abs_tol=1e-15)
        else replace(value, value=float(coordinate), raw=None)
        for value, coordinate in zip(site.fractional, normalized, strict=True)
    )
    return replace(site, fractional=fractional), adjustment > 1e-15, adjustment


def _site_stabilizer(
    site: IndependentSite,
    operations: tuple[AffineOperation, ...],
) -> tuple[AffineOperation, ...]:
    coordinates = np.asarray([float(value.value) for value in site.fractional])
    return tuple(
        operation
        for operation in operations
        if max(
            abs(_periodic_delta(value, reference))
            for value, reference in zip(
                _operation_position(operation, coordinates),
                coordinates,
                strict=True,
            )
        )
        <= 1e-5
    )


def _attach_anisotropic_displacements(
    block: CifBlock,
    sites: tuple[IndependentSite, ...],
    diagnostics: list[Diagnostic],
) -> tuple[IndependentSite, ...]:
    required = (
        names.ANISO_LABEL[0],
        names.ANISO_U11[0],
        names.ANISO_U22[0],
        names.ANISO_U33[0],
        names.ANISO_U12[0],
        names.ANISO_U13[0],
        names.ANISO_U23[0],
    )
    loop = _find_loop(block, required)
    if loop is None:
        return sites

    by_label: dict[str, tuple[tuple[MeasuredValue, ...], CifToken]] = {}
    for row in loop.row_tokens:
        label_token = _token(row, loop, names.ANISO_LABEL)
        if label_token is None:
            continue
        value_tokens = tuple(
            _token(row, loop, aliases)
            for aliases in (
                names.ANISO_U11,
                names.ANISO_U22,
                names.ANISO_U33,
                names.ANISO_U12,
                names.ANISO_U13,
                names.ANISO_U23,
            )
        )
        try:
            values = tuple(
                parse_measured_value(token.raw, unit="angstrom^2")
                for token in value_tokens
                if token is not None
            )
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    Severity.ERROR,
                    "cif.map.adp_invalid",
                    str(exc),
                    label_token.span,
                )
            )
            continue
        if len(values) != 6 or any(value.value is None for value in values):
            diagnostics.append(
                Diagnostic(
                    Severity.ERROR,
                    "cif.map.adp_incomplete",
                    f"Anisotropic tensor is incomplete for {label_token.value}",
                    label_token.span,
                )
            )
            continue
        by_label[label_token.value.casefold()] = (values, label_token)

    updated = []
    for site in sites:
        item = by_label.get(site.label.casefold())
        if item is None:
            updated.append(site)
            continue
        values, label_token = item
        u11, u22, u33, u12, u13, u23 = values
        tensor = (
            (u11, u12, u13),
            (u12, u22, u23),
            (u13, u23, u33),
        )
        numeric = np.array(
            [[float(value.value) for value in row] for row in tensor],
            dtype=float,
        )
        if float(np.linalg.eigvalsh(numeric).min()) < -1e-12:
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.adp_not_positive_semidefinite",
                    f"Reported anisotropic tensor is not positive semidefinite for {site.label}",
                    label_token.span,
                )
            )
        updated.append(
            replace(
                site,
                displacement=DisplacementParameters(
                    kind="U_aniso",
                    isotropic=(
                        site.displacement.isotropic
                        if site.displacement is not None
                        and site.displacement.kind == "U_iso"
                        else None
                    ),
                    tensor=tensor,
                    reported_kind="U",
                ),
            )
        )
    return tuple(updated)


def _sites(
    block: CifBlock,
    diagnostics: list[Diagnostic],
) -> tuple[IndependentSite, ...] | None:
    atom_loop = _find_loop(
        block,
        (
            names.ATOM_LABEL[0],
            names.FRACT_X[0],
            names.FRACT_Y[0],
            names.FRACT_Z[0],
        ),
    )
    if atom_loop is None:
        diagnostics.append(
            Diagnostic(
                Severity.ERROR,
                "cif.map.atom_loop_missing",
                f"Block {block.name!r} has no complete fractional atom-site loop",
                block.data_token.span,
            )
        )
        return None

    sites: list[IndependentSite] = []
    for row_index, row in enumerate(atom_loop.row_tokens):
        label_token = _token(row, atom_loop, names.ATOM_LABEL)
        coordinate_tokens = tuple(
            _token(row, atom_loop, aliases)
            for aliases in (names.FRACT_X, names.FRACT_Y, names.FRACT_Z)
        )
        try:
            coordinates = tuple(
                parse_measured_value(token.raw)
                for token in coordinate_tokens
                if token is not None
            )
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.coordinate_invalid",
                    str(exc),
                    label_token.span if label_token is not None else block.data_token.span,
                    recovery="Atom row omitted; remaining valid sites were preserved.",
                )
            )
            continue
        if len(coordinates) != 3 or any(value.value is None for value in coordinates):
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.coordinate_missing",
                    f"Atom row {row_index + 1} has incomplete fractional coordinates",
                    label_token.span if label_token is not None else block.data_token.span,
                    recovery="Atom row omitted; remaining valid sites were preserved.",
                )
            )
            continue

        label = label_token.value if label_token is not None else f"site{row_index + 1}"
        type_token = _token(row, atom_loop, names.ATOM_TYPE)
        try:
            element = _cif_species(type_token.value if type_token is not None else label)
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    Severity.ERROR,
                    "cif.map.element_invalid",
                    str(exc),
                    (type_token or label_token).span,
                )
            )
            continue

        occupancy_token = _token(row, atom_loop, names.OCCUPANCY)
        reported_occupancy_value: float | None = None
        if occupancy_token is None:
            occupancy = MeasuredValue(1.0, None, None)
            diagnostics.append(
                Diagnostic(
                    Severity.WARNING,
                    "cif.map.occupancy_defaulted",
                    f"Occupancy is absent for {label}; using CIF default 1",
                    label_token.span,
                    recovery="1.0",
                )
            )
        else:
            try:
                occupancy = parse_measured_value(occupancy_token.raw)
            except ValueError as exc:
                occupancy = MeasuredValue(1.0, None, occupancy_token.raw)
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.occupancy_invalid",
                        f"{exc}; using occupancy 1.0 for {label}",
                        occupancy_token.span,
                        recovery="1.0",
                    )
                )
            if occupancy.value is None:
                occupancy = replace(occupancy, value=1.0)
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.occupancy_missing",
                        f"Occupancy is unknown for {label}; using 1.0",
                        occupancy_token.span,
                        recovery="1.0",
                    )
                )
            if occupancy.value > 1:
                reported_occupancy_value = float(occupancy.value)
                occupancy = replace(occupancy, value=1.0)
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.occupancy_normalized_to_one",
                        (
                            f"Occupancy exceeds one for {label}: "
                            f"{reported_occupancy_value}; using 1.0"
                        ),
                        occupancy_token.span,
                        recovery="1.0",
                    )
                )
            elif occupancy.value < 0:
                reported_occupancy_value = float(occupancy.value)
                occupancy = replace(occupancy, value=0.0)
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.occupancy_normalized_to_zero",
                        (
                            f"Occupancy is below zero for {label}: "
                            f"{reported_occupancy_value}; using 0.0"
                        ),
                        occupancy_token.span,
                        recovery="0.0",
                    )
                )

        oxidation_token = _token(row, atom_loop, names.OXIDATION)
        multiplicity_token = _token(row, atom_loop, names.MULTIPLICITY)
        u_iso_token = _token(row, atom_loop, names.U_ISO)
        b_iso_token = _token(row, atom_loop, names.B_ISO)
        try:
            oxidation = _optional_measured(oxidation_token)
            reported_multiplicity = _optional_integer(multiplicity_token)
            u_iso = _optional_measured(u_iso_token, unit="angstrom^2")
            b_iso = _optional_measured(b_iso_token, unit="angstrom^2")
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    Severity.ERROR,
                    "cif.map.site_value_invalid",
                    f"Invalid reported value for {label}: {exc}",
                    label_token.span,
                )
            )
            continue

        displacement = None
        if u_iso is not None:
            displacement = DisplacementParameters(
                kind="U_iso",
                isotropic=u_iso,
                reported_kind="U",
            )
        elif b_iso is not None:
            displacement = DisplacementParameters(
                kind="B_iso",
                isotropic=b_iso,
                reported_kind="B",
            )

        component_metadata: dict[str, object] = {
            "reported_type_symbol": (
                type_token.value if type_token is not None else None
            ),
        }
        if reported_occupancy_value is not None:
            component_metadata["reported_occupancy_value"] = (
                reported_occupancy_value
            )

        try:
            sites.append(
                IndependentSite(
                    id=f"{block.name}:{label}:{row_index}",
                    label=label,
                    components=(
                        SiteComponent(
                            element,
                            occupancy,
                            oxidation_state=oxidation,
                            metadata=component_metadata,
                        ),
                    ),
                    fractional=coordinates,
                    wyckoff=_optional_text(_token(row, atom_loop, names.WYCKOFF)),
                    reported_multiplicity=reported_multiplicity,
                    disorder_assembly=_optional_text(
                        _token(row, atom_loop, names.DISORDER_ASSEMBLY)
                    ),
                    disorder_group=_optional_text(
                        _token(row, atom_loop, names.DISORDER_GROUP)
                    ),
                    displacement=displacement,
                    metadata={"source_row": row_index},
                )
            )
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    Severity.ERROR,
                    "cif.map.site_invalid",
                    f"Invalid atom site {label}: {exc}",
                    label_token.span,
                )
            )
            continue
    with_anisotropic = _attach_anisotropic_displacements(
        block,
        tuple(sites),
        diagnostics,
    )
    return _merge_coincident_sites(with_anisotropic, diagnostics)


def _metadata(block: CifBlock) -> dict[str, object]:
    values: dict[str, object] = {"cif_block": block.name}
    for key, aliases in names.METADATA.items():
        value = _scalar_text(block, aliases)
        if value is not None:
            values[key] = value
    return values


def _isotropic_adp_fallback(
    displacement: DisplacementParameters | None,
) -> DisplacementParameters | None:
    """Retain reported U_iso_or_equiv when anisotropic ADP is unusable."""

    if (
        displacement is None
        or displacement.kind != "U_aniso"
        or displacement.isotropic is None
        or displacement.isotropic.value is None
    ):
        return None
    return DisplacementParameters(
        kind="U_iso",
        isotropic=displacement.isotropic,
        reported_kind="U_iso_or_equiv",
    )


def _site_orbit_result(
    site: IndependentSite,
    symmetry: SpaceGroupDefinition,
    catalog_setting: SpaceGroupSetting | None,
    cell: UnitCell,
    structure_id: str,
) -> tuple[int, tuple[Diagnostic, ...]]:
    if catalog_setting is None:
        expanded = expand_orbit(
            site,
            symmetry.operations,
            cell=cell,
            structure_id=structure_id,
        )
        return len(expanded), ()
    orbit = build_orbit(
        site,
        catalog_setting,
        cell=cell,
        structure_id=structure_id,
    )
    assignment = assign_wyckoff(orbit, catalog_setting)
    return orbit.multiplicity, assignment.diagnostics


def map_cif_structures(
    document: CifDocument,
    *,
    crystallography: SpaceGroupCatalog | None = None,
) -> tuple[tuple[CrystalStructure, ...], tuple[Diagnostic, ...]]:
    """Map every structural CIF block to a canonical asymmetric-unit crystal."""

    structures: list[CrystalStructure] = []
    diagnostics: list[Diagnostic] = []
    catalog = SpaceGroupCatalog.default() if crystallography is None else crystallography
    for block in document.blocks:
        if not _looks_structural(block):
            continue
        cell = _cell(block, diagnostics)
        if cell is None:
            continue
        symmetry_result = _symmetry(block, diagnostics, catalog)
        if symmetry_result is None:
            continue
        symmetry, catalog_setting = symmetry_result
        sites = _sites(block, diagnostics)
        if sites is None:
            continue
        structure_id = f"cif:{block.name}"
        checked_sites: list[IndependentSite] = []
        block_failed = False
        for site in sites:
            try:
                site, coordinates_changed, coordinate_adjustment = (
                    _normalize_special_position(site, symmetry.operations)
                )
                if coordinates_changed:
                    diagnostics.append(
                        Diagnostic(
                            Severity.WARNING,
                            "cif.map.special_position_symmetrized",
                            f"{site.label}: fractional coordinates were projected "
                            f"onto the space-group site symmetry "
                            f"(maximum adjustment {coordinate_adjustment:.8g}).",
                        )
                    )
                stabilizer = _site_stabilizer(site, symmetry.operations)
                try:
                    symmetrized = symmetrize_displacement(
                        site.displacement,
                        stabilizer,
                    )
                except SymmetryConsistencyError as error:
                    fallback = _isotropic_adp_fallback(site.displacement)
                    site = replace(site, displacement=fallback)
                    diagnostics.append(
                        Diagnostic(
                            Severity.WARNING,
                            "cif.map.adp_symmetry_inconsistent",
                            f"{site.label}: {error}; anisotropic displacement was "
                            + (
                                "replaced by reported U_iso_or_equiv."
                                if fallback is not None
                                else "omitted while retaining the coordinate site."
                            ),
                        )
                    )
                    symmetrized = None
                if symmetrized is not None and symmetrized.changed:
                    site = replace(site, displacement=symmetrized.displacement)
                    diagnostics.append(
                        Diagnostic(
                            Severity.WARNING,
                            "cif.map.adp_symmetrized",
                            f"{site.label}: anisotropic displacement was projected "
                            f"onto the site symmetry within its reported uncertainty "
                            f"(maximum adjustment {symmetrized.max_adjustment:.8g} "
                            "angstrom^2).",
                        )
                    )
                try:
                    calculated_multiplicity, orbit_diagnostics = _site_orbit_result(
                        site,
                        symmetry,
                        catalog_setting,
                        cell,
                        structure_id,
                    )
                except SymmetryConsistencyError as error:
                    fallback = _isotropic_adp_fallback(site.displacement)
                    site = replace(site, displacement=fallback)
                    diagnostics.append(
                        Diagnostic(
                            Severity.WARNING,
                            "cif.map.adp_symmetry_inconsistent",
                            f"{site.label}: {error}; anisotropic displacement was "
                            + (
                                "replaced by reported U_iso_or_equiv."
                                if fallback is not None
                                else "omitted while retaining the coordinate site."
                            ),
                        )
                    )
                    calculated_multiplicity, orbit_diagnostics = _site_orbit_result(
                        site,
                        symmetry,
                        catalog_setting,
                        cell,
                        structure_id,
                    )
                diagnostics.extend(orbit_diagnostics)
            except SymmetryConsistencyError as error:
                diagnostics.append(
                    Diagnostic(
                        Severity.ERROR,
                        "cif.map.adp_symmetry_inconsistent",
                        f"{block.name}: {error}",
                    )
                )
                block_failed = True
                break
            checked_site = replace(
                site,
                calculated_multiplicity=calculated_multiplicity,
            )
            if (
                catalog_setting is None
                and
                site.reported_multiplicity is not None
                and site.reported_multiplicity != calculated_multiplicity
            ):
                diagnostics.append(
                    Diagnostic(
                        Severity.WARNING,
                        "cif.map.multiplicity_mismatch",
                        f"{site.label}: reported multiplicity {site.reported_multiplicity}, "
                        f"calculated {calculated_multiplicity}",
                    )
                )
            checked_sites.append(checked_site)
        if block_failed:
            continue
        sites = tuple(checked_sites)
        structures.append(
            CrystalStructure(
                name=block.name,
                cell=cell,
                sites=sites,
                id=structure_id,
                space_group=symmetry,
                formula=_scalar_text(block, names.FORMULA),
                metadata=_metadata(block),
            )
        )
    return tuple(structures), tuple(diagnostics)
