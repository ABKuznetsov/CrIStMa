"""Outward-only materialization of molecular packing graph orbits."""

from __future__ import annotations

from dataclasses import dataclass

from cristma.crystallography import PeriodicSymmetryRelation
from cristma.crystallography.pair_canonical import periodic_endpoint_instance
from cristma.crystallography.symmetry_context import _digest
from cristma.structure import PeriodicAtomRef

from .materialization import CellRange, ReferenceCell
from .intermolecular import molecular_component_stabilizer
from .packing_models import (
    MolecularPackingGraph,
    PackingInteractionKind,
    PackingInteractionRef,
)


Cell = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MaterializedMolecule:
    molecule_id: str
    molecule_orbit_id: str
    periodic_relation: PeriodicSymmetryRelation
    owner_cell_translation: Cell
    atom_refs: tuple[PeriodicAtomRef, ...]
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.molecule_id or not self.molecule_orbit_id or not self.atom_refs:
            raise ValueError("materialized molecule requires identities and atoms")
        keys = tuple((item.atom_id, item.cell_translation) for item in self.atom_refs)
        if len(set(keys)) != len(keys):
            raise ValueError("materialized molecule atom references must be unique")


@dataclass(frozen=True, slots=True)
class MaterializedPackingConnection:
    connection_id: str
    packing_connection_orbit_id: str
    first_molecule_id: str
    second_molecule_id: str
    interaction_refs: tuple[PackingInteractionRef, ...]
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not all((
            self.connection_id,
            self.packing_connection_orbit_id,
            self.first_molecule_id,
            self.second_molecule_id,
        )):
            raise ValueError("materialized packing connection requires identities")
        if not self.interaction_refs:
            raise ValueError("materialized packing connection requires evidence")


@dataclass(frozen=True, slots=True)
class MaterializedPackingGraph:
    molecules: tuple[MaterializedMolecule, ...]
    connections: tuple[MaterializedPackingConnection, ...]
    provenance: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        molecule_ids = tuple(item.molecule_id for item in self.molecules)
        connection_ids = tuple(item.connection_id for item in self.connections)
        if molecule_ids != tuple(sorted(set(molecule_ids))):
            raise ValueError("materialized molecule IDs must be unique and sorted")
        if connection_ids != tuple(sorted(set(connection_ids))):
            raise ValueError("materialized connection IDs must be unique and sorted")
        known = set(molecule_ids)
        if any(
            item.first_molecule_id not in known or item.second_molecule_id not in known
            for item in self.connections
        ):
            raise ValueError("materialized connection references an unknown molecule")


@dataclass(frozen=True, slots=True)
class _Instance:
    molecule_orbit_id: str
    periodic_relation: PeriodicSymmetryRelation
    owner_cell: Cell
    atom_refs: tuple[PeriodicAtomRef, ...]


def _add(left: Cell, right: Cell) -> Cell:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def _negative(value: Cell) -> Cell:
    return tuple(-item for item in value)  # type: ignore[return-value]


def _shift_instance(instance: _Instance, shift: Cell, context) -> _Instance:
    action = PeriodicSymmetryRelation(context.identity_operation_key, shift)
    return _Instance(
        instance.molecule_orbit_id,
        action.compose(instance.periodic_relation, context),
        _add(instance.owner_cell, shift),
        tuple(
            PeriodicAtomRef(item.atom_id, _add(item.cell_translation, shift))
            for item in instance.atom_refs
        ),
    )


def _instance(component, relation, mapping, context) -> _Instance:
    refs = set()
    stabilizer = molecular_component_stabilizer(component, context)
    for site_id, site_relation in component.image_relations:
        for internal in stabilizer:
            endpoint = periodic_endpoint_instance(
                site_id,
                relation.compose(internal, context).compose(site_relation, context),
                mapping,
            )
            refs.add(PeriodicAtomRef(endpoint[1], endpoint[2]))
    ordered = tuple(sorted(refs, key=lambda item: (item.atom_id, item.cell_translation)))
    variants = []
    for anchor in sorted({item.cell_translation for item in ordered}):
        shifted = tuple(
            PeriodicAtomRef(item.atom_id, _add(item.cell_translation, _negative(anchor)))
            for item in ordered
        )
        variants.append((
            tuple((item.atom_id, item.cell_translation) for item in shifted),
            anchor,
            shifted,
        ))
    _, owner_cell, _ = min(variants)
    return _Instance(
        component.component_orbit_id,
        relation,
        owner_cell,
        ordered,
    )


def _instance_key(instance: _Instance):
    return (
        instance.molecule_orbit_id,
        tuple((item.atom_id, item.cell_translation) for item in instance.atom_refs),
    )


def _molecule_owners(component, context, mapping):
    owners = {}
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        raw = _instance(component, action, mapping, context)
        instance = _shift_instance(raw, _negative(raw.owner_cell), context)
        owners[_instance_key(instance)] = instance
    if len(owners) != component.multiplicity_in_reference_cell:
        raise ValueError("materialized molecule count disagrees with molecule orbit")
    return tuple(owners[key] for key in sorted(owners))


def _pair_owners(connection, components, context, mapping):
    owners = {}
    for operation_key in context.operation_keys:
        action = PeriodicSymmetryRelation(operation_key, (0, 0, 0))
        first = _instance(
            components[connection.first_molecule_orbit_id], action, mapping, context
        )
        second = _instance(
            components[connection.second_molecule_orbit_id],
            action.compose(connection.canonical_molecule_relation, context),
            mapping,
            context,
        )
        variants = []
        for owner, other in ((first, second), (second, first)):
            shift = _negative(owner.owner_cell)
            variants.append((
                (_instance_key(_shift_instance(owner, shift, context)),
                 _instance_key(_shift_instance(other, shift, context))),
                owner.owner_cell,
            ))
        _, owner_cell = min(variants, key=lambda item: item[0])
        normalized_first = _shift_instance(first, _negative(owner_cell), context)
        normalized_second = _shift_instance(second, _negative(owner_cell), context)
        key = min(
            (_instance_key(normalized_first), _instance_key(normalized_second)),
            (_instance_key(normalized_second), _instance_key(normalized_first)),
        )
        owners[key] = (normalized_first, normalized_second)
    if len(owners) != connection.multiplicity_in_reference_cell:
        raise ValueError("materialized connection count disagrees with packing orbit")
    return tuple(owners[key] for key in sorted(owners))


def _materialized_molecule(instance: _Instance) -> MaterializedMolecule:
    molecule_id = "molecule-instance:" + _digest({
        "molecule_orbit_id": instance.molecule_orbit_id,
        "atom_refs": tuple(
            (item.atom_id, item.cell_translation) for item in instance.atom_refs
        ),
    })
    return MaterializedMolecule(
        molecule_id,
        instance.molecule_orbit_id,
        instance.periodic_relation,
        instance.owner_cell,
        instance.atom_refs,
        (("materialization", "canonical_molecule_owner"),),
    )


@dataclass(frozen=True, slots=True)
class PackingMaterializer:
    """Project a packing graph into explicit molecule and connection instances."""

    def materialize(
        self,
        graph: MolecularPackingGraph,
        region: ReferenceCell | CellRange,
        *,
        interaction_kinds: tuple[PackingInteractionKind, ...] | None = None,
        connection_orbit_ids: tuple[str, ...] | None = None,
    ) -> MaterializedPackingGraph:
        if not isinstance(graph, MolecularPackingGraph):
            raise TypeError("graph must be MolecularPackingGraph")
        if not isinstance(region, (ReferenceCell, CellRange)):
            raise TypeError("region must be ReferenceCell or CellRange")
        selected_kinds = None if interaction_kinds is None else set(interaction_kinds)
        if selected_kinds is not None and any(
            not isinstance(item, PackingInteractionKind) for item in selected_kinds
        ):
            raise TypeError("interaction kinds must be PackingInteractionKind values")
        selected_orbits = (
            None if connection_orbit_ids is None else set(connection_orbit_ids)
        )
        source = graph._intermolecular_analysis._contact_analysis
        molecular = graph._intermolecular_analysis._molecular_analysis
        context = source._symmetry_context
        mapping = source._asymmetric_unit_mapping
        components = {
            item.component_orbit_id: item for item in molecular.molecule_orbits
        }
        molecule_instances = {}
        for component_id in graph.molecule_orbit_ids:
            for base in _molecule_owners(components[component_id], context, mapping):
                for owner_cell in region.owner_cells:
                    instance = _shift_instance(base, owner_cell, context)
                    molecule = _materialized_molecule(instance)
                    molecule_instances[molecule.molecule_id] = molecule

        connections = []
        for connection in graph.connection_orbits:
            if selected_orbits is not None and (
                connection.packing_connection_orbit_id not in selected_orbits
            ):
                continue
            refs = tuple(
                ref for ref in connection.interaction_refs
                if selected_kinds is None or ref.kind in selected_kinds
            )
            if not refs:
                continue
            for first_base, second_base in _pair_owners(
                connection, components, context, mapping
            ):
                for owner_cell in region.owner_cells:
                    first = _materialized_molecule(
                        _shift_instance(first_base, owner_cell, context)
                    )
                    second = _materialized_molecule(
                        _shift_instance(second_base, owner_cell, context)
                    )
                    molecule_instances[first.molecule_id] = first
                    molecule_instances[second.molecule_id] = second
                    connection_id = "packing-connection:" + _digest({
                        "packing_connection_orbit_id": (
                            connection.packing_connection_orbit_id
                        ),
                        "first_molecule_id": first.molecule_id,
                        "second_molecule_id": second.molecule_id,
                    })
                    connections.append(MaterializedPackingConnection(
                        connection_id,
                        connection.packing_connection_orbit_id,
                        first.molecule_id,
                        second.molecule_id,
                        refs,
                        (("materialization", "canonical_connection_owner"),),
                    ))
        return MaterializedPackingGraph(
            tuple(sorted(
                molecule_instances.values(), key=lambda item: item.molecule_id
            )),
            tuple(sorted(connections, key=lambda item: item.connection_id)),
            (("method", "cristma.packing_materializer:1"),),
        )


__all__ = [
    "MaterializedMolecule",
    "MaterializedPackingConnection",
    "MaterializedPackingGraph",
    "PackingMaterializer",
]
