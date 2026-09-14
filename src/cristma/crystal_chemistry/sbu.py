"""Conservative single-node secondary-building-unit calculation."""

from __future__ import annotations

from dataclasses import dataclass

from cristma.crystallography import identity_relation
from cristma.crystallography.pair_canonical import periodic_endpoint_instance
from cristma.crystallography.symmetry_context import _digest

from .intermolecular import molecular_component_stabilizer
from .models import ResolutionStatus
from .reticular_models import LigandRole, ReticularPrecursorGraph
from .sbu_models import (
    SBUAtomMember,
    SBUAtomSourceRole,
    SecondaryBuildingUnitOrbit,
    SecondaryBuildingUnitResult,
)


def _status(*statuses: ResolutionStatus) -> ResolutionStatus:
    if ResolutionStatus.INCOMPLETE in statuses:
        return ResolutionStatus.INCOMPLETE
    if ResolutionStatus.AMBIGUOUS in statuses:
        return ResolutionStatus.AMBIGUOUS
    if statuses and all(item is ResolutionStatus.NOT_APPLICABLE for item in statuses):
        return ResolutionStatus.NOT_APPLICABLE
    return ResolutionStatus.RESOLVED


@dataclass(frozen=True, slots=True)
class SecondaryBuildingUnitAnalyzer:
    """Build one reproducible SBU orbit per precursor coordination node."""

    def get_config(self) -> dict[str, object]:
        return {}

    def clone(self, **changes: object) -> "SecondaryBuildingUnitAnalyzer":
        if changes:
            raise TypeError(
                "unknown SecondaryBuildingUnitAnalyzer configuration: "
                + ", ".join(sorted(changes))
            )
        return self

    def analyze(
        self, precursor_graph: ReticularPrecursorGraph
    ) -> SecondaryBuildingUnitResult:
        if not isinstance(precursor_graph, ReticularPrecursorGraph):
            raise TypeError("precursor_graph must be ReticularPrecursorGraph")

        configuration: tuple[tuple[str, object], ...] = ()
        provenance = (("method", "cristma.secondary_building_units:1"),)
        if not precursor_graph.node_orbits:
            return SecondaryBuildingUnitResult(
                precursor_graph,
                (),
                precursor_graph.status,
                precursor_graph.diagnostics,
                configuration,
                provenance,
                precursor_graph.fingerprint,
            )

        contact_result = precursor_graph._contact_analysis
        molecular_result = precursor_graph._molecular_analysis
        context = contact_result._symmetry_context
        mapping = contact_result._asymmetric_unit_mapping
        structure_site_by_id = {
            site.id: site for site in contact_result._structure.sites
        }
        component_by_id = {
            item.component_orbit_id: item
            for item in molecular_result.molecule_orbits
        }
        ligand_by_id = {
            item.ligand_orbit_id: item for item in precursor_graph.ligand_orbits
        }
        connection_by_id = {
            item.connection_orbit_id: item
            for item in precursor_graph.connection_orbits
        }
        contact_by_id = {
            item.resolved_contact_orbit_id: item
            for item in contact_result.contact_orbits
        }

        terminal_owner_by_ligand: dict[str, str] = {}
        for ligand in precursor_graph.ligand_orbits:
            if (
                ligand.role is not LigandRole.TERMINAL
                or ligand.status is not ResolutionStatus.RESOLVED
            ):
                continue
            node_ids = {
                connection_by_id[item].node_orbit_id
                for item in ligand.connection_orbit_ids
            }
            if len(node_ids) != 1:
                raise ValueError("resolved terminal ligand must belong to one SBU")
            terminal_owner_by_ligand[ligand.ligand_orbit_id] = next(iter(node_ids))

        sbus: list[SecondaryBuildingUnitOrbit] = []
        for node in precursor_graph.node_orbits:
            center_site = structure_site_by_id[node.center_independent_site_id]
            member_by_key = {
                (
                    center_site.id,
                    identity_relation(context),
                    SBUAtomSourceRole.NODE,
                ): SBUAtomMember(
                    center_site.id,
                    identity_relation(context),
                    1,
                    center_site.components,
                    SBUAtomSourceRole.NODE,
                )
            }
            terminal_ids = tuple(sorted(
                ligand_id
                for ligand_id, owner_node_id in terminal_owner_by_ligand.items()
                if owner_node_id == node.node_orbit_id
            ))
            source_connection_ids = tuple(sorted(
                connection_id
                for ligand_id in terminal_ids
                for connection_id in ligand_by_id[ligand_id].connection_orbit_ids
            ))
            source_contact_ids = set(node.source_resolved_contact_orbit_ids)
            statuses = [node.status]
            for ligand_id in terminal_ids:
                ligand = ligand_by_id[ligand_id]
                component = component_by_id[ligand.source_component_orbit_id]
                component_images = dict(component.image_relations)
                component_stabilizers = molecular_component_stabilizer(
                    component, context
                )
                statuses.append(ligand.status)
                for connection_id in ligand.connection_orbit_ids:
                    connection = connection_by_id[connection_id]
                    if connection.node_orbit_id != node.node_orbit_id:
                        raise ValueError("terminal ligand connection belongs to another SBU")
                    statuses.append(connection.status)
                    source_contact_ids.update(
                        connection.source_resolved_contact_orbit_ids
                    )
                    for contact_id in connection.source_resolved_contact_orbit_ids:
                        statuses.append(contact_by_id[contact_id].status)
                    physical_relations = {}
                    for node_to_ligand in connection.equivalent_node_ligand_relations:
                        for internal in component_stabilizers:
                            for site_id in component.independent_site_ids:
                                node_to_atom = (
                                    node_to_ligand
                                    .compose(internal, context)
                                    .compose(component_images[site_id], context)
                                )
                                endpoint = periodic_endpoint_instance(
                                    site_id, node_to_atom, mapping
                                )
                                previous = physical_relations.get(endpoint)
                                if previous is None or node_to_atom < previous:
                                    physical_relations[endpoint] = node_to_atom
                    for endpoint, node_to_atom in sorted(physical_relations.items()):
                        site_id = endpoint[0]
                        site = structure_site_by_id[site_id]
                        key = (
                            site_id,
                            node_to_atom,
                            SBUAtomSourceRole.TERMINAL_LIGAND,
                        )
                        member_by_key[key] = SBUAtomMember(
                            site_id,
                            node_to_atom,
                            1,
                            site.components,
                            SBUAtomSourceRole.TERMINAL_LIGAND,
                        )

            members = tuple(sorted(
                member_by_key.values(),
                key=lambda item: (
                    item.source_role.value,
                    item.independent_site_id,
                    item.periodic_relation,
                ),
            ))
            source_contacts = tuple(sorted(source_contact_ids))
            sbu_id = "secondary-building-unit-orbit:" + _digest(
                {
                    "node_orbit_id": node.node_orbit_id,
                    "terminal_ligand_orbit_ids": terminal_ids,
                    "source_connection_orbit_ids": source_connection_ids,
                    "source_contact_orbit_ids": source_contacts,
                    "atom_members": tuple(
                        (
                            item.independent_site_id,
                            item.periodic_relation.operation_key,
                            item.periodic_relation.lattice_translation,
                            item.multiplicity_per_sbu,
                            item.source_role.value,
                        )
                        for item in members
                    ),
                }
            )
            sbus.append(
                SecondaryBuildingUnitOrbit(
                    sbu_id,
                    (node.node_orbit_id,),
                    terminal_ids,
                    source_connection_ids,
                    source_contacts,
                    members,
                    node.multiplicity_in_reference_cell,
                    _status(*statuses),
                    tuple(dict.fromkeys(
                        diagnostic
                        for ligand_id in terminal_ids
                        for diagnostic in ligand_by_id[ligand_id].diagnostics
                    )),
                    (("method", "cristma.conservative_single_node_sbu:1"),),
                )
            )

        ordered = tuple(sorted(sbus, key=lambda item: item.sbu_orbit_id))
        result_status = _status(
            precursor_graph.status,
            *(item.status for item in ordered),
        )
        return SecondaryBuildingUnitResult(
            precursor_graph,
            ordered,
            result_status,
            precursor_graph.diagnostics,
            configuration,
            provenance,
            precursor_graph.fingerprint,
        )


__all__ = ["SecondaryBuildingUnitAnalyzer"]
