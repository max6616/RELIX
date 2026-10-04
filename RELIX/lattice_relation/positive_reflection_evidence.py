"""Frozen RELIX inference definitions; release packaging selects only required symbols."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from cctbx import crystal, sgtbx

from cctbx.sgtbx import lattice_symmetry

from lattice_aligned.reference_geometry import cell_from_indexing_matrix

def group_key(group):
    return tuple(sorted(op.as_xyz() for op in group))

@lru_cache(None)
def reference_templates():
    """All530 conventional settings, normalized by their Patterson symmetry."""
    templates = {}
    seen = set()
    for symbol in sgtbx.space_group_symbol_iterator():
        group = sgtbx.space_group_info(symbol=symbol.universal_hermann_mauguin()).group()
        patterson = group.build_derived_patterson_group()
        transform = patterson.type().cb_op()
        reference = group.change_basis(transform)
        key = group_key(reference.build_derived_patterson_group())
        # Some equal-axis settings (e.g. the two Pa-3 extinction orientations)
        # are not separate entries in the conventional-symbol iterator.
        # Complete their finite proper Euclidean-normalizer orbit, preserving
        # the exact Patterson group and space-group number.
        normalizer = reference.build_derived_patterson_group().type().expand_addl_generators_of_euclidean_normalizer(True,True)
        for operation in normalizer:
            if np.linalg.det(np.asarray(operation.r().as_double()).reshape(3,3))<0:
                continue
            conjugated = reference.change_basis(sgtbx.change_of_basis_op(operation))
            if group_key(conjugated.build_derived_patterson_group()) != key:
                continue
            assert conjugated.type().number() == symbol.number()
            signature = (symbol.number(), group_key(conjugated))
            if signature in seen:
                continue
            seen.add(signature)
            templates.setdefault(key, []).append((symbol.number(), conjugated,
                symbol.universal_hermann_mauguin()+';normalizer='+operation.as_xyz()))
    return templates

def metric_settings(basis, max_delta=1.):
    """Embed every compatible Laue subgroup and conventional extinction setting."""
    cell = cell_from_indexing_matrix(np.linalg.inv(np.asarray(basis, float)))
    symmetry = crystal.symmetry(unit_cell=tuple(cell), space_group_symbol='P1')
    subgroups = lattice_symmetry.metric_subgroups(symmetry, max_delta=max_delta,
                                                  bravais_types_only=False)
    templates = reference_templates()
    result = {}
    seen = set()
    for subgroup in subgroups.result_groups:
        best = subgroup['best_subsym']
        to_reference = best.space_group_info().type().cb_op()
        key = group_key(best.space_group().change_basis(to_reference))
        transform = to_reference * subgroup['cb_op_inp_best']
        for number, template, name in templates.get(key, []):
            candidate = template.change_basis(transform.inverse())
            signature = (number, group_key(candidate))
            if signature in seen:
                continue
            seen.add(signature)
            result.setdefault(number, []).append(dict(group=candidate,
                template=name, input_to_reference=str(transform),
                metric_delta=float(subgroup['max_angular_difference'])))
    return result
