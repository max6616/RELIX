"""Joint primitive-cell / orientation / HKL basis conversion for dataset labels.

This is a lattice indexing convention, not a claim about the atomic point group.
The returned full basis transform must accompany any crystal structure: direct
fractional sites transform by inv(hkl_transform).T. It is never an input feature.
This simulator module owns domain labels; machine learning consumers only read the exported labels.
"""
import numpy as np

CONVENTION = 'primitive-niggli-lattice-closest-identity-v1'


def canonical_primitive(crystal_symmetry, reciprocal_matrix, max_delta=.001):
    """Return one right-handed primitive Niggli basis and its exact HKL map.

    reciprocal_matrix maps column HKL to q (without 2*pi). The metric's proper
    lattice symmetry resolves equivalent orientations; it can exceed the actual
    structure symmetry, so this changes the structural setting as well as HKL.
    """
    from cctbx.sgtbx import lattice_symmetry
    source = np.asarray(reciprocal_matrix, dtype=float)
    cb = crystal_symmetry.change_of_basis_op_to_niggli_cell()
    reduced = crystal_symmetry.change_basis(cb)
    transform = np.asarray(cb.c_inv().r().as_double()).reshape(3, 3).T
    if np.linalg.det(source @ np.linalg.inv(transform)) < 0:
        transform = -transform
    A = source @ np.linalg.inv(transform)
    O = np.asarray(reduced.unit_cell().orthogonalization_matrix()).reshape(3, 3)
    U = A @ O.T
    if np.max(np.abs(U.T @ U - np.eye(3))) > 1e-4 or np.linalg.det(U) < .999:
        raise ValueError('Reciprocal matrix does not match the supplied unit-cell metric')
    operations = lattice_symmetry.group(reduced.unit_cell(), max_delta=max_delta)
    candidates = []
    for operation in operations:
        R = np.asarray(operation.r().as_double()).reshape(3, 3)
        if np.linalg.det(R) < .5:
            continue
        P = O @ R @ np.linalg.inv(O)
        if np.max(np.abs(P.T @ P - np.eye(3))) > 1e-4:
            raise ValueError('Approximate metric operation exceeds the numerical tolerance')
        candidate = U @ P
        key = (-round(float(np.trace(candidate)), 10), tuple(np.round(candidate.flat, 10)))
        candidates.append((key, candidate, R.T @ transform))
    _, orientation, total = min(candidates, key=lambda value: value[0])
    matrix = total @ np.linalg.inv(source)
    return dict(convention=CONVENTION, cell=np.array(reduced.unit_cell().parameters()),
                orientation=orientation, indexing_matrix=matrix, hkl_transform=total,
                fractional_site_transform=np.linalg.inv(total).T,
                proper_metric_operations=len(candidates), cctbx_niggli_change_of_basis=str(cb))


def site_change_of_basis(site_transform):
    """Exact zero-origin coordinate map, including the final metric-gauge choice."""
    from fractions import Fraction
    from cctbx import sgtbx
    expressions = []
    for row in site_transform:
        terms = []
        for value, axis in zip(row, 'xyz'):
            fraction = Fraction(float(value)).limit_denominator(192)
            if abs(float(fraction)-value) > 1e-9:
                raise ValueError('The site transform is not a small rational matrix')
            if fraction:
                terms.append(f'{fraction}*{axis}')
        expressions.append('+'.join(terms).replace('+-', '-') or '0')
    return sgtbx.change_of_basis_op(','.join(expressions))


def read_header(path):
    from pathlib import Path
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            key, separator, value = line[1:].partition(':')
            if separator:
                result[key.strip()] = value.strip()
    return result


def labels_complete(csv_path, label_base=None):
    """A label pair is usable only for exactly this source CSV and NPZ content."""
    import hashlib
    import json
    from pathlib import Path
    csv_path = Path(csv_path)
    base = Path(label_base) if label_base else csv_path.with_suffix('')
    try:
        meta = json.loads(base.with_suffix('.labels.json').read_text())
        return (meta['convention'] == CONVENTION and meta.get('label_schema') == 'primitive-training-labels-v1' and
                meta['source_csv_sha256'] == hashlib.sha256(csv_path.read_bytes()).hexdigest() and
                meta['labels_npz_sha256'] == hashlib.sha256(base.with_suffix('.labels.npz').read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError):
        return False


def write_lattice_labels(csv_path, crystal_symmetry=None, cif_path=None, label_base=None, overwrite=False):
    """Export jointly transformed labels beside an already written CSV.

    The CSV remains in its recorded source-CIF setting. The NPZ contains new
    labels; its JSON carries the full fractional-site and space-group setting
    transformation. Reading the written U and HKL guarantees batch export and
    generation-time export agree, including the CSV's declared rounding.
    """
    import hashlib
    import json
    import os
    import tempfile
    from pathlib import Path
    from iotbx import cif

    path = Path(csv_path)
    base = Path(label_base) if label_base else path.with_suffix('')
    npz_path, json_path = base.with_suffix('.labels.npz'), base.with_suffix('.labels.json')
    if labels_complete(path, base):
        return dict(status='verified-existing', source=str(path), labels=str(npz_path))
    if not overwrite and (npz_path.exists() or json_path.exists()):
        raise FileExistsError('Existing labels do not match their source; choose a new output directory')
    header = read_header(path)
    if (header.get('Angle convention') != 'dials-zero-based-array-index-v1' and
            header.get('Angle correction') != 'frame-index-offset-v1'):
        raise ValueError('Label export requires the fixed DIALS angle convention')
    if cif_path is None:
        copied = path.parent.parent/'input_cifs'/f'{path.parent.name}.cif'
        cif_path = copied if copied.exists() else Path(header['CIF path'])
    cif_path = Path(cif_path)
    if crystal_symmetry is None:
        structure = next(iter(cif.reader(file_path=str(cif_path)).build_crystal_structures().values()))
        crystal_symmetry = structure.crystal_symmetry()
    U = np.array([[float(value) for value in header[f'orientation_matrix row {i}'].split()] for i in range(3)])
    O = np.array(crystal_symmetry.unit_cell().orthogonalization_matrix()).reshape(3, 3)
    target = canonical_primitive(crystal_symmetry, U@np.linalg.inv(O).T)
    # Training equivalence classes are crystallographic labels, also exported
    # here so an ML data loader never computes a new crystallographic convention.
    from cctbx import uctbx
    from cctbx.sgtbx import lattice_symmetry
    cell = uctbx.unit_cell(tuple(target['cell']))
    reduced_O = np.array(cell.orthogonalization_matrix()).reshape(3, 3)
    lower = reduced_O.T
    hkl_operations, cartesian_operations = [], []
    for operation in lattice_symmetry.group(cell, max_delta=.001):
        R = np.array(operation.r().as_double()).reshape(3, 3)
        if np.linalg.det(R) > .5:
            hkl_operations.append(R.T)
            cartesian_operations.append(np.linalg.solve(lower, R.T@lower))
    columns = [value.strip() for value in header['Columns'].split(',')]
    indices = tuple(columns.index(value) for value in ('h', 'k', 'l'))
    source_hkl = np.loadtxt(path, delimiter=',', ndmin=2, usecols=indices)
    if not source_hkl.size:
        source_hkl = np.empty((0, 3))
    transformed = source_hkl@target['hkl_transform'].T
    nearest = np.rint(transformed)
    if transformed.size and np.max(np.abs(transformed-nearest)) > 1e-6:
        raise ValueError('The primitive transformation produced noninteger HKL')
    sg_number = crystal_symmetry.space_group_info().type().number()
    site_cb = site_change_of_basis(target['fractional_site_transform'])
    new_group = crystal_symmetry.space_group().change_basis(site_cb)
    if new_group.type().number() != sg_number:
        raise ValueError('Changing the structural setting changed the space-group type')
    arrays = {key:target[key] for key in ('cell', 'orientation', 'indexing_matrix', 'hkl_transform', 'fractional_site_transform')}
    arrays.update(hkl=nearest.astype(np.int64), crystal_system_index=np.array(np.searchsorted([2,15,74,142,167,194], sg_number)),
                  convention=np.array(CONVENTION), row_count=np.array(len(source_hkl)),
                  orthogonalization_matrix=reduced_O,
                  proper_metric_hkl_operations=np.array(hkl_operations),
                  proper_metric_cartesian_operations=np.array(cartesian_operations))
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(convention=CONVENTION, label_schema='primitive-training-labels-v1',
                    source_csv=str(path.resolve()), source_csv_sha256=source_hash,
                    source_angle_convention=header.get('Angle convention', header.get('Angle correction')),
                    source_cif=str(cif_path.resolve()), source_cif_sha256=hashlib.sha256(cif_path.read_bytes()).hexdigest(),
                    rows=len(source_hkl), source_cell=list(crystal_symmetry.unit_cell().parameters()), source_orientation=U.tolist(),
                    primitive_cell=target['cell'].tolist(), hkl_transform=target['hkl_transform'].tolist(),
                    fractional_site_transform=target['fractional_site_transform'].tolist(), fractional_origin_shift=[0.,0.,0.],
                    site_change_of_basis=str(site_cb), source_space_group_number=sg_number,
                    transformed_space_group_operations=[op.as_xyz() for op in new_group],
                    proper_metric_operations=target['proper_metric_operations'],
                    site_rule='x_new = fractional_site_transform @ x_source; original CIF sites and source space-group setting must not be reused unchanged',
                    labels_only=True, source_csv_unchanged=True,
                    convention_scope='Primitive lattice basis, including metric symmetry possibly larger than the atomic point group; the full structure setting transform is recorded')
    base.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=base.parent, suffix='.labels.npz.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        metadata['labels_npz_sha256'] = hashlib.sha256(Path(temporary).read_bytes()).hexdigest()
        os.replace(temporary, npz_path)
        descriptor, temporary = tempfile.mkstemp(dir=base.parent, suffix='.labels.json.tmp')
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(metadata, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, json_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if not labels_complete(path, base):
        raise ValueError('Label verification failed; source CSV may have changed during export')
    return dict(status='written', source=str(path), labels=str(npz_path), rows=len(source_hkl))
