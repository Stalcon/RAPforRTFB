"""Calculate how a redox state relaxes when the slow solvent response is frozen.

The fast solvent response follows the new density. The slow response keeps
the polarization from the initial density. In the Pekar partition,
G[D;D0] = E_vac[D] + 1/2 V Qopt V
 + V (Qstatic-Qopt) V0 - 1/2 V0 (Qstatic-Qopt) V0.
SMD adds the geometry-dependent CDS term once. The adapter evaluates energies
at fixed geometries.
"""
from pathlib import Path
import hashlib
import types
import numpy as np
from ase.io import read
from core import backend
from core.paths import HARTREE_EV
from core.protocol import ACTIVE_SP, file_hash, write_record
from .sp import make_mf, electronic_checks


def _surface_potential(solvent, density):
    if density.ndim == 3:
        density = density.sum(axis=0)
    return solvent.v_grids_n - solvent._get_v(density[None])[0]


def attach_frozen_slow(mf, reference_density, *, solvent='acetonitrile', order=41):
    on_gpu = backend.is_gpu_mean_field(mf)
    if on_gpu:
        import cupy as xp
        from gpu4pyscf.solvent.smd import SMD, solvent_db
    else:
        xp = np
        from pyscf.solvent.smd import SMD, solvent_db
    reference_density = xp.asarray(reference_density)
    if reference_density.shape[-2:] != (mf.mol.nao, mf.mol.nao):
        raise ValueError('reference density must use the target geometry and AO basis')
    descriptors = solvent_db[solvent]
    eps_static, eps_optical = float(descriptors[5]), float(descriptors[0]**2)
    if not 1 <= eps_optical <= eps_static:
        raise ValueError('invalid native SMD dielectric constants')
    cavities = []
    for epsilon in (eps_static, eps_optical):
        cavity = SMD(mf.mol, solvent=solvent)
        cavity.eps = epsilon
        cavity.lebedev_order = order
        cavity.build()
        cavity._get_vind(reference_density)
        cavities.append(cavity)
    static, optical = cavities
    for key in ('grid_coords', 'charge_exp'):
        if not xp.array_equal(static.surface[key], optical.surface[key]):
            raise ValueError('static and optical cavities must be identical')
    slow_charge = static._intermediates['q_sym'] - optical._intermediates['q_sym']
    slow_matrix = optical._get_vmat(slow_charge[None])[0]
    self_work = (_surface_potential(optical, reference_density) @ slow_charge) / 2
    native = optical._get_vind

    def potential(self, density):
        energy, matrix = native(density)
        return (energy + _surface_potential(self, density) @ slow_charge - self_work,
                matrix + slow_matrix)

    optical._get_vind = types.MethodType(potential, optical)
    result = mf.SMD(optical)
    host_density = xp.asnumpy(reference_density) if on_gpu else reference_density
    result.frozen_slow_context = {
        'partition': 'Pekar', 'solvent': solvent, 'eps_static': eps_static,
        'eps_optical': eps_optical, 'grid_order': order,
        'engine': 'gpu4pyscf' if on_gpu else 'pyscf',
        'reference_density_sha256': hashlib.sha256(host_density.tobytes()).hexdigest(),
        'CDS': 'native geometry-dependent term, once', 'experimental_accuracy_validated': False}
    return result


def directional_pair(initial_xyz, final_xyz, initial_charge, final_charge,
                     initial_mult, final_mult, *, directory, settings=None,
                     memory_mb=4000, threads=4):
    """Calculate one transfer direction. Swap the endpoints to get the reverse direction."""
    from core.properties.reorg import directional_relaxation
    initial_xyz, final_xyz = Path(initial_xyz).resolve(), Path(final_xyz).resolve()
    initial, final = read(initial_xyz), read(final_xyz)
    if initial.get_chemical_symbols() != final.get_chemical_symbols() or abs(initial_charge - final_charge) != 1:
        raise ValueError('matched atom order and one-electron endpoint change required')
    settings = {**ACTIVE_SP, 'solvent_grid_order': 41, **(settings or {})}
    if not settings.get('solvent'):
        raise ValueError('a solution model is required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    states = {}

    def evaluate(name, atoms, charge, mult, frozen_density=None):
        options = {**settings}
        if frozen_density is not None:
            options['solvent'] = None
        mf = make_mf(atoms, charge, mult, options, memory_mb=memory_mb, threads=threads)
        if frozen_density is not None:
            mf = attach_frozen_slow(mf, frozen_density, solvent=settings['solvent'],
                                    order=settings['solvent_grid_order'])
        mf.chkfile = str(directory / f'{name}.chk')
        mf.kernel()
        qc = electronic_checks(mf)
        states[name] = {'energy_eV': float(mf.e_tot) * HARTREE_EV, 'qc': qc}
        return mf

    reference = evaluate('initial_eq', initial, initial_charge, initial_mult)
    density = reference.make_rdm1()
    evaluate('final_at_initial_eq', initial, final_charge, final_mult)
    noneq = evaluate('final_nonequilibrium', initial, final_charge, final_mult, density)
    evaluate('final_eq', final, final_charge, final_mult)
    result = directional_relaxation(*(states[name]['energy_eV'] for name in
        ('initial_eq', 'final_at_initial_eq', 'final_nonequilibrium', 'final_eq')))
    return write_record(directory / 'result.json', {
        **result, 'status': 'complete', 'quantity': 'directional_electronic_solution_relaxation',
        'states': states, 'method': {**settings, 'engine': backend.get_dft_backend_name(),
                                      'accelerator': backend.resolve_dft_backend()},
        'nonequilibrium_context': noneq.frozen_slow_context,
        'sources': {str(p): file_hash(p) for p in (initial_xyz, final_xyz)},
        'endpoint_minima_independently_verified': False,
        'scope': 'finite endpoint calculation; not a rate or equilibrium ensemble'})
