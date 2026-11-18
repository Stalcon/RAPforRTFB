"""Check imports and device execution without downloading model weights."""
import importlib
import json

from core.backend import describe, get_dft_module, get_uma_device


def main():
    failures = []
    for name in ("numpy", "scipy", "rdkit", "ase", "pyscf", "geometric", "pyscf.dispersion", "fairchem.core"):
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    try:
        import torch
        device = get_uma_device()
        x = torch.eye(4, device=device)
        if not torch.allclose(x @ x, x):
            raise RuntimeError("device arithmetic failed")
        get_dft_module()
    except Exception as exc:
        failures.append(str(exc))
    try:
        from pyscf import gto, dft, lib
        lib.num_threads(1)
        mol = gto.M(atom='O 0 0 0; H 0 .757 .586; H 0 -.757 .586', basis='sto-3g', verbose=0)
        mf = dft.RKS(mol).density_fit().SMD()
        mf.with_solvent.solvent = 'acetonitrile'
        mf.xc = 'pbe'
        mf.grids.level = 0
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('SMD smoke SCF did not converge')
        print('OK CPU SMD energy execution')
    except Exception as exc:
        failures.append(f'SMD execution: {exc}')
    print(json.dumps(describe(), indent=2))
    for failure in failures:
        print(f"FAIL {failure}")
    print('Run python tests/gpu_test.py to check model inference.')
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
