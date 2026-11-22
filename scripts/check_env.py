"""Check UMA device arithmetic and DFT execution without downloading weights."""

import argparse
import importlib
import json

from core.backend import describe, get_uma_device, is_gpu_mean_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dft-only", action="store_true", help="Skip PyTorch and UMA checks.")
    args = parser.parse_args()
    failures = []
    packages = ["numpy", "scipy", "rdkit", "ase", "pyscf", "geometric", "pyscf.dispersion"]
    for name in packages:
        try:
            importlib.import_module(name)
            print(f"OK {name}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    # Initialize the DFT runtime before loading any libraries that import PyTorch.
    summary = describe(include_uma=False)
    try:
        from ase import Atoms
        from core.modeling.sp import electronic_checks, make_mf

        water = Atoms("OH2", positions=[[0, 0, 0], [0, .757, .586], [0, -.757, .586]])
        settings = {"basis": "sto-3g", "xc": "pbe", "grid_level": 0,
                    "solvent": "acetonitrile", "dispersion": "d4"}
        mf = make_mf(water, 0, 1, settings, threads=1)
        expected_gpu = summary["dft_backend"] != "cpu"
        if is_gpu_mean_field(mf) != expected_gpu:
            raise RuntimeError("the mean-field engine does not match the selected DFT backend")
        mf.kernel()
        electronic_checks(mf)
        summary["dft_smd_execution"] = "passed"
        print(f"OK {summary['dft_backend']} SMD/D4 energy execution")
    except Exception as exc:
        summary["dft_smd_execution"] = "failed"
        failures.append(f"SMD/D4 execution: {exc}")

    if not args.dft_only:
        try:
            import torch
            importlib.import_module("fairchem.core")
            device = get_uma_device()
            x = torch.eye(4, device=device)
            if not torch.allclose(x @ x, x):
                raise RuntimeError("device arithmetic failed")
            summary["uma_device"] = device
            summary["torch_version"] = torch.__version__
            summary["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:
            failures.append(f"UMA environment: {exc}")

    print(json.dumps(summary, indent=2))
    for failure in failures:
        print(f"FAIL {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
