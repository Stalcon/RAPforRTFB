"""ROCm/CUDA UMA execution and separately selected DFT execution."""
import importlib
import os


def resolve_backend(requested=None):
    requested = (requested or os.environ.get("BACKEND", "auto")).lower()
    if requested not in ("auto", "cpu", "cuda", "rocm"):
        raise ValueError("BACKEND must be auto, cpu, cuda, or rocm")
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return "rocm" if torch.version.hip else "cuda"


def configure_threads(uma_threads=2, dft_threads=4):
    if min(uma_threads, dft_threads) < 1:
        raise ValueError("thread counts must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(dft_threads)
    return {"uma_threads": uma_threads, "dft_threads": dft_threads}


def get_uma_device(backend=None):
    backend = resolve_backend(backend)
    if backend == "cpu":
        return "cpu"
    import torch
    actual = "rocm" if torch.version.hip else "cuda"
    if not torch.cuda.is_available() or actual != backend:
        raise RuntimeError(f"Requested {backend}, but no matching PyTorch GPU is available")
    # PyTorch uses its cuda device API for HIP builds too.
    return "cuda:0"


def get_dft_backend_name(backend=None):
    requested = os.environ.get("DFT_BACKEND", "auto").lower()
    if requested == "rocm":
        raise RuntimeError("AMD DFT is not implemented; set DFT_BACKEND=cpu")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("DFT_BACKEND must be auto, cpu, or cuda")
    if requested == "auto":
        requested = "cuda" if resolve_backend(backend) == "cuda" else "cpu"
    return "gpu4pyscf" if requested == "cuda" else "pyscf"


def get_dft_module(backend=None):
    return importlib.import_module(get_dft_backend_name(backend) + ".dft")


def get_sp_backend_name():
    requested = os.environ.get("SP_BACKEND", "auto").lower()
    if requested == "rocm":
        return "gpu4pyscf-rocm"
    if requested != "auto":
        raise ValueError("SP_BACKEND must be auto or rocm")
    return get_dft_backend_name()


def get_rocm_dft_module():
    try:
        import cupy
        if not cupy.cuda.runtime.is_hip or not cupy.cuda.runtime.getDeviceCount():
            raise RuntimeError("ROCm CuPy did not find an AMD GPU")
        return importlib.import_module("gpu4pyscf.dft")
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "SP_BACKEND=rocm requires a HIP build of GPU4PySCF and ROCm CuPy "
            f"with access to an AMD GPU: {exc}"
        ) from exc


def get_geomopt_fn():
    from pyscf.geomopt.geometric_solver import optimize
    return optimize


def describe():
    result = {"requested_backend": resolve_backend(), "dft_engine": get_dft_backend_name(),
              "amd_dft_acceleration": False}
    try:
        import torch
        result.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      cuda_version=torch.version.cuda,
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        result["uma_device"] = get_uma_device()
    except (ImportError, RuntimeError) as exc:
        result["uma_error"] = str(exc)
    return result
