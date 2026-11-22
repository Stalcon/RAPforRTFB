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


def _cupy_backend():
    import cupy
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy cannot see a GPU")
    return "rocm" if cupy.cuda.runtime.is_hip else "cuda"


def resolve_dft_backend(backend=None):
    requested = os.environ.get("DFT_BACKEND", "auto").lower()
    if requested not in ("auto", "cpu", "cuda", "rocm"):
        raise ValueError("DFT_BACKEND must be auto, cpu, cuda, or rocm")
    if requested != "auto":
        return requested
    preference = (backend or os.environ.get("BACKEND", "auto")).lower()
    if preference != "auto":
        return resolve_backend(preference)
    # Probe the DFT runtime before importing PyTorch so ROCm uses CuPy's HIP compiler.
    try:
        return _cupy_backend()
    except (ImportError, OSError, RuntimeError):
        return resolve_backend()


def get_dft_backend_name(backend=None):
    return "pyscf" if resolve_dft_backend(backend) == "cpu" else "gpu4pyscf"


def get_dft_module(backend=None):
    selected = resolve_dft_backend(backend)
    if selected == "cpu":
        return importlib.import_module("pyscf.dft")
    try:
        device = _cupy_backend()
        if device != selected:
            raise RuntimeError(f"CuPy is using {device}, expected {selected}")
        package = importlib.import_module("gpu4pyscf")
        compiled = getattr(package, "BACKEND", "cuda")
        expected = "hip" if selected == "rocm" else "cuda"
        if compiled != expected:
            raise RuntimeError(f"GPU4PySCF is built for {compiled}, expected {expected}")
        return package.dft
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"DFT selected {selected} but its GPU4PySCF runtime is unavailable: {exc}. "
            "Install the matching GPU build or explicitly set DFT_BACKEND=cpu."
        ) from exc


def is_gpu_mean_field(mf):
    # Density fitting wraps SCF objects in dynamically generated classes.
    return any(cls.__module__.startswith("gpu4pyscf.") for cls in type(mf).__mro__)


def host_array(value):
    """Transfer computed derivatives for file output and host-side analysis."""
    return value.get() if hasattr(value, "get") else value


def get_geomopt_fn():
    from pyscf.geomopt.geometric_solver import optimize
    return optimize


def describe(*, include_uma=True):
    dft_backend = resolve_dft_backend()
    result = {"dft_backend": dft_backend, "dft_engine": get_dft_backend_name()}
    if dft_backend != "cpu":
        try:
            get_dft_module()
        except RuntimeError as exc:
            result["dft_error"] = str(exc)
    result["amd_dft_acceleration"] = dft_backend == "rocm" and "dft_error" not in result
    if not include_uma:
        return result
    result["requested_backend"] = resolve_backend()
    try:
        import torch
        result.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      cuda_version=torch.version.cuda,
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        result["uma_device"] = get_uma_device()
    except (ImportError, RuntimeError) as exc:
        result["uma_error"] = str(exc)
    return result
