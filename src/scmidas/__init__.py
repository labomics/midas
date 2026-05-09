from .model import *
from .data import *
from .nn import *
from .utils import *
from .config import *
from . import datasets
from . import plot
from . import plot as pl  # short alias: scmidas.pl.umap(...)
from .api import integrate

from importlib.metadata import PackageNotFoundError, version as _pkg_version
try:
    __version__ = _pkg_version('scmidas')
except PackageNotFoundError:
    __version__ = '0.0.0+unknown'
del PackageNotFoundError, _pkg_version


def _check_gpu_compat():
    # torch 2.11 dropped Volta (V100, CC 7.0) and Pascal (P100/GTX 10xx,
    # CC 6.x) from its default cu128/cu129 wheels. On those GPUs the first
    # CUDA op raises "no kernel image is available" — translate that here.
    try:
        import warnings
        import torch
        if not torch.cuda.is_available():
            return
        try:
            (torch.zeros(1, device='cuda') + 1).cpu()
        except Exception as e:
            if 'no kernel image is available' not in str(e):
                return
            major, minor = torch.cuda.get_device_capability()
            warnings.warn(
                f"torch {torch.__version__} has no CUDA kernels for your GPU "
                f"({torch.cuda.get_device_name(0)}, compute capability "
                f"{major}.{minor}). torch 2.11+ dropped Volta/Pascal from its "
                f"default cu128/cu129 wheels. Fix: "
                f"pip install 'torch<2.11' 'torchvision<0.26' 'torchaudio<2.11' "
                f"or use the cu126 wheel via "
                f"--index-url https://download.pytorch.org/whl/cu126",
                UserWarning, stacklevel=2,
            )
    except Exception:
        pass


_check_gpu_compat()
del _check_gpu_compat
