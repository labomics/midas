"""High-level conveniences on top of the MIDAS class."""
from __future__ import annotations

import logging
from typing import Any, Optional

import mudata as mu

from .config import load_config
from .model import MIDAS

logger = logging.getLogger(__name__)


# Tuned for the bundled quickstart dataset (PBMC mosaic, 1200 cells, 500
# HVGs + 224 ADT). On a single mid-range GPU these defaults converge in
# roughly one minute and produce a clean lineage-separated UMAP. They
# are NOT general-purpose — for full datasets, fall back to the
# paper defaults via ``MIDAS.configure_data_from_mdata(...).train()``.
_QUICKSTART_DEFAULTS: dict[str, Any] = {
    'batch_size': 128,
    'max_epochs': 65,
    'lr_net': 3e-4,
    'lr_dsc': 3e-4,
}


def integrate(
    mdata: mu.MuData,
    *,
    batch_key: str = 'batch',
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    accelerator: str = 'auto',
    devices: Any = 1,
    strategy: str = 'auto',
    save_model_path: str = './saved_models/scmidas',
    seed: Optional[int] = 42,
    key_added: str = 'X_midas',
    **kwargs: Any,
) -> MIDAS:
    """One-call MIDAS pipeline for users who want a sensible default.

    Equivalent to::

        scmidas.MIDAS.setup_mudata(mdata, batch_key=batch_key)
        model = scmidas.MIDAS(mdata, configs=..., batch_size=..., ...)
        model.train(max_epochs=..., accelerator=..., ...)
        mdata.obsm[key_added] = model.get_latent_representation()

    .. warning::
        The default training hyperparameters (``batch_size=128``,
        ``max_epochs=65``, ``lr=3e-4``) are tuned for the **toy
        quickstart dataset** (1600 cells). They are **not appropriate
        for real analyses** — for full datasets pass your own
        ``max_epochs`` (typically 1000-2000) and consider letting
        ``batch_size`` default back to 256.

    Parameters:
        mdata : MuData
            Multi-modal single-cell data.
        batch_key : str
            Column in each modality's ``.obs`` that identifies the
            source batch.
        max_epochs : int, optional
            Training epochs. Default 65 (quickstart-tuned). For real
            data, override with 1000-2000.
        batch_size : int, optional
            Mini-batch size. Default 128 (quickstart-tuned). For real
            data, 256 is a more typical choice.
        accelerator, devices, strategy
            Forwarded to ``lightning.Trainer``. Default ``'auto'``
            picks GPU if available.
        save_model_path : str
            Where to write checkpoints during training.
        seed : int, optional
            If not None, calls ``lightning.seed_everything(seed)``
            before setup, so the run is reproducible.
        key_added : str
            Key under which the biological latent ``z_c`` is written
            to ``mdata.obsm``. Defaults to ``'X_midas'`` so that
            ``sc.pp.neighbors(mdata, use_rep='X_midas')`` works
            without further arguments.
        **kwargs
            Additional keyword arguments forwarded to ``MIDAS(...)``.

    Returns:
        MIDAS:
            A trained MIDAS model. The biological latent has already
            been written to ``mdata.obsm[key_added]``.
    """
    if seed is not None:
        import lightning as L
        L.seed_everything(seed, verbose=False)

    configs = load_config()
    configs['lr_net'] = _QUICKSTART_DEFAULTS['lr_net']
    configs['lr_dsc'] = _QUICKSTART_DEFAULTS['lr_dsc']

    bsz = batch_size if batch_size is not None else _QUICKSTART_DEFAULTS['batch_size']
    eps = max_epochs if max_epochs is not None else _QUICKSTART_DEFAULTS['max_epochs']

    logger.info(
        'scmidas.integrate(): toy-tuned defaults — '
        'batch_size=%d, max_epochs=%d, lr=%g. '
        'For real datasets, override max_epochs (e.g. 2000) '
        'and consider batch_size=256.',
        bsz, eps, _QUICKSTART_DEFAULTS['lr_net'],
    )

    MIDAS.setup_mudata(mdata, batch_key=batch_key)
    model = MIDAS(
        mdata,
        configs=configs,
        batch_size=bsz,
        save_model_path=save_model_path,
        **kwargs,
    )
    model.train(
        max_epochs=eps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
    )
    mdata.obsm[key_added] = model.get_latent_representation()
    return model
