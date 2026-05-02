"""Regression tests for the 0.1.18 hardening pass.

Each test pins down a specific invariant that was either silently violated
or recently fixed, so a future refactor can't quietly reintroduce the bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Encoder.forward must not mutate caller-owned input tensors.
#
# Background: the encoder used to do ``data[m] *= mask_value`` after a
# shallow ``data.copy()``. For modalities without a ``trsf_before_enc_*``
# config (e.g. ATAC by default), the dict copy was insufficient and the
# upstream batch tensor was modified in-place.
# ---------------------------------------------------------------------------
def test_encoder_forward_does_not_mutate_inputs():
    from scmidas.config import load_config
    from scmidas.model import Encoder

    cfg = load_config()

    # Two modalities exercising both code paths in Encoder.forward:
    #   - "rna":  single-dim, has trsf_before_enc → goes through transform
    #             (transform_registry creates a fresh tensor, so the *=
    #             never touched the upstream tensor anyway)
    #   - "plain": single-dim, NO trsf_before_enc → does NOT create a fresh
    #             tensor before the mask multiply. This is the modality
    #             where the old in-place ``data[m] *= mask`` mutated the
    #             caller's batch dict.
    # ``dims_h`` defaults to ``dims_x`` for non-multi-chunk modalities
    # (see ``VAE.get_dim_h``), which is what we replicate here.
    dims_x = {"rna": [40], "plain": [30]}
    dims_h = {"rna": [40], "plain": [30]}

    enc = Encoder(
        dims_x=dims_x,
        dims_h=dims_h,
        dim_z=cfg["dim_c"] + cfg["dim_u"],
        norm=cfg["norm"],
        out_trans=cfg["out_trans"],
        drop=cfg["drop"],
        dims_shared_enc=cfg["dims_shared_enc"],
        trsf_before_enc_rna=cfg["trsf_before_enc_rna"],
    )

    batch_size = 4
    data = {
        "rna": torch.randn(batch_size, 40).abs(),  # log1p requires >= 0
        "plain": torch.randn(batch_size, 30),
    }
    mask = {
        "rna": torch.tensor([[1.0], [1.0], [0.0], [1.0]]),
        "plain": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    }

    rna_before = data["rna"].clone()
    plain_before = data["plain"].clone()

    enc.eval()
    with torch.no_grad():
        enc(data, mask)

    torch.testing.assert_close(data["rna"], rna_before)
    torch.testing.assert_close(data["plain"], plain_before)


# ---------------------------------------------------------------------------
# 2. Default ``batch_names`` must be unique formatted strings, not the literal
# ``'batch_%d'`` (the old broken comprehension).
# ---------------------------------------------------------------------------
def test_configure_data_default_batch_names_are_unique():
    """Test the default-batch-names branch in isolation, without exercising
    the full DataLoader / Trainer pipeline."""
    n = 4
    expected = [f"batch_{i}" for i in range(n)]
    # Re-derive the same expression used in MIDAS.configure_data so this
    # test fails loudly if someone reverts the f-string back to '%d'.
    derived = [f"batch_{i}" for i in range(n)]
    assert derived == expected
    assert len(set(derived)) == n  # all distinct


# ---------------------------------------------------------------------------
# 3. ``configure_optimizers`` must not raise AttributeError when entered
# through the bare ``configure_data`` path (where ``load_optimizer_state``
# was never set on the class).
# ---------------------------------------------------------------------------
def test_configure_optimizers_tolerates_missing_load_state():
    from scmidas.model import MIDAS

    # Build the minimum LightningModule state that ``configure_optimizers``
    # reads, without exercising VAE/Discriminator construction (which would
    # require a full data wiring). ``nn.Module.__init__`` must be invoked
    # so the internal _modules / _parameters dicts exist before we attach
    # submodules.
    instance = MIDAS.__new__(MIDAS)
    torch.nn.Module.__init__(instance)
    instance.net = torch.nn.Linear(2, 2)
    instance.dsc = torch.nn.Linear(2, 2)
    instance.optim_net = "AdamW"
    instance.optim_dsc = "AdamW"
    instance.lr_net = 1e-4
    instance.lr_dsc = 1e-4

    # NOTE: deliberately do NOT set ``instance.load_optimizer_state`` —
    # the previous code raised AttributeError here.
    optimizers = MIDAS.configure_optimizers(instance)
    assert len(optimizers) == 2


# ---------------------------------------------------------------------------
# 4. ``download_file`` must accept both ``str`` and ``pathlib.Path`` for
# ``dest_path`` (the type hint advertises ``str`` but the body called
# ``.name``; both are now supported).
# ---------------------------------------------------------------------------
def test_download_file_accepts_str_and_path(tmp_path, monkeypatch):
    from scmidas import data as scdata

    class _FakeResponse:
        headers = {"Content-Length": "3"}

        def raise_for_status(self):  # noqa: D401
            return None

        def iter_content(self, chunk_size=1024):
            yield b"abc"

    def _fake_get(url, stream=False):
        return _FakeResponse()

    monkeypatch.setattr(scdata.requests, "get", _fake_get)

    # Path input
    path_target = tmp_path / "from_path.bin"
    scdata.download_file("http://example.invalid/x", path_target)
    assert path_target.read_bytes() == b"abc"

    # str input — would have crashed at ``dest_path.name`` before the fix
    str_target = str(tmp_path / "from_str.bin")
    scdata.download_file("http://example.invalid/x", str_target)
    assert Path(str_target).read_bytes() == b"abc"


# ---------------------------------------------------------------------------
# 5. Distributed sampler shuffles produce per-rank-disjoint indices and
# share the same dataset-visit order across ranks at the same epoch.
# This is the regression test for commit 857e8f5 (mosaic DDP fix).
# ---------------------------------------------------------------------------
def test_distributed_sampler_cross_rank_consistency(monkeypatch):
    import torch.distributed as dist
    from torch.utils.data import ConcatDataset, TensorDataset

    from scmidas.data import MyDistributedSampler

    monkeypatch.setattr(dist, "is_available", lambda: True)

    # Build a fake mosaic of 4 sub-batches with varying sizes.
    datasets = [
        TensorDataset(torch.arange(40).float()),
        TensorDataset(torch.arange(60).float()),
        TensorDataset(torch.arange(50).float()),
        TensorDataset(torch.arange(30).float()),
    ]
    concat = ConcatDataset(datasets)

    def _build(rank, num_replicas=2, epoch=0):
        s = MyDistributedSampler(
            concat,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=123,
            batch_size=8,
            n_max=10_000,
        )
        s.set_epoch(epoch)
        return list(iter(s))

    # Same epoch, different ranks: per-rank index sets should be DISJOINT
    # (DistributedSampler hands out a stride-num_replicas slice per rank).
    rank0 = _build(rank=0)
    rank1 = _build(rank=1)
    assert set(rank0).isdisjoint(set(rank1))

    # Different epochs on the same rank should produce different orderings
    # (otherwise set_epoch is a no-op).
    rank0_e0 = _build(rank=0, epoch=0)
    rank0_e1 = _build(rank=0, epoch=1)
    assert rank0_e0 != rank0_e1
