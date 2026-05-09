"""Regression tests for the v0.3 API surface.

Pins down the public entry points:

- ``MIDAS.setup_mudata`` — registers data setup info on the MuData.
- ``MIDAS(mdata)`` — constructs the model from a registered MuData
  (instance state, no class-level mutation).
- ``model.get_latent_representation`` — stitches per-batch latents and
  reorders to ``mdata.obs_names``.
- ``model.save`` / ``MIDAS.load`` — round-trip without losing weights.
- ``MIDAS.configure_data_from_mdata`` — still works, but now warns.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _quickstart():
    import scmidas
    return scmidas.datasets.quickstart()


def _build_untrained_model(mdata):
    """Construct a MIDAS model on the quickstart data without training.

    Used by tests that only need a constructed model + optional 1-step
    train via Trainer to populate optimizer states.
    """
    import scmidas
    scmidas.MIDAS.setup_mudata(mdata, batch_key='batch')
    model = scmidas.MIDAS(
        mdata,
        save_model_path='/tmp/scmidas_test_save',
        batch_size=64,
    )
    return model


# ---------------------------------------------------------------------------
# setup_mudata
# ---------------------------------------------------------------------------
def test_setup_mudata_writes_uns():
    import scmidas
    mdata = _quickstart()
    assert '_scmidas' not in mdata.uns

    scmidas.MIDAS.setup_mudata(mdata, batch_key='batch')

    assert '_scmidas' in mdata.uns
    setup = mdata.uns['_scmidas']
    assert setup['batch_key'] == 'batch'
    assert setup['mods'] == ['rna', 'adt']
    assert setup['dims_x']['rna'] == [500]
    assert setup['dims_x']['adt'] == [224]
    assert sorted(setup['batch_names']) == ['p1_0', 'p2_0', 'p3_0', 'p4_0']


def test_setup_mudata_rejects_missing_batch_key():
    import scmidas
    mdata = _quickstart()
    with pytest.raises(ValueError, match="batch_key='nope'"):
        scmidas.MIDAS.setup_mudata(mdata, batch_key='nope')


# ---------------------------------------------------------------------------
# MIDAS(mdata) constructor
# ---------------------------------------------------------------------------
def test_midas_construct_from_mdata():
    import scmidas
    mdata = _quickstart()
    scmidas.MIDAS.setup_mudata(mdata, batch_key='batch')

    model = scmidas.MIDAS(mdata, batch_size=64)
    # Instance state, not class state
    assert model.dims_x == {'rna': [500], 'adt': [224]}
    assert model.batch_names == ['p1_0', 'p2_0', 'p3_0', 'p4_0']
    assert model.batch_size == 64
    assert hasattr(model, 'net')
    assert hasattr(model, 'dsc')


def test_midas_construct_without_setup_raises():
    import scmidas
    mdata = _quickstart()
    with pytest.raises(RuntimeError, match='setup_mudata'):
        scmidas.MIDAS(mdata)


def test_two_instances_have_independent_state():
    """Regression for the old class-level state mutation bug."""
    import scmidas
    mdata1 = _quickstart()
    mdata2 = _quickstart()

    scmidas.MIDAS.setup_mudata(mdata1, batch_key='batch')
    scmidas.MIDAS.setup_mudata(mdata2, batch_key='batch')

    m1 = scmidas.MIDAS(mdata1, batch_size=64)
    m2 = scmidas.MIDAS(mdata2, batch_size=128)

    # If state were class-level, batch_size on m1 would have been clobbered
    # by m2's construction.
    assert m1.batch_size == 64
    assert m2.batch_size == 128


# ---------------------------------------------------------------------------
# get_latent_representation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kind,expected_dim', [
    ('c', 32),     # default biological latent
    ('u', 2),
    ('joint', 34),
])
def test_get_latent_representation_shape(kind, expected_dim):
    import lightning as L
    L.seed_everything(0, verbose=False)

    mdata = _quickstart()
    model = _build_untrained_model(mdata)
    # 1 epoch is enough to populate the network with non-NaN weights
    model.train(max_epochs=1, accelerator='auto', devices=1)

    z = model.get_latent_representation(kind=kind)
    assert z.shape == (mdata.n_obs, expected_dim)
    assert not np.isnan(z).any(), 'every cell in mdata.obs_names should map'


def test_get_latent_representation_invalid_kind():
    mdata = _quickstart()
    model = _build_untrained_model(mdata)
    with pytest.raises(ValueError, match="kind must be 'c'"):
        model.get_latent_representation(kind='nope')


# ---------------------------------------------------------------------------
# get_imputed_values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('modality,n_features', [
    ('rna', 500),
    ('adt', 224),
])
def test_get_imputed_values_shape(modality, n_features):
    import lightning as L
    L.seed_everything(0, verbose=False)

    mdata = _quickstart()
    model = _build_untrained_model(mdata)
    model.train(max_epochs=1, accelerator='auto', devices=1)

    x = model.get_imputed_values(modality=modality)
    assert x.shape == (mdata.n_obs, n_features)
    assert not np.isnan(x).any()


def test_get_imputed_values_invalid_modality():
    mdata = _quickstart()
    model = _build_untrained_model(mdata)
    with pytest.raises(ValueError, match='not in registered modalities'):
        model.get_imputed_values(modality='atac')


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    import lightning as L
    import scmidas
    L.seed_everything(0, verbose=False)

    mdata = _quickstart()
    model = _build_untrained_model(mdata)
    model.train(max_epochs=1, accelerator='auto', devices=1)
    z_before = model.get_latent_representation()

    save_dir = tmp_path / 'mymodel'
    model.save(str(save_dir))
    assert (save_dir / 'model.pt').exists()
    assert (save_dir / 'setup.json').exists()

    # Fresh mdata + fresh model = should reproduce same latent
    mdata2 = _quickstart()
    loaded = scmidas.MIDAS.load(str(save_dir), mdata2, batch_size=64)
    z_after = loaded.get_latent_representation()

    np.testing.assert_allclose(z_before, z_after, rtol=1e-4, atol=1e-5)


def test_save_refuses_to_overwrite(tmp_path):
    mdata = _quickstart()
    model = _build_untrained_model(mdata)
    save_dir = tmp_path / 'mymodel'
    model.save(str(save_dir))
    # Second save into the same non-empty dir must error without overwrite
    with pytest.raises(FileExistsError):
        model.save(str(save_dir))
    # ... unless overwrite=True
    model.save(str(save_dir), overwrite=True)


# ---------------------------------------------------------------------------
# Legacy path keeps working (with DeprecationWarning)
# ---------------------------------------------------------------------------
def test_configure_data_from_mdata_emits_deprecation_warning():
    import scmidas
    from scmidas.config import load_config
    mdata = _quickstart()
    configs = load_config()
    dims_x = {m: [mdata[m].n_vars] for m in mdata.mod}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scmidas.MIDAS.configure_data_from_mdata(
            configs=configs,
            mdata=mdata,
            dims_x=dims_x,
            batch_key='batch',
            batch_size=64,
        )

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any('configure_data_from_mdata' in str(w.message) for w in deprecations)
