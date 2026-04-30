"""Smoke tests: package imports cleanly and trivial APIs return expected types."""


def test_version_is_set():
    import scmidas

    assert isinstance(scmidas.__version__, str)
    assert scmidas.__version__


def test_top_level_midas_importable():
    from scmidas import MIDAS  # noqa: F401


def test_load_config_default_returns_dict():
    from scmidas.config import load_config

    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "dim_c" in cfg
    assert "dim_u" in cfg


def test_load_config_unknown_raises():
    from scmidas.config import load_config

    import pytest

    with pytest.raises(KeyError):
        load_config("definitely-not-a-real-config-name")
