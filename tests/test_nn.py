"""Unit tests for the distribution registry in scmidas.nn."""

import pytest
import torch

from scmidas.nn import DistributionRegistry, distribution_registry


def test_default_distributions_registered():
    names = distribution_registry.list_registered()
    assert "POISSON" in names
    assert "BERNOULLI" in names
    assert "CE" in names


def test_get_loss_returns_callable():
    assert callable(distribution_registry.get_loss("POISSON"))


def test_get_loss_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not registered"):
        distribution_registry.get_loss("NONEXISTENT")


def test_get_sampling_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not registered"):
        distribution_registry.get_sampling("NONEXISTENT")


def test_get_activate_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not registered"):
        distribution_registry.get_activate("NONEXISTENT")


def test_register_overrides_existing_entry():
    reg = DistributionRegistry()
    initial_loss = reg.get_loss("POISSON")
    new_loss = torch.nn.MSELoss()
    reg.register("POISSON", new_loss, reg.null, reg.null)
    assert reg.get_loss("POISSON") is new_loss
    assert reg.get_loss("POISSON") is not initial_loss


def test_bernoulli_sampling_returns_int_tensor_with_correct_shape():
    out = DistributionRegistry.bernoulli_sampling(torch.full((4,), 0.5))
    assert out.dtype == torch.int32
    assert out.shape == (4,)
    assert ((out == 0) | (out == 1)).all()


def test_poisson_sampling_returns_nonnegative_int_tensor():
    torch.manual_seed(42)
    out = DistributionRegistry.poisson_sampling(torch.full((4,), 1.0))
    assert out.dtype == torch.int32
    assert (out >= 0).all()


def test_null_returns_input_unchanged():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert DistributionRegistry.null(x) is x
