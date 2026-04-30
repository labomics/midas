"""Unit tests for pure helper functions in scmidas.utils."""

import numpy as np
import torch

from scmidas.utils import (
    detach_tensors,
    exp,
    extract_params,
    extract_values,
    filter_keys,
    generate_all_combinations,
    get_name_fmt,
    log,
    ref_sort,
    reverse_dict,
    to_numpy,
)


# --- exp / log: numerically stable transforms --------------------------------

def test_exp_handles_negative_and_positive():
    y = exp(torch.tensor([-10.0, 0.0, 5.0]))
    assert torch.isfinite(y).all()
    assert (y >= 0).all()


def test_log_handles_zero():
    y = log(torch.tensor([0.0, 1.0]))
    assert torch.isfinite(y).all()


def test_log_exp_roundtrip_for_positive_inputs():
    x = torch.tensor([0.1, 1.0, 10.0])
    torch.testing.assert_close(exp(log(x)), x, rtol=1e-3, atol=1e-3)


# --- extract_params: prefix-stripping dict filter ----------------------------

def test_extract_params_strips_prefix():
    cfg = {"opt_lr": 0.001, "opt_momentum": 0.9, "drop": 0.2}
    assert extract_params(cfg, "opt_") == {"lr": 0.001, "momentum": 0.9}


def test_extract_params_no_match_returns_empty():
    assert extract_params({"a": 1, "b": 2}, "xyz_") == {}


# --- ref_sort: order-by-reference --------------------------------------------

def test_ref_sort_keeps_reference_order():
    assert ref_sort(["c", "a"], ["a", "b", "c"]) == ["a", "c"]


def test_ref_sort_drops_unreferenced_elements():
    assert ref_sort(["a", "z"], ["a", "b"]) == ["a"]


def test_ref_sort_handles_empty_inputs():
    assert ref_sort([], ["a"]) == []
    assert ref_sort(["a"], []) == []


# --- extract_values: recursive flatten ---------------------------------------

def test_extract_values_flattens_nested_structures():
    out = extract_values({"a": [1, 2], "b": {"c": 3}})
    assert sorted(out) == [1, 2, 3]


def test_extract_values_scalar_input_wraps_in_list():
    assert extract_values(42) == [42]


# --- reverse_dict: nested key swap -------------------------------------------

def test_reverse_dict_swaps_outer_and_inner_keys():
    src = {"x": {"a": 1}, "y": {"a": 2, "b": 3}}
    assert reverse_dict(src) == {"a": {"x": 1, "y": 2}, "b": {"y": 3}}


# --- filter_keys: substring-match dict filter --------------------------------

def test_filter_keys_keeps_substring_matches():
    d = {"foo_x": 1, "bar_x": 2, "baz_y": 3}
    assert filter_keys(d, "_x") == {"foo_x": 1, "bar_x": 2}


def test_filter_keys_no_match_returns_empty():
    assert filter_keys({"a": 1}, "z") == {}


# --- get_name_fmt: zero-padded format string ---------------------------------

def test_get_name_fmt_picks_correct_width():
    assert get_name_fmt(9) == "%01d"
    assert get_name_fmt(10) == "%02d"
    assert get_name_fmt(999) == "%03d"
    assert get_name_fmt(1000) == "%04d"


# --- generate_all_combinations: input/output mod combinations ----------------

def test_generate_all_combinations_two_modalities():
    out = generate_all_combinations(["rna", "adt"])
    inputs = [tuple(p[0]) for p in out]
    assert ("rna",) in inputs
    assert ("adt",) in inputs
    assert len(out) == 2


def test_generate_all_combinations_three_modalities_count():
    # r=1: C(3,1)=3, r=2: C(3,2)=3, total = 6
    out = generate_all_combinations(["rna", "adt", "atac"])
    assert len(out) == 6


# --- to_numpy / detach_tensors: tensor conversion ----------------------------

def test_to_numpy_passthrough_for_ndarray():
    a = np.array([1.0, 2.0])
    assert to_numpy(a) is a


def test_to_numpy_converts_tensor():
    out = to_numpy(torch.tensor([1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_detach_tensors_recurses_into_nested_dicts():
    t = torch.tensor([1.0], requires_grad=True)
    out = detach_tensors({"a": t, "b": {"c": t}})
    assert not out["a"].requires_grad
    assert not out["b"]["c"].requires_grad
