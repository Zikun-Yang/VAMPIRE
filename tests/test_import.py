"""Smoke tests to verify the package can be imported."""

import importlib


def test_import_vampire():
    """Test that the top-level vampire package imports without error."""
    mod = importlib.import_module("vampire")
    assert mod is not None
