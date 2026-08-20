"""Tests for IMAGE_GEN_PLATFORM / IMAGE_EDIT_PLATFORM parsing.

These values are parsed in the OpenaiBase class body, i.e. at import time, so a
bad value takes the whole service down at startup rather than failing a request.
"""

import pytest

from openai_forward.base import _parse_platforms
from openai_forward.routers.image_gen_platform import ImageEditPlatform, ImageGenPlatform


def gen(value):
    return [p.name for p in _parse_platforms(value, ImageGenPlatform, "IMAGE_GEN_PLATFORM", "dalle3")]


def edit(value):
    return [p.name for p in _parse_platforms(value, ImageEditPlatform, "IMAGE_EDIT_PLATFORM", "openai")]


def test_single_platform():
    assert gen("openai") == ["openai"]
    assert edit("openai") == ["openai"]


def test_order_is_preserved():
    """The first entry is the default platform, so order is significant."""
    assert gen("flux1_kontext,openai") == ["flux1_kontext", "openai"]
    assert gen("openai,flux1_kontext") == ["openai", "flux1_kontext"]


def test_whitespace_is_tolerated():
    assert gen("  openai , flux1_1  ") == ["openai", "flux1_1"]


def test_dalle3_is_an_alias_for_openai():
    assert gen("dalle3") == ["openai"]


def test_trailing_comma_does_not_crash():
    """A stray comma used to raise KeyError('') and stop the service booting."""
    assert gen("openai,") == ["openai"]
    assert gen("openai,,flux1_1") == ["openai", "flux1_1"]
    assert edit("openai,") == ["openai"]


def test_empty_value_falls_back_to_the_default():
    """An unset-but-present var used to raise KeyError('') at import."""
    assert gen("") == ["openai"]  # "dalle3" default, which is an openai alias
    assert gen("   ") == ["openai"]
    assert edit("") == ["openai"]


def test_unknown_platform_names_the_variable_and_the_valid_values():
    with pytest.raises(ValueError) as exc_info:
        gen("flux1_1x")
    message = str(exc_info.value)
    assert "IMAGE_GEN_PLATFORM" in message
    assert "flux1_1x" in message
    assert "flux1_kontext" in message  # lists what is valid


def test_unknown_platform_is_rejected_rather_than_ignored():
    """A typo must not silently route traffic to an unintended platform."""
    with pytest.raises(ValueError):
        gen("openai,fluxx")
    with pytest.raises(ValueError):
        edit("flux1_1")  # valid for generation, not for edits
