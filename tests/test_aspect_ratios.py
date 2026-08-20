"""The app offers seven aspect ratios; the proxy has to turn each into a size OpenAI accepts.

Before gpt-image-2 only three sizes existed, so every ratio was snapped by orientation and
3:4 came back the same shape as 9:21. gpt-image-2 accepts any size meeting its constraints,
so the ratio is honoured exactly -- but only for models that actually allow it.
"""

from math import gcd

import pytest

from openai_forward.base import (
    _aspect_ratio_to_openai_dimensions as convert,
    _model_takes_any_size,
)

# exactly the cases in AspectRatio (chatster: ChatStreamBackOffViewModel.swift)
APP_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]

FLEXIBLE = "gpt-image-2"
LEGACY = "gpt-image-1.5"

LEGACY_BUDGET = {
    "landscape": 1536 * 1024,
    "portrait": 1024 * 1536,
    "square": 1024 * 1024,
}


def orientation(ratio):
    w, h = map(int, ratio.split(":"))
    return "landscape" if w > h else "portrait" if h > w else "square"


@pytest.mark.parametrize("ratio", APP_RATIOS)
def test_ratio_is_exact_for_gpt_image_2(ratio):
    w_ratio, h_ratio = map(int, ratio.split(":"))
    w, h = convert(ratio, FLEXIBLE)
    divisor = gcd(w, h)
    assert (w // divisor, h // divisor) == (
        w_ratio // gcd(w_ratio, h_ratio),
        h_ratio // gcd(w_ratio, h_ratio),
    ), f"{ratio} -> {w}x{h} is not that ratio"


@pytest.mark.parametrize("ratio", APP_RATIOS)
def test_result_satisfies_every_documented_constraint(ratio):
    w, h = convert(ratio, FLEXIBLE)
    assert w % 16 == 0 and h % 16 == 0, "edges must be multiples of 16"
    assert max(w, h) <= 3840, "longest edge must be <= 3840"
    assert max(w, h) / min(w, h) <= 3, "long:short must be <= 3:1"
    assert 655_360 <= w * h <= 8_294_400, "total pixels out of range"


@pytest.mark.parametrize("ratio", APP_RATIOS)
def test_never_costs_more_than_the_size_it_replaces(ratio):
    w, h = convert(ratio, FLEXIBLE)
    assert w * h <= LEGACY_BUDGET[orientation(ratio)]


def test_every_app_ratio_now_maps_to_a_distinct_size():
    """The reported bug: 3:4 and 9:21 produced identical images."""
    sizes = {r: convert(r, FLEXIBLE) for r in APP_RATIOS}
    assert len(set(sizes.values())) == len(APP_RATIOS), sizes
    assert sizes["3:4"] != sizes["9:21"]
    assert sizes["21:9"] != sizes["4:3"]


@pytest.mark.parametrize("ratio", APP_RATIOS)
def test_legacy_models_still_get_the_three_legacy_sizes(ratio):
    assert convert(ratio, LEGACY) in {(1024, 1024), (1536, 1024), (1024, 1536)}


def test_unknown_model_is_treated_as_legacy():
    """Conservative by design: a legacy size is valid everywhere, an arbitrary one is not."""
    assert convert("16:9", "some-future-model") == (1536, 1024)
    assert convert("16:9", None) == (1536, 1024)


def test_flexible_model_matching_is_by_prefix_and_case_insensitive():
    assert _model_takes_any_size("gpt-image-2")
    assert _model_takes_any_size("GPT-Image-2-Mini")
    assert not _model_takes_any_size("gpt-image-1.5")
    assert not _model_takes_any_size(None)


def test_dated_snapshots_match_too():
    """OpenAI publishes dated snapshots alongside the alias -- gpt-image-2-2026-04-21 is in
    the documented model enum. Prefix matching is what keeps those working without a code
    change; an exact-match list would silently snap them to the legacy sizes."""
    assert _model_takes_any_size("gpt-image-2-2026-04-21")
    assert convert("16:9", "gpt-image-2-2026-04-21") == convert("16:9", "gpt-image-2")


def test_aspect_limit_boundary_is_inclusive():
    """The docs say the ratio must be between 1:3 and 3:1, so exactly 3:1 is allowed."""
    w, h = convert("3:1", FLEXIBLE)
    assert max(w, h) / min(w, h) == 3
    assert convert("4:1", FLEXIBLE) == (
        1536,
        1024,
    )  # beyond the limit -> legacy fallback


@pytest.mark.parametrize(
    "bad", ["", "auto", "1024x1024", "not:a:ratio", "0:5", "x:y", None]
)
def test_unparseable_ratios_fall_back_to_a_square(bad):
    assert convert(bad, FLEXIBLE) == (1024, 1024)


def test_ratio_too_elongated_for_the_model_falls_back_to_orientation():
    """4:1 exceeds the documented 3:1 limit, so it cannot be honoured exactly."""
    assert convert("4:1", FLEXIBLE) == (1536, 1024)
    assert convert("1:4", FLEXIBLE) == (1024, 1536)


# --- end to end: the pin has to resolve before the size decision -------------------------

import asyncio
import hashlib
import hmac
import json

import httpx
from starlette.requests import Request

import openai_forward.base as base
from openai_forward.openai import OpenaiBase

SECRET = "test-secret"


def forward_generation(payload, pin=""):
    """Run the real request path and return the JSON body that reaches OpenAI."""
    body = json.dumps(payload).encode()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/generations",
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", b"Bearer test"),
            (b"x-request-signature", signature.encode()),
        ],
    }
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def run():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = await request.aread()
            return httpx.Response(200, json={"data": []})

        client = httpx.AsyncClient(
            base_url=OpenaiBase.BASE_URL, transport=httpx.MockTransport(handler)
        )
        request = Request(scope, receive)
        original_secret, original_pin = OpenaiBase.APP_SECRET, base.OPENAI_IMAGE_MODEL
        OpenaiBase.APP_SECRET, base.OPENAI_IMAGE_MODEL = SECRET, pin
        try:
            assert await OpenaiBase.validate_request(request)
            aiter_bytes, _, _, background = await OpenaiBase.to_openai(
                client, request, "/v1/images/generations"
            )
        finally:
            OpenaiBase.APP_SECRET, base.OPENAI_IMAGE_MODEL = (
                original_secret,
                original_pin,
            )
        async for _ in aiter_bytes:
            pass
        if background is not None:
            await background()
        await client.aclose()
        return json.loads(captured["content"])

    return asyncio.run(run())


def test_pin_is_applied_before_the_size_is_chosen():
    """The client sends gpt-image-1.5 and 3:4. With the pin to gpt-image-2 the request must
    leave as gpt-image-2 *and* with the exact size that model allows -- resolving the size
    first would snap it to the legacy 1024x1536."""
    forwarded = forward_generation(
        {"model": LEGACY, "prompt": "a cat", "n": 1, "size": "3:4"}, pin=FLEXIBLE
    )
    assert forwarded["model"] == FLEXIBLE
    assert forwarded["size"] == "1056x1408"


def test_without_the_pin_the_client_model_still_gets_legacy_sizes():
    forwarded = forward_generation(
        {"model": LEGACY, "prompt": "a cat", "n": 1, "size": "3:4"}
    )
    assert forwarded["model"] == LEGACY
    assert forwarded["size"] == "1024x1536"


def test_the_reported_bug_end_to_end():
    """3:4 and 9:21 used to produce the same image."""
    portrait_sizes = {
        ratio: forward_generation(
            {"model": LEGACY, "prompt": "a cat", "n": 1, "size": ratio}, pin=FLEXIBLE
        )["size"]
        for ratio in ("3:4", "9:16", "9:21")
    }
    assert len(set(portrait_sizes.values())) == 3, portrait_sizes


def test_auto_survives_for_a_model_that_understands_it():
    forwarded = forward_generation(
        {"model": LEGACY, "prompt": "a cat", "n": 1, "size": "auto"}, pin=FLEXIBLE
    )
    assert forwarded["size"] == "auto"


def test_auto_is_still_replaced_for_legacy_models():
    forwarded = forward_generation(
        {"model": LEGACY, "prompt": "a cat", "n": 1, "size": "auto"}
    )
    assert forwarded["size"] == "1024x1024"
