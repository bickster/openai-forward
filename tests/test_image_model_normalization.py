"""A client that asked for Flux still sends a Flux model id in the body.

Once Flux is out of the platform list those requests resolve to OpenAI
(`_resolve_platform` falls through to the first platform), and api.openai.com
answers 400 "The model 'flux-kontext' does not exist." These tests pin the
rewrite that keeps that from happening, on both image endpoints.

The requests go through `validate_request` first, exactly as `_reverse_proxy`
does, because that call is what caches the body for the passthrough path.
"""

import asyncio
import hashlib
import hmac
import json

import httpx
import pytest
from starlette.requests import Request

import openai_forward.base as base
from openai_forward.openai import OpenaiBase

SECRET = "test-secret"
BOUNDARY = "TestBoundary"


def edit_body(model: str, size: str | None = None) -> bytes:
    """The multipart body ChatsterGateway builds: model, prompt, n, image; no size."""
    parts = [
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n',
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nmake it blue\r\n',
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="n"\r\n\r\n1\r\n',
    ]
    if size is not None:
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="size"\r\n\r\n{size}\r\n'
        )
    parts.append(
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="image"; filename="image.png"\r\n'
        f'Content-Type: image/png\r\n\r\nPNGBYTES\r\n'
    )
    parts.append(f'--{BOUNDARY}--\r\n')
    return "".join(parts).encode()


def make_request(body: bytes, path: str, content_type: str) -> Request:
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "headers": [
            (b"content-type", content_type.encode()),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", b"Bearer test"),
            (b"x-imagemodel", b"flux"),
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

    return Request(scope, receive)


def forward(body: bytes, path: str, content_type: str) -> bytes:
    """Run the production path and return what would reach api.openai.com."""

    async def run():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = await request.aread()
            return httpx.Response(200, json={"data": []})

        client = httpx.AsyncClient(
            base_url=OpenaiBase.BASE_URL, transport=httpx.MockTransport(handler)
        )
        request = make_request(body, path, content_type)
        original_secret = OpenaiBase.APP_SECRET
        OpenaiBase.APP_SECRET = SECRET
        try:
            assert await OpenaiBase.validate_request(request)
            aiter_bytes, status_code, _, background = await OpenaiBase.to_openai(
                client, request, path
            )
        finally:
            OpenaiBase.APP_SECRET = original_secret
        if hasattr(aiter_bytes, "__aiter__"):
            async for _ in aiter_bytes:
                pass
        else:
            for _ in aiter_bytes:
                pass
        if background is not None:
            await background()
        await client.aclose()
        assert status_code == 200
        return captured["content"]

    return asyncio.run(run())


def forward_generation(payload: dict) -> dict:
    body = json.dumps(payload).encode()
    return json.loads(forward(body, "/v1/images/generations", "application/json"))


def forward_edit(body: bytes) -> bytes:
    return forward(
        body, "/v1/images/edits", f"multipart/form-data; boundary={BOUNDARY}"
    )


def test_generation_flux_model_is_rewritten():
    forwarded = forward_generation(
        {
            "model": "flux-kontext",
            "prompt": "a cat",
            "n": 1,
            "size": "16:9",
            "quality": "low",
        }
    )
    assert forwarded["model"] == "gpt-image-1.5"
    assert forwarded["size"] == "1536x1024"  # existing conversion still applies
    assert forwarded["prompt"] == "a cat"


def test_generation_openai_model_is_left_alone():
    forwarded = forward_generation(
        {"model": "gpt-image-1.5", "prompt": "a cat", "n": 1, "size": "1024x1024"}
    )
    assert forwarded["model"] == "gpt-image-1.5"


def test_edit_flux_model_is_rewritten_without_a_size_field():
    """The client sends no `size` on edits, so the rewrite has to trigger the
    rebuild on its own -- the size conversion never will."""
    forwarded = forward_edit(edit_body("flux-kontext"))
    assert b"gpt-image-1.5" in forwarded
    assert b"flux-kontext" not in forwarded
    assert b"PNGBYTES" in forwarded
    assert b"make it blue" in forwarded


def test_edit_openai_model_is_forwarded_untouched():
    body = edit_body("gpt-image-1.5")
    assert forward_edit(body) == body


@pytest.fixture
def pinned(monkeypatch):
    """OPENAI_IMAGE_MODEL set: every OpenAI-routed image request uses that model."""
    monkeypatch.setattr(base, "OPENAI_IMAGE_MODEL", "gpt-image-9")


def test_pin_overrides_the_model_the_client_asked_for(pinned):
    forwarded = forward_generation(
        {"model": "gpt-image-1.5", "prompt": "a cat", "n": 1, "size": "1024x1024"}
    )
    assert forwarded["model"] == "gpt-image-9"


def test_pin_also_covers_flux_clients(pinned):
    forwarded = forward_generation(
        {"model": "flux-kontext", "prompt": "a cat", "n": 1, "size": "16:9"}
    )
    assert forwarded["model"] == "gpt-image-9"
    assert forwarded["size"] == "1536x1024"


def test_pin_applies_to_edits(pinned):
    forwarded = forward_edit(edit_body("gpt-image-1.5"))
    assert b"gpt-image-9" in forwarded
    assert b"gpt-image-1.5" not in forwarded
    assert b"PNGBYTES" in forwarded


def test_pin_leaves_a_matching_request_byte_identical(pinned):
    """No pointless multipart rebuild when the client already sends the pinned model."""
    body = edit_body("gpt-image-9")
    assert forward_edit(body) == body
