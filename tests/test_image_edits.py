"""Regression tests for the /v1/images/edits body handling in OpenaiBase.to_openai."""

import asyncio

import httpx
from starlette.requests import Request

from openai_forward.openai import OpenaiBase

BOUNDARY = "TestBoundary"


def multipart_body(size: str) -> bytes:
    return (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="size"\r\n\r\n{size}\r\n'
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nmake it blue\r\n'
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="image"; filename="a.png"\r\n'
        f'Content-Type: image/png\r\n\r\nPNGBYTES\r\n'
        f'--{BOUNDARY}--\r\n'
    ).encode()


def multipart_body_multi_image(size: str) -> bytes:
    """A gpt-image-1 style edit: repeated image[] parts, plus a repeated field."""
    return (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="size"\r\n\r\n{size}\r\n'
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nmake it blue\r\n'
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="image[]"; filename="a.png"\r\n'
        f'Content-Type: image/png\r\n\r\nFIRSTIMAGE\r\n'
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="image[]"; filename="b.png"\r\n'
        f'Content-Type: image/png\r\n\r\nSECONDIMAGE\r\n'
        f'--{BOUNDARY}--\r\n'
    ).encode()


def make_request(body: bytes) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/edits",
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode()),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", b"Bearer test"),
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


def forward(body: bytes):
    """Run to_openai against a mock upstream; return (status, what it forwarded)."""

    async def run():
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = await request.aread()
            captured["content_type"] = request.headers.get("content-type")
            return httpx.Response(200, json={"data": []})

        client = httpx.AsyncClient(
            base_url=OpenaiBase.BASE_URL, transport=httpx.MockTransport(handler)
        )
        aiter_bytes, status_code, _, background = await OpenaiBase.to_openai(
            client, make_request(body), "/v1/images/edits"
        )
        async for _ in aiter_bytes:
            pass
        if background is not None:
            await background()
        await client.aclose()
        return status_code, captured

    return asyncio.run(run())


def test_image_edit_passthrough_when_size_needs_no_conversion():
    """request.form() consumes the request stream without caching it, so the
    untouched request.stream() generator used to raise RuntimeError('Stream
    consumed') for every edit that did not need a size rewrite."""
    body = multipart_body("1024x1024")
    status_code, captured = forward(body)
    assert status_code == 200
    assert captured["content"] == body


def test_image_edit_rebuilds_body_for_aspect_ratio_size():
    """Aspect-ratio sizes are still rewritten to valid OpenAI dimensions."""
    _, captured = forward(multipart_body("16:9"))
    forwarded = captured["content"]
    assert b"1536x1024" in forwarded
    assert b"16:9" not in forwarded
    assert b"PNGBYTES" in forwarded
    assert b"make it blue" in forwarded
    assert captured["content_type"].startswith("multipart/form-data; boundary=")


def test_image_edit_rebuild_keeps_every_repeated_field():
    """The rebuild exists only to rewrite `size`; every other part must survive.

    Iterating `for key in form` yields each key once and `form[key]` returns the
    last value, so repeated parts - such as the image[] list gpt-image-1 accepts -
    were silently collapsed to one.
    """
    _, captured = forward(multipart_body_multi_image("16:9"))
    forwarded = captured["content"]
    assert b"1536x1024" in forwarded
    assert b"FIRSTIMAGE" in forwarded
    assert b"SECONDIMAGE" in forwarded
    assert forwarded.count(b'name="image[]"') == 2
    assert b'filename="a.png"' in forwarded
    assert b'filename="b.png"' in forwarded
