import asyncio
import hashlib
import hmac
import importlib.util
import os
import sys

import pytest
from fastapi import HTTPException, Request

from openai_forward import base
from openai_forward.__main__ import main


@pytest.fixture
def configured_proxy(monkeypatch):
    def configure(value):
        if value is None:
            monkeypatch.delenv("APP_SECRET", raising=False)
        else:
            monkeypatch.setenv("APP_SECRET", value)
        monkeypatch.setenv("LOG_CHAT", "False")
        # Load configuration without replacing classes used by other tests.
        spec = importlib.util.spec_from_file_location(
            "openai_forward._test_secret_config", base.__file__
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.OpenaiBase

    return configure


def signed_request(secret, body=b'{"message": "hello"}', signature=None):
    if signature is None:
        signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "headers": [(b"x-request-signature", signature.encode("latin-1"))],
        },
        receive=receive,
    )


@pytest.mark.parametrize(
    "configuration,secret",
    [
        ("old-secret", "old-secret"),
        ("old-secret new-secret other-app-secret", "old-secret"),
        ("old-secret new-secret other-app-secret", "new-secret"),
        ("old-secret new-secret other-app-secret", "other-app-secret"),
        ("  old-secret   new-secret  ", "new-secret"),
        ("old,secret", "old,secret"),
        ("old,secret new,secret", "old,secret"),
        ("old,secret new,secret", "new,secret"),
    ],
)
def test_accepts_configured_secrets(configured_proxy, configuration, secret):
    proxy = configured_proxy(configuration)
    request = signed_request(secret)
    assert asyncio.run(proxy.validate_request(request)) is True
    # Validation leaves the original bytes available for forwarding.
    assert asyncio.run(request.body()) == b'{"message": "hello"}'


@pytest.mark.parametrize("secret", ["old", "secret", "new"])
def test_rejects_parts_of_comma_containing_secrets(configured_proxy, secret):
    proxy = configured_proxy("old,secret new,secret")
    assert asyncio.run(proxy.validate_request(signed_request(secret))) is False


@pytest.mark.parametrize(
    "configuration", ["old-secret new-secret", "new-secret", "", "   "]
)
def test_rejects_unknown_or_retired_secret(configured_proxy, configuration):
    proxy = configured_proxy(configuration)
    assert (
        asyncio.run(proxy.validate_request(signed_request("retired-secret"))) is False
    )


@pytest.mark.parametrize("configuration", [None, "", "   ", "old-secret   new-secret"])
def test_never_accepts_empty_secret(configured_proxy, configuration):
    proxy = configured_proxy(configuration)
    assert asyncio.run(proxy.validate_request(signed_request(""))) is False


def test_rotation_removes_old_secret(configured_proxy):
    proxy = configured_proxy("old,secret new,secret")
    assert asyncio.run(proxy.validate_request(signed_request("old,secret"))) is True
    assert asyncio.run(proxy.validate_request(signed_request("new,secret"))) is True
    proxy = configured_proxy("new,secret")
    assert asyncio.run(proxy.validate_request(signed_request("old,secret"))) is False
    assert asyncio.run(proxy.validate_request(signed_request("new,secret"))) is True


def test_rejects_modified_body(configured_proxy):
    proxy = configured_proxy("old-secret new-secret")
    signature = hmac.new(b"new-secret", b'{"amount":1}', hashlib.sha256).hexdigest()
    request = signed_request("new-secret", body=b'{"amount":2}', signature=signature)
    assert asyncio.run(proxy.validate_request(request)) is False


@pytest.mark.parametrize("signature", ["", "invalid", "0" * 64, "\u00e9" * 64])
def test_invalid_signature_returns_generic_403(configured_proxy, signature):
    proxy = configured_proxy("old-secret new-secret")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            proxy._reverse_proxy(signed_request("new-secret", signature=signature))
        )
    assert exc.value.status_code == 403
    assert exc.value.detail == "Forbidden"


def test_missing_signature_returns_generic_403(configured_proxy):
    proxy = configured_proxy("old-secret new-secret")
    request = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(proxy._reverse_proxy(request))
    assert exc.value.status_code == 403
    assert exc.value.detail == "Forbidden"


@pytest.mark.parametrize(
    "configuration,secret",
    [
        ("old-secret", "old-secret"),
        ("old-secret new-secret", "old-secret"),
        ("old-secret new-secret", "new-secret"),
        ("old,secret new,secret", "old,secret"),
        ("old,secret new,secret", "new,secret"),
    ],
)
def test_cli_uses_existing_app_secret_setting(
    monkeypatch, configured_proxy, configuration, secret
):
    monkeypatch.setenv("APP_SECRET", "previous-config")
    monkeypatch.setattr("openai_forward.__main__.uvicorn.run", lambda **kwargs: None)
    monkeypatch.setattr(
        sys, "argv", ["openai-forward", "run", f"--app_secret={configuration}"]
    )
    main()
    assert os.environ["APP_SECRET"] == configuration
    proxy = configured_proxy(os.environ["APP_SECRET"])
    assert asyncio.run(proxy.validate_request(signed_request(secret))) is True
