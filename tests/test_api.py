from itertools import cycle

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from openai_forward.openai import OpenaiBase


@pytest.fixture(scope="module")
def openai() -> OpenaiBase:
    return OpenaiBase()


def make_request(ip: str, forwarded_for: str | None = None) -> Request:
    """Minimal Request carrying a client IP, for validate_request_host."""
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode()))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "client": (ip, 1234),
            "headers": headers,
        }
    )


class TestOpenai:
    @staticmethod
    def teardown_method():
        OpenaiBase.IP_BLACKLIST = []
        OpenaiBase.IP_WHITELIST = []
        OpenaiBase.UA_BLACKLIST = []
        OpenaiBase.UA_WHITELIST = []
        OpenaiBase._compile_ua_patterns()
        OpenaiBase._openai_api_key_list = []
        OpenaiBase._cycle_api_key = cycle(OpenaiBase._openai_api_key_list)

    def test_env(self, openai: OpenaiBase):
        assert openai.BASE_URL == "https://api.openai.com"

    def test_api_keys(self, openai: OpenaiBase):
        assert openai._openai_api_key_list == []
        OpenaiBase._openai_api_key_list = ["a", "b"]
        OpenaiBase._cycle_api_key = cycle(OpenaiBase._openai_api_key_list)
        assert next(openai._cycle_api_key) == "a"
        assert next(openai._cycle_api_key) == "b"
        assert next(openai._cycle_api_key) == "a"
        assert next(openai._cycle_api_key) == "b"
        assert next(openai._cycle_api_key) == "a"

    def test_validate_ip(self, openai: OpenaiBase):
        ip1 = "1.1.1.1"
        ip2 = "2.2.2.2"
        assert openai.validate_request_host(make_request("*")) is None
        OpenaiBase.IP_WHITELIST.append(ip1)
        assert openai.validate_request_host(make_request(ip1)) is None
        with pytest.raises(HTTPException):
            openai.validate_request_host(make_request(ip2))
        OpenaiBase.IP_WHITELIST = []
        OpenaiBase.IP_BLACKLIST.append(ip1)
        assert openai.validate_request_host(make_request(ip2)) is None
        with pytest.raises(HTTPException):
            openai.validate_request_host(make_request(ip1))

    def test_validate_forwarded_for(self, openai: OpenaiBase):
        ip1 = "1.1.1.1"
        OpenaiBase.IP_BLACKLIST.append(ip1)
        # x-forwarded-for is matched exactly against the blacklist
        assert openai.validate_request_host(make_request("9.9.9.9")) is None
        with pytest.raises(HTTPException):
            openai.validate_request_host(make_request("9.9.9.9", forwarded_for=ip1))

    def test_validate_user_agent(self, openai: OpenaiBase):
        my_app_ua = "okhttp/3.9.3"
        bad_ua = "okhttp/5.0.0-alpha.2"
        browser_ua = "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36"

        # no lists configured -> nothing matches, everything passes
        assert openai.validate_request_user_agent(browser_ua) is None

        # blacklist alone
        OpenaiBase.UA_BLACKLIST = ["okhttp/*"]
        OpenaiBase._compile_ua_patterns()
        with pytest.raises(HTTPException) as exc_info:
            openai.validate_request_user_agent(bad_ua)
        # response must not hint at the reason for the block
        assert exc_info.value.detail == "Forbidden"
        assert openai.validate_request_user_agent(browser_ua) is None

        # whitelist overrides blacklist
        OpenaiBase.UA_WHITELIST = ["okhttp/3.9.*"]
        OpenaiBase._compile_ua_patterns()
        assert openai.validate_request_user_agent(my_app_ua) is None
        with pytest.raises(HTTPException):
            openai.validate_request_user_agent(bad_ua)
        assert openai.validate_request_user_agent(browser_ua) is None

        # strict mode: blacklist * blocks everything not whitelisted
        OpenaiBase.UA_BLACKLIST = ["*"]
        OpenaiBase._compile_ua_patterns()
        assert openai.validate_request_user_agent(my_app_ua) is None
        with pytest.raises(HTTPException):
            openai.validate_request_user_agent(browser_ua)

        # missing/empty UA is blocked whenever filtering is enabled
        OpenaiBase.UA_BLACKLIST = ["okhttp/*"]
        OpenaiBase._compile_ua_patterns()
        with pytest.raises(HTTPException):
            openai.validate_request_user_agent("")

        # matching is case-insensitive
        with pytest.raises(HTTPException):
            openai.validate_request_user_agent("OkHttp/5.0")

        # patterns match the full UA string, not a prefix
        OpenaiBase.UA_BLACKLIST = ["okhttp"]
        OpenaiBase._compile_ua_patterns()
        assert openai.validate_request_user_agent(my_app_ua) is None
