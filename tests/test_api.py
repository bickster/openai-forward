from itertools import cycle

import pytest
from fastapi import HTTPException, Request

from openai_forward.openai import OpenaiBase


@pytest.fixture(scope="module")
def openai() -> OpenaiBase:
    return OpenaiBase()


class TestOpenai:
    @staticmethod
    def teardown_method():
        OpenaiBase.IP_BLACKLIST = []
        OpenaiBase.IP_WHITELIST = []
        OpenaiBase._default_api_key_list = []

    def test_env(self, openai: OpenaiBase):
        assert openai.BASE_URL == "https://api.openai.com"

    def test_api_keys(self, openai: OpenaiBase):
        assert openai._default_api_key_list == []
        openai._default_api_key_list = ["a", "b"]
        openai._cycle_api_key = cycle(openai._default_api_key_list)
        assert next(openai._cycle_api_key) == "a"
        assert next(openai._cycle_api_key) == "b"
        assert next(openai._cycle_api_key) == "a"
        assert next(openai._cycle_api_key) == "b"
        assert next(openai._cycle_api_key) == "a"

    def test_validate_ip(self, openai: OpenaiBase):
        ip1 = "1.1.1.1"
        ip2 = "2.2.2.2"
        request1 = Request({"type": "http", "client": (ip1, 1234), "headers": []})
        request2 = Request({"type": "http", "client": (ip2, 1234), "headers": []})
        assert openai.validate_request_host(request1) is None
        openai.IP_WHITELIST.append(ip1)
        assert openai.validate_request_host(request1) is None
        with pytest.raises(HTTPException):
            openai.validate_request_host(request2)
        openai.IP_WHITELIST = []
        openai.IP_BLACKLIST.append(ip1)
        assert openai.validate_request_host(request2) is None
        with pytest.raises(HTTPException):
            openai.validate_request_host(request1)
