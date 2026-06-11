from fastapi import Request

from .base import OpenaiBase
from .routers.schemas import OpenAIV1ChatCompletion


class Openai(OpenaiBase):
    def __init__(self):
        if self.IP_BLACKLIST or self.IP_WHITELIST:
            self.validate_host = True
        else:
            self.validate_host = False
        self.validate_ua = bool(self.UA_WHITELIST or self.UA_BLACKLIST)

    async def reverse_proxy(self, request: Request):
        if self.validate_host:
            self.validate_request_host(request)
        if self.validate_ua:
            self.validate_request_user_agent(request.headers.get("user-agent", ""))
        return await self._reverse_proxy(request)

    async def v1_chat_completions(self, data: OpenAIV1ChatCompletion, request: Request):
        ...
