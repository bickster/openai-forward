import os
from itertools import cycle

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger
from starlette.background import BackgroundTask

from .config import print_startup_info, setting_log
from .content.chat import ChatSaver
from .tool import env2list

import hmac
import hashlib

from .routers.image_gen_platform import ImageGenPlatform, ImageEditPlatform
from .flux.bfl_api import FluxPro11, FluxKontextGen, FluxKontext, ContentModerationError
import json


def _aspect_ratio_to_openai_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """Convert aspect ratio string to the nearest valid OpenAI dimensions.

    OpenAI only accepts: 1024x1024, 1536x1024 (landscape), 1024x1536 (portrait).
    """
    try:
        w_ratio, h_ratio = map(int, aspect_ratio.split(':'))
    except (ValueError, TypeError):
        return 1024, 1024

    if w_ratio > h_ratio:
        return 1536, 1024
    elif h_ratio > w_ratio:
        return 1024, 1536
    else:
        return 1024, 1024


class OpenaiBase:
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").strip()
    ROUTE_PREFIX = os.environ.get("ROUTE_PREFIX", "").strip()
    _LOG_CHAT = os.environ.get("LOG_CHAT", "False").strip().lower() == "true"
    _openai_api_key_list = env2list("OPENAI_API_KEY", sep=" ")
    _cycle_api_key = cycle(_openai_api_key_list)
    _FWD_KEYS = set(env2list("FORWARD_KEY", sep=" "))
    _no_auth_mode = _openai_api_key_list != [] and _FWD_KEYS == set()
    IP_WHITELIST = env2list("IP_WHITELIST", sep=" ")
    IP_BLACKLIST = env2list("IP_BLACKLIST", sep=" ")
    APP_SECRET = os.environ.get("APP_SECRET", "").strip()
    _IMAGE_GEN_PLATFORMS_STR = os.environ.get("IMAGE_GEN_PLATFORM", "dalle3").strip()
    _IMAGE_EDIT_PLATFORMS_STR = os.environ.get("IMAGE_EDIT_PLATFORM", "openai").strip()

    if ROUTE_PREFIX:
        if ROUTE_PREFIX.endswith("/"):
            ROUTE_PREFIX = ROUTE_PREFIX[:-1]
        if not ROUTE_PREFIX.startswith("/"):
            ROUTE_PREFIX = "/" + ROUTE_PREFIX
    timeout = 600

    IMAGE_GEN_PLATFORMS = [ImageGenPlatform[p.strip()] for p in _IMAGE_GEN_PLATFORMS_STR.split(",")]
    IMAGE_EDIT_PLATFORMS = [ImageEditPlatform[p.strip()] for p in _IMAGE_EDIT_PLATFORMS_STR.split(",")]

    print_startup_info(
        BASE_URL, ROUTE_PREFIX, _openai_api_key_list, _no_auth_mode, _LOG_CHAT, IMAGE_GEN_PLATFORMS, IMAGE_EDIT_PLATFORMS
    )
    if _LOG_CHAT:
        setting_log(save_file=False)
        chatsaver = ChatSaver()

    def validate_request_host(self, request: Request):
        ip = request.client.host
        if self.IP_WHITELIST and ip not in self.IP_WHITELIST:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden, ip={ip} not in whitelist!",
            )
        if self.IP_BLACKLIST and ip in self.IP_BLACKLIST:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden, ip={ip} in blacklist!",
            )

        forward_for = request.headers.get("x-forwarded-for")
        if self.IP_BLACKLIST and forward_for in self.IP_BLACKLIST:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden",
            )

    @classmethod
    async def aiter_bytes(cls, r: httpx.Response, route_path: str, uid: str):
        bytes_ = b""
        async for chunk in r.aiter_bytes():
            bytes_ += chunk
            yield chunk
        try:
            target_info = cls.chatsaver.parse_bytes_to_content(bytes_, route_path)
            cls.chatsaver.add_chat(
                {target_info["role"]: target_info["content"], "uid": uid}
            )
        except Exception as e:
            logger.debug(f"log chat (not) error:\n{e=}")

    @classmethod
    async def validate_request(cls, request: Request):
        signature = request.headers.get('X-Request-Signature')
        if not signature:
            return False
        request_data = await request.body()
        expected_signature = hmac.new(cls.APP_SECRET.encode(), request_data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected_signature)

    @staticmethod
    def _resolve_platform(platforms, header_value):
        """Pick platform from list based on X-ImageModel header.

        If header matches a family ("openai"/"flux"), return the first
        platform in that family. Otherwise return the first platform (default).
        """
        if header_value in ("openai", "flux"):
            for p in platforms:
                if p.family == header_value:
                    return p
        return platforms[0]

    @classmethod
    async def _reverse_proxy(cls, request: Request):
        if not await cls.validate_request(request):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden",
            )

        client = httpx.AsyncClient(base_url=cls.BASE_URL, http1=True, http2=False)
        url_path = request.url.path
        url_path = url_path[len(cls.ROUTE_PREFIX):]

        image_model = request.headers.get("x-imagemodel", "").strip().lower()

        if url_path.endswith("images/generations"):
            platform = cls._resolve_platform(cls.IMAGE_GEN_PLATFORMS, image_model)
            logger.info(f"Image generation -> {platform.name}")

            match platform:
                case ImageGenPlatform.dalle3 | ImageGenPlatform.openai:
                    aiter_bytes, status_code, media_type, background = await cls.to_openai(client, request, url_path)

                    return StreamingResponse(
                        aiter_bytes,
                        status_code=status_code,
                        media_type=media_type,
                        background=background
                    )

                case ImageGenPlatform.flux1_1:
                    try:
                        json_response, content_length = await cls.to_flux(client, request, url_path)

                        return StreamingResponse(
                            json_response,
                            status_code=200,
                            headers={"Content-Length": str(content_length)},
                            media_type="application/json"
                        )
                    except ContentModerationError as e:
                        return JSONResponse(
                            content={
                                "error": {
                                    "code": "content_policy_violation",
                                    "message": e.message,
                                    "type": "content_policy_violation"
                                }
                            },
                            status_code=200
                        )

                case ImageGenPlatform.flux1_kontext:
                    try:
                        json_response, content_length = await cls.to_flux_kontext_gen(client, request, url_path)

                        return StreamingResponse(
                            json_response,
                            status_code=200,
                            headers={"Content-Length": str(content_length)},
                            media_type="application/json"
                        )
                    except ContentModerationError as e:
                        return JSONResponse(
                            content={
                                "error": {
                                    "code": "content_policy_violation",
                                    "message": e.message,
                                    "type": "content_policy_violation"
                                }
                            },
                            status_code=200
                        )
        elif url_path.endswith("images/edits"):
            platform = cls._resolve_platform(cls.IMAGE_EDIT_PLATFORMS, image_model)
            logger.info(f"Image edit -> {platform.name}")

            match platform:
                case ImageEditPlatform.openai:
                    aiter_bytes, status_code, media_type, background = await cls.to_openai(client, request, url_path)

                    return StreamingResponse(
                        aiter_bytes,
                        status_code=status_code,
                        media_type=media_type,
                        background=background
                    )

                case ImageEditPlatform.flux1_kontext:
                    try:
                        json_response, content_length = await cls.to_flux_kontext(client, request, url_path)

                        return StreamingResponse(
                            json_response,
                            status_code=200,
                            headers={"Content-Length": str(content_length)},
                            media_type="application/json"
                        )
                    except ContentModerationError as e:
                        return JSONResponse(
                            content={
                                "error": {
                                    "code": "content_policy_violation",
                                    "message": e.message,
                                    "type": "content_policy_violation"
                                }
                            },
                            status_code=200
                        )
        else:
            aiter_bytes, status_code, media_type, background = await cls.to_openai(client, request, url_path)

            return StreamingResponse(
                aiter_bytes,
                status_code=status_code,
                media_type=media_type,
                background=background
            )

    @classmethod
    async def to_flux(cls, client, request, url_path):
        logger.info("to_flux: generate")

        flux = FluxPro11()
        return await flux.generate_image(request)

    @classmethod
    async def to_flux_kontext_gen(cls, client, request, url_path):
        logger.info("to_flux_kontext: generate")

        flux_kontext_gen = FluxKontextGen()
        return await flux_kontext_gen.generate_image(request)

    @classmethod
    async def to_flux_kontext(cls, client, request, url_path):
        logger.info("to_flux_kontext: edit")

        flux_kontext = FluxKontext()
        return await flux_kontext.generate_image(request)

    @classmethod
    async def to_openai(cls, client, request, url_path):
        # Configure URL
        url = httpx.URL(path=url_path, query=request.url.query.encode("utf-8"))
        headers = dict(request.headers)
        auth = headers.pop("authorization", "")
        auth_headers_dict = {"Content-Type": headers.get("content-type", "application/json"), "Authorization": auth}
        auth_prefix = "Bearer "
        if cls._no_auth_mode or auth and auth[len(auth_prefix):] in cls._FWD_KEYS:
            auth = auth_prefix + next(cls._cycle_api_key)
            auth_headers_dict["Authorization"] = auth
        log_chat_completions = False
        uid = None
        if cls._LOG_CHAT and request.method == "POST":
            try:
                chat_info = await cls.chatsaver.parse_payload_to_content(
                    request, route_path=url_path
                )
                if chat_info:
                    cls.chatsaver.add_chat(chat_info)
                    uid = chat_info.get("uid")
                    log_chat_completions = True
            except Exception as e:
                logger.debug(
                    f"log chat error:\n{request.client.host=} {request.method=}: {e}"
                )
        # Convert aspect ratio to dimensions for OpenAI image generation
        content = request.stream()
        if url_path.endswith("images/generations"):
            try:
                body = await request.body()
                data = json.loads(body)
                size = data.get('size', '1024x1024')
                if ':' in size:
                    w, h = _aspect_ratio_to_openai_dimensions(size)
                    data['size'] = f"{w}x{h}"
                    logger.info(f"Converted size '{size}' -> '{data['size']}'")
                elif 'x' not in size:
                    data['size'] = '1024x1024'
                content = json.dumps(data).encode()
            except Exception as e:
                logger.debug(f"Failed to parse image generation body for size conversion: {e}")

        elif url_path.endswith("images/edits"):
            try:
                form = await request.form()
                size = form.get('size', '1024x1024')
                needs_rebuild = False

                if isinstance(size, str) and ':' in size:
                    w, h = _aspect_ratio_to_openai_dimensions(size)
                    new_size = f"{w}x{h}"
                    logger.info(f"Converted edit size '{size}' -> '{new_size}'")
                    needs_rebuild = True
                else:
                    new_size = size

                if needs_rebuild:
                    import uuid
                    boundary = f"----OpenAIForwardBoundary{uuid.uuid4().hex}"
                    parts = []

                    for key in form:
                        value = form[key]
                        if hasattr(value, 'read'):  # UploadFile
                            file_bytes = await value.read()
                            parts.append(
                                f'--{boundary}\r\n'
                                f'Content-Disposition: form-data; name="{key}"; filename="{value.filename}"\r\n'
                                f'Content-Type: {value.content_type}\r\n\r\n'
                            )
                            parts.append(file_bytes)
                            parts.append(b'\r\n')
                        else:
                            field_value = new_size if key == 'size' else str(value)
                            parts.append(
                                f'--{boundary}\r\n'
                                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                                f'{field_value}\r\n'
                            )

                    parts.append(f'--{boundary}--\r\n')

                    body_bytes = b''
                    for part in parts:
                        body_bytes += part.encode('utf-8') if isinstance(part, str) else part

                    content = body_bytes
                    auth_headers_dict["Content-Type"] = f"multipart/form-data; boundary={boundary}"

            except Exception as e:
                logger.debug(f"Failed to parse image edit body for size conversion: {e}")

        logger.info(f"to_openai: {request.method} {url}")

        req = client.build_request(
            request.method,
            url,
            headers=auth_headers_dict,
            content=content,
            timeout=cls.timeout,
        )
        try:
            r = await client.send(req, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            error_info = (
                f"{type(e)}: {e} | "
                f"Please check if host={request.client.host} can access [{cls.BASE_URL}] successfully?"
            )
            logger.error(error_info)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=error_info
            )
        except Exception as e:
            logger.exception(f"{type(e)}:")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e
            )
        logger.info(f"to_openai response: status={r.status_code} content-type={r.headers.get('content-type')}")
        if r.status_code >= 400:
            response_body = await r.aread()
            logger.error(f"to_openai error response body: {response_body.decode(errors='replace')}")
            return iter([response_body]), r.status_code, r.headers.get("content-type"), BackgroundTask(r.aclose)

        # Get bytes from response
        aiter_bytes = (
            cls.aiter_bytes(r, url_path, uid)
            if log_chat_completions
            else r.aiter_bytes()
        )
        return aiter_bytes, r.status_code, r.headers.get("content-type"), BackgroundTask(r.aclose)
