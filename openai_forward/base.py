import os
from functools import lru_cache
from itertools import cycle
from math import gcd

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
import fnmatch
import re

from .routers.image_gen_platform import ImageGenPlatform, ImageEditPlatform
from .flux.bfl_api import FluxPro11, FluxKontextGen, FluxKontext, ContentModerationError
import json


# Image model ids are pinned in the client and only change with an app release, so both of
# these exist to move that decision to the server.
#
# OPENAI_IMAGE_MODEL: when set, every image request routed to OpenAI is sent with this model,
# whatever the client asked for. This is how a new OpenAI image model is rolled out -- set it,
# restart, done. Chat models deliberately have no equivalent: the client shapes chat params by
# model family (max_tokens vs max_completion_tokens, system vs developer role), so swapping the
# id server-side would send old-family params to a new-family endpoint. Chat stays on the
# client's own remote config.
#
# OPENAI_IMAGE_MODEL_FALLBACK: used only when there is no pin. A client that asked for Flux
# still puts a Flux model id in the body, and once the request routes to OpenAI (because Flux
# left the platform list) that id comes back as
# 400 "The model 'flux-kontext' does not exist." Deliberately a deny-list of Flux ids rather
# than an allow-list of OpenAI ones: OpenAI keeps adding image models, and an allow-list would
# reject ids this proxy has never heard of.
#
# Both rewrite an existing `model` field; neither adds one to a request that omits it.
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "").strip()
OPENAI_IMAGE_MODEL_FALLBACK = os.environ.get("OPENAI_IMAGE_MODEL_FALLBACK", "gpt-image-1.5").strip()


def _normalize_image_model(model) -> str | None:
    """Return the model to send to OpenAI, or None if the value can stay as it is."""
    if not isinstance(model, str):
        return None
    if OPENAI_IMAGE_MODEL:
        return None if model == OPENAI_IMAGE_MODEL else OPENAI_IMAGE_MODEL
    if model.strip().lower().startswith("flux"):
        return OPENAI_IMAGE_MODEL_FALLBACK
    return None


# Models that accept an arbitrary `size` instead of the three legacy dimensions. Kept as an
# allow-list, which is the opposite of the model deny-list above, because the failure modes
# are opposite: a legacy size is valid for every model, while an arbitrary size sent to a
# model that only takes the legacy three is a 400. An unrecognised model therefore gets the
# conservative treatment. Add ids here (comma separated, prefix match) as they ship.
FLEXIBLE_SIZE_MODELS = tuple(
    m.strip().lower()
    for m in os.environ.get("OPENAI_IMAGE_FLEXIBLE_SIZE_MODELS", "gpt-image-2").split(",")
    if m.strip()
)

# gpt-image-2 size rules: both edges a multiple of 16, longest edge <= 3840, long:short <= 3:1,
# and total pixels within these bounds.
_SIZE_STEP = 16
_MAX_EDGE = 3840
_MAX_ASPECT = 3.0
_MIN_PIXELS = 655_360
_MAX_PIXELS = 8_294_400


def _model_takes_any_size(model) -> bool:
    return isinstance(model, str) and model.strip().lower().startswith(FLEXIBLE_SIZE_MODELS)


def _legacy_dimensions(w_ratio: int, h_ratio: int) -> tuple[int, int]:
    """The three sizes every GPT Image model before gpt-image-2 accepts, by orientation."""
    if w_ratio > h_ratio:
        return 1536, 1024
    if h_ratio > w_ratio:
        return 1024, 1536
    return 1024, 1024


@lru_cache(maxsize=64)
def _exact_dimensions(w_ratio: int, h_ratio: int) -> tuple[int, int] | None:
    """Largest size with exactly this aspect ratio that stays within the pixel budget the
    legacy size for the same orientation would have used, so honouring the ratio never costs
    more than snapping to it did. Falls back to the smallest valid size above the budget, and
    to None when the ratio cannot be expressed at all (too elongated, or no multiple of 16
    lands inside the pixel bounds).
    """
    if max(w_ratio, h_ratio) / min(w_ratio, h_ratio) > _MAX_ASPECT:
        return None

    divisor = gcd(w_ratio, h_ratio)
    unit_w, unit_h = w_ratio // divisor, h_ratio // divisor

    candidates = []
    k = 1
    while max(unit_w * k, unit_h * k) <= _MAX_EDGE:
        w, h = unit_w * k, unit_h * k
        if w % _SIZE_STEP == 0 and h % _SIZE_STEP == 0 and _MIN_PIXELS <= w * h <= _MAX_PIXELS:
            candidates.append((w, h))
        k += 1
    if not candidates:
        return None

    legacy_w, legacy_h = _legacy_dimensions(w_ratio, h_ratio)
    budget = legacy_w * legacy_h
    within = [c for c in candidates if c[0] * c[1] <= budget]
    if within:
        return max(within, key=lambda c: c[0] * c[1])
    return min(candidates, key=lambda c: c[0] * c[1])


def _aspect_ratio_to_openai_dimensions(aspect_ratio: str, model=None) -> tuple[int, int]:
    """Convert an aspect ratio string such as "16:9" to dimensions OpenAI will accept.

    gpt-image-2 takes any size meeting its constraints, so the ratio is honoured exactly.
    Earlier models take only 1024x1024, 1536x1024 and 1024x1536, so the ratio is snapped to
    whichever matches its orientation -- which is why every portrait ratio used to come back
    the same shape.
    """
    try:
        w_ratio, h_ratio = map(int, aspect_ratio.split(':'))
    except (ValueError, TypeError, AttributeError):
        return 1024, 1024
    if w_ratio <= 0 or h_ratio <= 0:
        return 1024, 1024

    if _model_takes_any_size(model):
        exact = _exact_dimensions(w_ratio, h_ratio)
        if exact is not None:
            return exact

    return _legacy_dimensions(w_ratio, h_ratio)


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
    # Comma-separated glob patterns (e.g. "okhttp/3.9.*"); comma because UA strings contain spaces
    UA_WHITELIST = env2list("UA_WHITELIST", sep=",")
    UA_BLACKLIST = env2list("UA_BLACKLIST", sep=",")
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

    logger.debug(f"IMAGE_GEN_PLATFORM env: {_IMAGE_GEN_PLATFORMS_STR!r}")
    logger.debug(f"IMAGE_GEN_PLATFORMS resolved: {[p.name for p in IMAGE_GEN_PLATFORMS]}")
    logger.debug(f"IMAGE_EDIT_PLATFORM env: {_IMAGE_EDIT_PLATFORMS_STR!r}")
    logger.debug(f"IMAGE_EDIT_PLATFORMS resolved: {[p.name for p in IMAGE_EDIT_PLATFORMS]}")
    # info, not debug: the startup banner's column can be clipped at default terminal width,
    # so this is the line that reliably shows the pin in journalctl.
    logger.info(
        f"Image model pin: {OPENAI_IMAGE_MODEL or 'none (client chooses; flux ids -> ' + OPENAI_IMAGE_MODEL_FALLBACK + ')'}"
    )

    print_startup_info(
        BASE_URL, ROUTE_PREFIX, _openai_api_key_list, _no_auth_mode, _LOG_CHAT, IMAGE_GEN_PLATFORMS, IMAGE_EDIT_PLATFORMS,
        OPENAI_IMAGE_MODEL
    )
    if _LOG_CHAT:
        setting_log(save_file=False)
        chatsaver = ChatSaver()

    @classmethod
    def _compile_ua_patterns(cls):
        cls._ua_whitelist_patterns = [
            re.compile(fnmatch.translate(p), re.IGNORECASE) for p in cls.UA_WHITELIST
        ]
        cls._ua_blacklist_patterns = [
            re.compile(fnmatch.translate(p), re.IGNORECASE) for p in cls.UA_BLACKLIST
        ]

    def validate_request_user_agent(self, user_agent: str):
        # detail must stay identical to the HMAC-failure response so blocked
        # clients can't tell which gate rejected them
        forbidden = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )
        if not user_agent:
            logger.info("UA filter: blocked request with missing User-Agent")
            raise forbidden
        if any(p.match(user_agent) for p in self._ua_whitelist_patterns):
            return
        if any(p.match(user_agent) for p in self._ua_blacklist_patterns):
            logger.info(f"UA filter: blocked user-agent={user_agent!r}")
            raise forbidden

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
        logger.debug(f"_resolve_platform: platforms={[p.name for p in platforms]}, header_value={header_value!r}")
        if header_value in ("openai", "flux"):
            for p in platforms:
                logger.debug(f"_resolve_platform: checking {p.name} (family={p.family})")
                if p.family == header_value:
                    logger.debug(f"_resolve_platform: matched {p.name} by family")
                    return p
        logger.debug(f"_resolve_platform: falling through to default platforms[0]={platforms[0].name}")
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
        logger.debug(f"_reverse_proxy: url_path={url_path}, x-imagemodel={image_model!r}")
        logger.debug(f"_reverse_proxy: all request headers={dict(request.headers)}")

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
                # Resolve the model first: it decides which sizes are legal below.
                new_model = _normalize_image_model(data.get('model'))
                if new_model is not None:
                    logger.info(f"Rewrote model '{data.get('model')}' -> '{new_model}'")
                    data['model'] = new_model
                model = data.get('model')

                size = data.get('size', '1024x1024')
                if ':' in size:
                    w, h = _aspect_ratio_to_openai_dimensions(size, model)
                    data['size'] = f"{w}x{h}"
                    logger.info(f"Converted size '{size}' -> '{data['size']}' for model {model!r}")
                elif 'x' not in size and not (size == 'auto' and _model_takes_any_size(model)):
                    # 'auto' is a real value for models that accept any size -- let it through
                    # rather than pinning them to a square.
                    data['size'] = '1024x1024'
                content = json.dumps(data).encode()
            except Exception as e:
                logger.debug(f"Failed to parse image generation body for size conversion: {e}")

        elif url_path.endswith("images/edits"):
            try:
                # Cache the body before parsing the form: request.form() consumes the
                # request stream without caching it, which would leave `content` (the
                # request.stream() generator created above) raising "Stream consumed"
                # when httpx forwards an unmodified body.
                await request.body()
                form = await request.form()
                size = form.get('size', '1024x1024')
                needs_rebuild = False

                new_model = _normalize_image_model(form.get('model'))
                if new_model is not None:
                    logger.info(f"Rewrote edit model '{form.get('model')}' -> '{new_model}'")
                    needs_rebuild = True
                model = new_model if new_model is not None else form.get('model')

                if isinstance(size, str) and ':' in size:
                    w, h = _aspect_ratio_to_openai_dimensions(size, model)
                    new_size = f"{w}x{h}"
                    logger.info(f"Converted edit size '{size}' -> '{new_size}' for model {model!r}")
                    needs_rebuild = True
                else:
                    new_size = size

                if needs_rebuild:
                    import uuid
                    boundary = f"----OpenAIForwardBoundary{uuid.uuid4().hex}"
                    parts = []

                    # multi_items() yields every part; `for key in form` would
                    # yield each key once and form[key] returns only the last
                    # value, dropping repeated parts such as image[] lists.
                    for key, value in form.multi_items():
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
                            if key == 'size':
                                field_value = new_size
                            elif key == 'model' and new_model is not None:
                                field_value = new_model
                            else:
                                field_value = str(value)
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


OpenaiBase._compile_ua_patterns()
