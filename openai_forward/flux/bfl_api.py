import asyncio
import httpx
from fastapi import HTTPException, Request, status

from loguru import logger
import json
import base64
import os
import time
from math import gcd

BASE_URL = "https://api.bfl.ai/"
BFL_API_KEY = os.environ.get("BFL_API_KEY", None)

JSON_PREFIX = f'''{{
    "created": {int(time.time())},
    "data": [
        {{
            "b64_json": "'''
JSON_MID = '", "revised_prompt": '
JSON_SUFFIX = f'''
        }}
    ]
}}
'''


def _clamp_dimensions(width: int, height: int) -> tuple[int, int]:
    """Clamp dimensions to FLUX 1.1 Pro constraints while preserving aspect ratio."""
    MIN_DIM = 256
    MAX_DIM = 1440

    # Scale down if either dimension exceeds max
    if width > MAX_DIM or height > MAX_DIM:
        scale = min(MAX_DIM / width, MAX_DIM / height)
        width = int(width * scale)
        height = int(height * scale)

    # Scale up if either dimension is below min
    if width < MIN_DIM or height < MIN_DIM:
        scale = max(MIN_DIM / width, MIN_DIM / height)
        width = int(width * scale)
        height = int(height * scale)

    # Round to nearest multiple of 32
    width = max(MIN_DIM, min(MAX_DIM, round(width / 32) * 32))
    height = max(MIN_DIM, min(MAX_DIM, round(height / 32) * 32))

    return width, height


def _aspect_ratio_to_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """Convert aspect ratio string (e.g., '16:9') to dimensions with larger side = 1440."""
    MAX_DIM = 1440

    try:
        w_ratio, h_ratio = map(int, aspect_ratio.split(':'))
    except (ValueError, TypeError):
        return 1024, 1024  # Default to square

    # Set larger dimension to MAX_DIM, calculate smaller proportionally
    if w_ratio >= h_ratio:
        width = MAX_DIM
        height = int(MAX_DIM * h_ratio / w_ratio)
    else:
        height = MAX_DIM
        width = int(MAX_DIM * w_ratio / h_ratio)

    # Round to nearest multiple of 32
    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class FluxBase:
    API_ENDPOINT = ""
    ACCEPT = ""

    TIMEOUT = 600

    @classmethod
    async def generate_image(cls, request: Request):
        headers = {
            "Accept": cls.ACCEPT,
            "x-key": BFL_API_KEY,
        }

        if headers["x-key"] is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid configuration.",
            )

        polling_url = await cls._image_request(request, headers)

        image_url, prompt = await cls._poll_for_result(polling_url, headers)

        # Escape special characters in prompt
        prompt = json.dumps(prompt)

        return (
            cls._stream_image(image_url, prompt),
            await cls._image_size(image_url, prompt)
        )

    @classmethod
    async def _image_request(cls, request: Request, headers):
        data = await request.json()

        # Parse size - supports both aspect ratio ("16:9") and dimensions ("1024x1024")
        size = data.get('size', '1:1')
        if ':' in size:
            # Aspect ratio format (e.g., "16:9")
            width, height = _aspect_ratio_to_dimensions(size)
            original_size = size
        elif 'x' in size:
            # Dimensional format (e.g., "1024x1024")
            try:
                width, height = map(int, size.split('x'))
            except (ValueError, TypeError):
                width, height = 1024, 1024
            original_size = f"{width}x{height}"
            width, height = _clamp_dimensions(width, height)
        else:
            # Default to square
            width, height = 1024, 1024
            original_size = "1:1"

        body = {
            "prompt": data.get('prompt'),
            "width": width,
            "height": height,
            "prompt_upsampling": True
        }

        logger.info(f"FLUX image generation: endpoint={cls.API_ENDPOINT}, requested={original_size}, actual={width}x{height}")

        async with httpx.AsyncClient(base_url=BASE_URL, http1=True, http2=False) as client:
            url = httpx.URL(path=cls.API_ENDPOINT, query=request.url.query.encode("utf-8"))

            req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=json.dumps(body),
                timeout=cls.TIMEOUT,
            )

            try:
                r = await client.send(req, stream=False)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                error_info = (
                    f"{type(e)}: {e} | "
                    f"Please check if host={request.client.host} can access [{BASE_URL}] successfully?"
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

            if r.status_code != 200:
                logger.exception(f"{r.status_code}:{r.json()}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error from Flux call: {r.status_code}"
                )

            response_data = r.json()
            polling_url = response_data.get('polling_url')

            if polling_url is None:
                logger.exception("No polling_url returned from FLUX")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"No polling_url returned"
                )

            return polling_url

    @classmethod
    async def _poll_for_result(cls, polling_url, headers):
        timeout, start_time = 240, time.time()
        task_not_found_count = 0
        max_task_not_found_retries = 5

        logger.info(f"FLUX polling: {polling_url}")

        # Create the AsyncClient once for the entire polling duration
        async with httpx.AsyncClient(http1=True, http2=False) as client:
            while True:
                req = client.build_request(
                    "GET",
                    polling_url,
                    headers=headers,
                    timeout=cls.TIMEOUT,
                )

                try:
                    r = await client.send(req, stream=False)

                    if r.status_code == status.HTTP_200_OK:
                        json_response = r.json()

                        if json_response.get('status') == "Ready":
                            return (
                                json_response.get('result').get('sample'),
                                json_response.get('result').get('prompt')
                            )
                        elif json_response.get('status') == "Content Moderated":
                            raise ContentModerationError("Moderated")
                        elif json_response.get('status') == "Task not found":
                            task_not_found_count += 1
                            if task_not_found_count > max_task_not_found_retries:
                                logger.error(f"FLUX task not found after {max_task_not_found_retries} retries, giving up")
                                raise HTTPException(
                                    status_code=status.HTTP_502_BAD_GATEWAY,
                                    detail=f"Task not found after {max_task_not_found_retries} retries"
                                )
                            logger.info(f"FLUX poll status: Task not found ({task_not_found_count}/{max_task_not_found_retries}), sleeping 5s before retry")
                            await asyncio.sleep(5)
                        elif json_response.get('status') == "Pending":
                            logger.info(f"FLUX poll status: Pending, sleeping 5s before retry")
                            await asyncio.sleep(5)
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_418_IM_A_TEAPOT,
                                detail=f"API Error: {json_response}"
                            )
                    elif r.status_code == status.HTTP_202_ACCEPTED:
                        await asyncio.sleep(5)
                    elif time.time() - start_time > timeout:
                        logger.exception("Polling timed out")
                        raise HTTPException(
                            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail="Polling timed out"
                        )
                    else:
                        # Check if it's a "Task not found" response (may come as 404)
                        try:
                            json_response = r.json()
                            if json_response.get('status') == "Task not found":
                                task_not_found_count += 1
                                if task_not_found_count > max_task_not_found_retries:
                                    logger.error(f"FLUX task not found after {max_task_not_found_retries} retries, giving up")
                                    raise HTTPException(
                                        status_code=status.HTTP_502_BAD_GATEWAY,
                                        detail=f"Task not found after {max_task_not_found_retries} retries"
                                    )
                                logger.info(f"FLUX poll status: Task not found (HTTP {r.status_code}) ({task_not_found_count}/{max_task_not_found_retries}), sleeping 5s before retry")
                                await asyncio.sleep(5)
                                continue
                        except Exception:
                            pass
                        logger.exception(f"API Error:{r.json()}")
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"API Error: {r.json()}"
                        )
                finally:
                    # Close the response explicitly after each use
                    await r.aclose()

    @classmethod
    async def _stream_image(cls, image_url, prompt):
        async with httpx.AsyncClient(http1=True, http2=False) as client:
            async with client.stream("GET", image_url) as response:
                if response.status_code == 200:
                    # Start the JSON response and the base64 field
                    yield JSON_PREFIX

                    # Stream the image data in chunks and encode each in base64
                    buffer = b""

                    async for chunk in response.aiter_bytes():
                        buffer += chunk

                        # Process full 3-byte chunks from the buffer
                        while len(buffer) >= 3:
                            to_encode, buffer = buffer[:3], buffer[3:]
                            encoded_chunk = base64.b64encode(to_encode).decode('utf-8')
                            yield encoded_chunk

                    # Encode any remaining data in the buffer, including necessary padding
                    if buffer:
                        yield base64.b64encode(buffer).decode('utf-8')

                    yield JSON_MID
                    yield prompt
                    yield JSON_SUFFIX

                else:
                    logger.exception(f"Failed to download image")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Failed to download image",
                    )

    @classmethod
    async def _image_size(cls, image_url, prompt):
        # Fetch the original content length from the image's HTTP headers
        response = await httpx.AsyncClient().head(image_url)
        original_content_length = response.headers.get("content-length")
        if original_content_length is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to determine original Content-Length for encoding"
            )

        original_content_length = int(original_content_length)

        # Calculate the base64 content length, explicitly accounting for padding
        full_base64_groups = original_content_length // 3
        leftover_bytes = original_content_length % 3
        base64_content_length = 4 * full_base64_groups + (4 if leftover_bytes > 0 else 0)

        # Calculate total content length by adding JSON parts
        total_content_length = (
                len(JSON_PREFIX.encode("utf-8"))
                + base64_content_length
                + len(JSON_MID.encode("utf-8"))
                + len(prompt.encode("utf-8"))
                + len(JSON_SUFFIX.encode("utf-8"))
        )
        return total_content_length

class FluxPro11(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.1"
    ACCEPT = "image/*"


class FluxKontextGen(FluxBase):
    API_ENDPOINT = "v1/flux-kontext-pro"
    ACCEPT = "application/json"

    @classmethod
    async def _image_request(cls, request: Request, headers):
        data = await request.json()

        # Parse size - supports both aspect ratio ("16:9") and dimensions ("1024x1024")
        size = data.get('size', '1:1')
        if ':' in size:
            # Aspect ratio format - pass through directly
            aspect_ratio = size
        elif 'x' in size:
            # Dimensional format - convert to aspect ratio
            try:
                width, height = map(int, size.split('x'))
            except (ValueError, TypeError):
                width, height = 1024, 1024
            divisor = gcd(width, height)
            aspect_ratio = f"{width // divisor}:{height // divisor}"
        else:
            # Default to square
            aspect_ratio = "1:1"

        logger.info(f"FLUX Kontext image generation: endpoint={cls.API_ENDPOINT}, requested={size}, aspect_ratio={aspect_ratio}")

        body = {
            "prompt": data.get('prompt'),
            "aspect_ratio": aspect_ratio,
            "prompt_upsampling": True
        }

        async with httpx.AsyncClient(base_url=BASE_URL, http1=True, http2=False) as client:
            url = httpx.URL(path=cls.API_ENDPOINT, query=request.url.query.encode("utf-8"))

            req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=json.dumps(body),
                timeout=cls.TIMEOUT,
            )

            try:
                r = await client.send(req, stream=False)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                error_info = (
                    f"{type(e)}: {e} | "
                    f"Please check if host={request.client.host} can access [{BASE_URL}] successfully?"
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

            if r.status_code != 200:
                logger.exception(f"{r.status_code}:{r.json()}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error from Flux Kontext call: {r.status_code}"
                )

            response_data = r.json()
            polling_url = response_data.get('polling_url')

            if polling_url is None:
                logger.exception("No polling_url returned from FLUX Kontext")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"No polling_url returned"
                )

            return polling_url


class FluxKontext(FluxBase):
    API_ENDPOINT = "v1/flux-kontext-pro"
    ACCEPT = "application/json"

    @classmethod
    async def generate_image(cls, request: Request):
        headers = {
            "Accept": cls.ACCEPT,
            "x-key": BFL_API_KEY,
            "Content-Type": "application/json"
        }

        if headers["x-key"] is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid configuration.",
            )

        polling_url = await cls._image_request(request, headers)
        image_url, prompt = await cls._poll_for_result(polling_url, headers)

        # Escape special characters in prompt
        prompt = json.dumps(prompt)

        return (
            cls._stream_image(image_url, prompt),
            await cls._image_size(image_url, prompt)
        )

    @classmethod
    async def _image_request(cls, request: Request, headers):
        # Parse multipart form data from OpenAI request
        form_data = await request.form()
        
        # Extract image file from form data
        image_file = form_data.get('image')
        if not image_file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required 'image' parameter"
            )
        
        # Read image data and convert to base64
        image_bytes = await image_file.read()
        input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Extract prompt from form data
        prompt = form_data.get('prompt', '')
        if isinstance(prompt, bytes):
            prompt = prompt.decode('utf-8')

        body = {
            "prompt": prompt,
            "input_image": input_image_b64,
            "aspect_ratio": None,
            "output_format": "png"
        }

        # Handle optional parameters from form data
        if 'seed' in form_data:
            try:
                body['seed'] = int(form_data['seed'])
            except (ValueError, TypeError):
                pass  # Ignore invalid seed values
                
        if 'safety_tolerance' in form_data:
            try:
                body['safety_tolerance'] = int(form_data['safety_tolerance'])
            except (ValueError, TypeError):
                pass  # Ignore invalid safety_tolerance values

        async with httpx.AsyncClient(base_url=BASE_URL, http1=True, http2=False) as client:
            url = httpx.URL(path=cls.API_ENDPOINT, query=request.url.query.encode("utf-8"))

            req = client.build_request(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body),
                timeout=cls.TIMEOUT,
            )

            try:
                r = await client.send(req, stream=False)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                error_info = (
                    f"{type(e)}: {e} | "
                    f"Please check if host={request.client.host} can access [{BASE_URL}] successfully?"
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

            if r.status_code != 200:
                logger.exception(f"{r.status_code}:{r.json()}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, 
                    detail=f"Error from Flux Kontext call: {r.status_code}"
                )

            response_data = r.json()
            polling_url = response_data.get('polling_url')

            if polling_url is None:
                logger.exception("No polling_url returned from FLUX Kontext")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"No polling_url returned from Flux Kontext"
                )

            return polling_url


class ContentModerationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message