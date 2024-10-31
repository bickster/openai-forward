import httpx
from fastapi import HTTPException, Request, status

from loguru import logger
import json
import base64
import os
import time

BASE_URL = "https://api.bfl.ml/"
BFL_API_KEY = os.environ.get("BFL_API_KEY", None)

JSON_PREFIX = f'''{{
    "created": {int(time.time())},
    "data": [
        {{
            "b64_json": "'''
JSON_MID = '", "revised_prompt": "'
JSON_SUFFIX = f'''"
        }}
    ]
}}
'''

class FluxBase:
    API_ENDPOINT = ""
    POLL_ENDPOINT = ""
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

        image_id = await cls._image_request(request, headers)

        image_url, prompt = await cls._poll_for_result(image_id, headers)

        return (
            cls._stream_image(image_url, prompt),
            await cls._image_size(image_url, prompt)
        )

    @classmethod
    async def _image_request(cls, request: Request, headers):
        data = await request.json()
        body = {
            "prompt": data.get('prompt'),
            "width": 1024,
            "height": 1024,
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
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error from Flux call: {r.status_code}"
                )

            imageId = r.json().get('id')
            if imageId is None:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"No id returned"
                )

            return imageId

    @classmethod
    async def _poll_for_result(cls, id, headers):
        timeout, start_time = 240, time.time()

        # Create the AsyncClient once for the entire polling duration
        async with httpx.AsyncClient(base_url=BASE_URL, http1=True, http2=False) as client:
            while True:
                url = httpx.URL(path=cls.POLL_ENDPOINT, query=f"id={id}".encode("utf-8"))

                req = client.build_request(
                    "GET",
                    url,
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
                        elif json_response.get('status') == "Pending":
                            time.sleep(5)  # Use async sleep to avoid blocking
                        else:
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"API Error: {json_response}"
                            )
                    elif r.status_code == status.HTTP_202_ACCEPTED:
                        time.sleep(5)  # Use async sleep
                    elif time.time() - start_time > timeout:
                        raise HTTPException(
                            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                            detail="Polling timed out"
                        )
                    else:
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
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
