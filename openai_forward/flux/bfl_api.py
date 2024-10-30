import httpx
from fastapi import HTTPException, Request, status

from loguru import logger
import json
import base64
import os
import time

BASE_URL = "https://api.bfl.ml/"
BFL_API_KEY = os.environ.get("BFL_API_KEY", None)

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
        image_base64 = await cls._download_image(image_url)
        return (
            image_base64,
            prompt
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

        client = httpx.AsyncClient(base_url=BASE_URL, http1=True, http2=False)
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
        while True:
            client = httpx.AsyncClient(base_url=BASE_URL, http1=True, http2=False)
            url = httpx.URL(path=cls.POLL_ENDPOINT, query=f"id={id}".encode("utf-8"))

            req = client.build_request(
                "GET",
                url,
                headers=headers,
                timeout=cls.TIMEOUT,
            )

            try:
                r = await client.send(req, stream=False)
            except:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error during polling (TODO - more detail here)"
                )

            if r.status_code == status.HTTP_200_OK:
                json = r.json()

                if json.get('status') == "Ready":
                    return (
                        json.get('result').get('sample'),
                        json.get('result').get('prompt')
                    )
                elif json.get('status') == "Pending":
                    time.sleep(10)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY, detail=f"API Error: {json}"
                    )
            elif r.status_code == status.HTTP_202_ACCEPTED:
                time.sleep(10)
            elif time.time() - start_time > timeout:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"Poll did not succeed"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=f"API Error: {r.json()}"
                )

    @classmethod
    async def _download_image(cls, imageUrl):
        r = httpx.get(imageUrl)
        if r.status_code == status.HTTP_200_OK:
            image_bytes = r.content
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            return image_base64

        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Unable to download image {imageUrl} {r.status_code}"
            )

class FluxPro11(FluxBase):
    API_ENDPOINT = "v1/flux-pro-1.1"
    POLL_ENDPOINT = "v1/get_result"
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "prompt": ("STRING", {"multiline": True}),
        },
        "optional": {
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967294}),
            "guidance": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 5, "step": 0.01}),
            "width": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "height": (
                "INT",
                {"default": 1024, "min": 0, "max": 1440, "step": 32},
            ),
            "interval": ("INT", {"default": 1, "min": 1, "max": 10}),
            "prompt_upsampling": (
                "BOOLEAN",
                {"default": True, "label_on": "True", "label_off": "False"},
            ),
            "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            "api_key_override": ("STRING", {"multiline": False}),
        },
    }


NODE_CLASS_MAPPINGS = {
    "FLUX 1.1 [pro]": FluxPro11,
}