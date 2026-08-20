import subprocess
import time

import httpx
from sparrow.multiprocess import kill
from utils import rm


class TestRun:
    @classmethod
    def setup_class(cls):
        kill(8000)
        base_url = "https://api.openai-forward.com"
        subprocess.Popen(["nohup", "openai-forward", "run", "--base_url", base_url])
        time.sleep(3)

    @classmethod
    def teardown_class(cls):
        kill(8000)
        rm("nohup.out")

    def test_get_doc(self):
        resp = httpx.get("http://localhost:8000/docs")
        assert resp.is_success

    def test_get_chat_completions(self):
        # 403, not 401: every proxied request must carry an X-Request-Signature HMAC, and
        # validate_request rejects the ones that do not before any auth handling happens.
        # The 401 this used to assert predates that gate.
        resp = httpx.get("http://localhost:8000/v1/chat/completions")
        assert resp.status_code == 403
