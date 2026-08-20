import shutil
import subprocess
import time

import httpx
import pytest
from sparrow.multiprocess import kill
from utils import rm

# These are integration tests: setup_class launches the real server and the tests talk to it
# over localhost. In a checkout without `pip install -e .` the console script does not exist,
# Popen raises FileNotFoundError, nothing binds port 8000, and both tests fail with connection
# refused -- on every commit, which teaches everyone to ignore a red suite. Skip instead, so a
# failure here means something is actually broken.
CLI = shutil.which("openai-forward")

pytestmark = pytest.mark.skipif(
    CLI is None,
    reason="openai-forward CLI not installed (`pip install -e .`); skipping server integration tests",
)


class TestRun:
    @classmethod
    def setup_class(cls):
        kill(8000)
        base_url = "https://api.openai-forward.com"
        subprocess.Popen(["nohup", CLI, "run", "--base_url", base_url])
        time.sleep(3)

    @classmethod
    def teardown_class(cls):
        kill(8000)
        rm("nohup.out")

    def test_get_doc(self):
        resp = httpx.get("http://localhost:8000/docs")
        assert resp.is_success

    def test_get_chat_completions(self):
        resp = httpx.get("http://localhost:8000/v1/chat/completions")
        assert resp.status_code == 401
