import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import copy
from itertools import cycle

import pytest


# OpenaiBase keeps its configuration in mutable class attributes, and several tests
# rewrite them in place. tests/test_api.py in particular blanks the api key list and
# replaces _cycle_api_key with cycle([]) in its teardown -- after which any later test
# that reaches to_openai hits next() on an exhausted cycle, and the StopIteration
# surfaces as "RuntimeError: coroutine raised StopIteration". The image tests only fail
# when they run after test_api.py, which is exactly the order pytest collects them in.
_MUTATED_BY_TESTS = (
    "IP_WHITELIST",
    "IP_BLACKLIST",
    "UA_WHITELIST",
    "UA_BLACKLIST",
    "APP_SECRET",
    "_openai_api_key_list",
    "_FWD_KEYS",
)


@pytest.fixture(autouse=True)
def restore_openai_base_state():
    """Snapshot the class attributes tests mutate, and put them back afterwards."""
    from openai_forward.openai import OpenaiBase

    # copy, don't just hold the reference: tests mutate these lists in place
    # (IP_BLACKLIST.append(...)), so putting the same object back restores nothing.
    saved = {name: copy.copy(getattr(OpenaiBase, name)) for name in _MUTATED_BY_TESTS}
    yield
    for name, value in saved.items():
        setattr(OpenaiBase, name, value)
    # A cycle cannot be rewound, so rebuild it from the restored key list rather than
    # putting back a partially consumed one.
    OpenaiBase._cycle_api_key = cycle(OpenaiBase._openai_api_key_list)
    OpenaiBase._compile_ua_patterns()
