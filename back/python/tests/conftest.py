import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup_config import DEFAULT_CONFIG  # noqa: E402


@pytest.fixture
def config():
    return copy.deepcopy(DEFAULT_CONFIG)
