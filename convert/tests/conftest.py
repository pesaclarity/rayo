import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

RAYO = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def lib():
    """rayo's C library, built from this checkout."""
    from rayo_convert.runtime import Library

    if not (shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")):
        pytest.skip("no C compiler")
    return Library.build(RAYO)
