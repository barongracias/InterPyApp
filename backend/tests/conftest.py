import os
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True, scope="session")
def ensure_backend_dirs():
    """Ensure backend directories exist for tests that rely on them."""
    for dirname in ["uploads", "outputs_numpy", "outputs_tf"]:
        path = os.path.join(BASE_DIR, dirname)
        os.makedirs(path, exist_ok=True)
    # Default CORS for tests
    os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
    yield
