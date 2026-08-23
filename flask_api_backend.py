"""
Shim module so code can import `flask_api_backend` even though the main file
is named `flask-api-backend.py` (hyphen is not importable). This keeps
`check_database.py` and other helpers working without renaming the original
entry point.
"""

import importlib.util
import pathlib
import sys


def _load_main_module():
    """Load the real Flask app module from `flask-api-backend.py`."""
    here = pathlib.Path(__file__).resolve()
    target = here.with_name("flask-api-backend.py")

    spec = importlib.util.spec_from_file_location("flask_api_backend_impl", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find flask-api-backend.py at {target}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["flask_api_backend_impl"] = module  # allow relative imports
    spec.loader.exec_module(module)
    return module


_impl = _load_main_module()

# Re-export the app components used elsewhere
app = _impl.app
db = _impl.db
Optimization = _impl.Optimization

# Optional: allow `from flask_api_backend import *`
__all__ = ["app", "db", "Optimization"]
