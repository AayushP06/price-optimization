"""Entry point for the Flask backend.

Run from the project root:
    python backend/app/main.py

Or from the backend/ directory:
    python app/main.py
"""

import os
import sys
import logging


_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app import create_app
from app.api.core.config import Config

app = create_app()

if __name__ == "__main__":
    port = Config.PORT
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Starting price optimization server on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=Config.DEBUG)
