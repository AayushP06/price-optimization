"""Entry point for the Flask backend.

Run from the project root:
    python backend/app/main.py

Or from the backend/ directory:
    python app/main.py
"""

import os
import sys

# Ensure `backend/` is on sys.path so `from app.*` imports resolve correctly
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app import create_app
from app.core.config import Config

app = create_app()

if __name__ == "__main__":
    port = Config.PORT
    print("\n" + "=" * 60)
    print("  PRICE OPTIMIZATION API SERVER")
    print("=" * 60)
    print("\nAvailable Endpoints:")
    print("  POST /api/optimize        - Run complete optimization")
    print("  POST /api/generate-sample - Generate sample competitor data")
    print("  POST /api/sort            - Sort prices using Quick Sort")
    print("  POST /api/search          - Search using Binary Search")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/history         - Optimization history")
    print("  GET  /api/stats           - Statistics")
    print(f"\nStarting server on http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=Config.DEBUG)
