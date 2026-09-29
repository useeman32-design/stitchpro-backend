"""
cPanel "Setup Python App" entry point. Passenger looks for a module-level
`application` WSGI callable in this exact file at the application root.

On cPanel, set:
  Application startup file: passenger_wsgi.py
  Application Entry point:  application
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.app import app as application  # noqa: E402  (Passenger requires this name)
