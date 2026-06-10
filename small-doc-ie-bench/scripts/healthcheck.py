from __future__ import annotations

import sys
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/healthz"
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        sys.exit(0 if 200 <= response.status < 300 else 1)
except Exception:
    sys.exit(1)
