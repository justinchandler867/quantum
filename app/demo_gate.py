"""
Demo deployment gate: HTTP Basic Auth over every route, a noindex header on
every response, and the DEMO_MODE flag the frontend reads.

Env vars are read per request (not at import) so the process behaves the same
whether they were set before or after import, and tests can toggle them.

  DEMO_PASSWORD  non-empty -> Basic Auth required (username "demo") on every
                 path except /health, which Render's health check must reach.
                 unset/empty -> gate inactive; localhost dev unchanged.
  DEMO_MODE      truthy -> frontend hides the Trade Desk. Unset, empty, or one
                 of 0/false/no/off -> off.
"""

import base64
import binascii
import os
import secrets

from starlette.requests import Request
from starlette.responses import Response

DEMO_USERNAME = "demo"
_FALSY = {"", "0", "false", "no", "off"}


def demo_password() -> str:
    return os.environ.get("DEMO_PASSWORD", "")


def demo_mode() -> bool:
    return os.environ.get("DEMO_MODE", "").strip().lower() not in _FALSY


def _credentials_ok(auth_header: str, password: str) -> bool:
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "basic" or not token:
        return False
    try:
        decoded = base64.b64decode(token.strip(), validate=True)
    except (binascii.Error, ValueError):
        return False
    user, sep, pw = decoded.partition(b":")
    if not sep:
        return False
    # Evaluate both comparisons (no short-circuit) so timing doesn't reveal
    # which half was wrong.
    user_ok = secrets.compare_digest(user, DEMO_USERNAME.encode())
    pw_ok = secrets.compare_digest(pw, password.encode("utf-8"))
    return user_ok & pw_ok


async def demo_gate_middleware(request: Request, call_next):
    password = demo_password()
    if password and request.url.path != "/health" and not _credentials_ok(request.headers.get("authorization", ""), password):
        response = Response(
            "Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Quantex demo", charset="UTF-8"'},
        )
    else:
        response = await call_next(request)
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response
