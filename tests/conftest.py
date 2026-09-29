"""Test-session environment: no background network refresh, no snapshot load at
app startup (tests that need the snapshot loader call it directly)."""
import os

os.environ.setdefault("QX_LIVE_REFRESH", "0")
os.environ.setdefault("QX_SNAPSHOT_LOAD", "0")
