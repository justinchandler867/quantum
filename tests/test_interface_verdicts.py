"""
INTERFACE_VERDICTS_SPEC.md Acceptance item 4 — tests for the SIGNAL→SHARPE(1Y)
and AI ADVISER→Ask Quantex remediation.

(a) verdict-vocabulary absence in the Sharpe (1Y) column output
(b) Ask Quantex chip text matches the replacement set
(c) the AI system prompt contains the educational guardrail

These are display/copy assertions, so the frontend ones read the single-file
frontend as text (there is no build step to hook into — see CLAUDE.md).
"""
import os
import re

import pytest

from app.main import _sharpe_band

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FRONTEND = os.path.join(ROOT, "static", "quantex.html")
MAIN_PY = os.path.join(ROOT, "app", "main.py")

BANDS = {"High", "Moderate", "Low", "Minimal", "Negative"}
GUARDRAIL_MARKERS = [
    "educational explainer for Quantex",
    "never recommend buying, selling, holding",
    "not a financial adviser",
]


@pytest.fixture(scope="module")
def frontend():
    with open(FRONTEND, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def main_src():
    with open(MAIN_PY, encoding="utf-8") as fh:
        return fh.read()


# ── (a) closed-set Sharpe bands, no verdict vocabulary ────────────────────────
@pytest.mark.parametrize("sharpe,expected", [
    (2.0, "High"), (1.5, "High"),
    (1.49, "Moderate"), (1.0, "Moderate"),
    (0.99, "Low"), (0.5, "Low"),
    (0.49, "Minimal"), (0.0, "Minimal"),
    (-0.01, "Negative"), (-3.0, "Negative"),
])
def test_sharpe_band_thresholds(sharpe, expected):
    assert _sharpe_band(sharpe) == expected


def test_sharpe_band_only_closed_set():
    for i in range(-500, 500):
        assert _sharpe_band(i / 100.0) in BANDS


def test_no_verdict_vocabulary_in_band_output():
    banned = {"buy", "sell", "hold", "strong", "accumulate", "avoid", "weak"}
    for i in range(-500, 500):
        label = _sharpe_band(i / 100.0).lower()
        assert not (banned & set(label.split())), label


def test_backend_sig_producer_uses_band_not_verdict(main_src):
    # The /api/ticker/add producer must assign the band helper, not a verdict string.
    assert "sig = _sharpe_band(sharpe)" in main_src
    assert 'sig = "Strong Buy"' not in main_src
    assert 'sig = "Buy"' not in main_src
    assert 'sig = "Hold"' not in main_src
    assert 'sig = "Sell"' not in main_src


def test_frontend_signal_column_renamed_and_no_sig_literals(frontend):
    assert "SHARPE (1Y)" in frontend
    # no verdict labels remain as Discovery signal literals
    assert not re.search(r'sig:"(Strong Buy|Buy|Hold|Sell)"', frontend)
    # the sharpe cell helper is present
    assert "function sharpeBand(" in frontend
    assert "function sharpeCell(" in frontend


# ── (b) Ask Quantex chips ─────────────────────────────────────────────────────
EXPECTED_CHIPS = [
    "Walk me through my portfolio's key metrics",
    "Explain my Sharpe ratio",
    "How does hedging work for a portfolio like this?",
    "How does screen match rank assets for my profile?",
    "What does rebalancing do, and what are its tradeoffs?",
]
RETIRED_CHIPS = [
    "What should I hedge?",
    "Best assets for my profile?",
    "Should I rebalance?",
]


def test_chips_match_replacement_set(frontend):
    m = re.search(r"const QUICK=\[(.*?)\];", frontend)
    assert m, "QUICK chip array not found"
    chip_blob = m.group(1)
    for chip in EXPECTED_CHIPS:
        assert chip in chip_blob, f"missing chip: {chip}"


def test_retired_advice_chips_absent(frontend):
    m = re.search(r"const QUICK=\[(.*?)\];", frontend)
    chip_blob = m.group(1)
    for chip in RETIRED_CHIPS:
        assert chip not in chip_blob, f"advice-shaped chip still present: {chip}"


def test_panel_renamed_no_adviser_word(frontend):
    assert '"Ask Quantex"' in frontend
    assert "AI ADVISER" not in frontend
    # regulated word must not appear in user-facing panel/login copy
    assert "AI Advisory" not in frontend


# ── (c) system-prompt guardrail ───────────────────────────────────────────────
def test_backend_system_prompt_has_guardrail(main_src):
    for marker in GUARDRAIL_MARKERS:
        assert marker in main_src, f"backend system prompt missing: {marker}"
    assert "professional financial adviser assistant" not in main_src


def test_frontend_system_prompt_has_guardrail(frontend):
    for marker in GUARDRAIL_MARKERS:
        assert marker in frontend, f"frontend sysPrompt missing: {marker}"
    assert "professional financial adviser assistant" not in frontend
