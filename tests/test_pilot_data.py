"""Validate that the raw pilot data reproduces the headline numbers in REPORT.md.

These tests are the contract between results/raw/*.json and the claims in the
report and figure: if anyone edits the raw data in a way that breaks a stated
result, a test fails. They depend only on the standard library, so they run even
without the modeling harness installed (the wmeval logic tests live under
src/wmeval/ and run in the same `pytest` invocation once that package is present).
"""
import json
import os
from statistics import mean

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")


def load(name):
    with open(os.path.join(RAW, name), encoding="utf-8") as fh:
        return json.load(fh)["records"]


def r2(x):
    return round(x, 2)


# ---- WM-C0 (stock WebWorld-8B) -------------------------------------------------

def test_wm_c0_size_and_split():
    recs = load("wm_c0.json")
    assert len(recs) == 6
    assert sum(r["tier"] == "conflict" for r in recs) == 4
    assert sum(r["tier"] == "honest" for r in recs) == 2


def test_wm_c0_zero_of_four_conflicts():
    recs = load("wm_c0.json")
    conflicts = [r for r in recs if r["tier"] == "conflict"]
    assert sum(r["correct"] for r in conflicts) == 0  # 0/4


def test_wm_c0_confidence_does_not_separate():
    recs = load("wm_c0.json")
    incorrect = [r["confidence"] for r in recs if not r["correct"]]
    correct = [r["confidence"] for r in recs if r["correct"]]
    # Report: WebWorld-8B 0.85 (incorrect) vs 0.85 (correct)
    assert r2(mean(incorrect)) == 0.85
    assert r2(mean(correct)) == 0.85


def test_wm_c0_conflict_confidence_band():
    recs = load("wm_c0.json")
    conf = [r["confidence"] for r in recs if r["tier"] == "conflict"]
    # Report: every conflict wrong at 0.80-0.90 confidence
    assert min(conf) >= 0.80 and max(conf) <= 0.90


def test_wm_c0_two_highest_are_wrong():
    recs = sorted(load("wm_c0.json"), key=lambda r: r["confidence"], reverse=True)
    assert recs[0]["confidence"] == 0.90 and recs[0]["correct"] is False
    assert recs[1]["confidence"] == 0.88 and recs[1]["correct"] is False


# ---- LLM conflict-prompt -------------------------------------------------------

def test_llm_size():
    recs = load("llm_conflict_prompt.json")
    assert len(recs) == 15
    assert sum(r["tier"] == "conflict" for r in recs) == 12
    assert sum(r["tier"] == "honest" for r in recs) == 3


def test_llm_overall_and_conflict_accuracy():
    recs = load("llm_conflict_prompt.json")
    # Report §3.1 (fixed): overall 8/15, conflict-only 5/12
    assert sum(r["correct"] for r in recs) == 8
    conflicts = [r for r in recs if r["tier"] == "conflict"]
    assert sum(r["correct"] for r in conflicts) == 5


def test_llm_confidence_does_not_separate():
    recs = load("llm_conflict_prompt.json")
    incorrect = [r["confidence"] for r in recs if not r["correct"]]
    correct = [r["confidence"] for r in recs if r["correct"]]
    # Report: LLM 0.45 (incorrect) vs 0.46 (correct)
    assert r2(mean(incorrect)) == 0.45
    assert r2(mean(correct)) == 0.46


def test_llm_flag_misses_and_false_flag():
    recs = load("llm_conflict_prompt.json")
    conflicts = [r for r in recs if r["tier"] == "conflict"]
    honest = [r for r in recs if r["tier"] == "honest"]
    # Report: flag missed 6/12 real conflicts; false-flagged the 1 honest delete
    assert sum(not r["flag_correct"] for r in conflicts) == 6
    assert sum(not r["flag_correct"] for r in honest) == 1


def test_detect_but_dont_act_cases_present():
    recs = {r["block"]: r for r in load("llm_conflict_prompt.json")}
    # Blocks 08 and 10: flag fired correctly, prediction still wrong
    for b in ("08", "10"):
        assert recs[b]["flag_correct"] is True
        assert recs[b]["correct"] is False
