# Pilot results (raw)

Human-readable view of [`results/raw/wm_c0.json`](raw/wm_c0.json) and
[`results/raw/llm_conflict_prompt.json`](raw/llm_conflict_prompt.json) — those JSON files
are the single source of truth; the figure and the tests derive from them.

Hand-graded, single-step, free-form accessibility-tree predictions on synthetic
interfaces. **Directional only** (n≤15). Conflict = remapped label contradicts true
behavior; honest = label matches behavior (control). `conf` is each model's own
confidence (self-reported for LLM conditions; geometric-mean token probability for
WebWorld-8B — the two are *not* comparable across families and are never pooled).

## WM-C0 — stock WebWorld-8B (Qwen3-8B based, no fine-tuning), 6 blocks

Run on an **independent 6-block subset** (4 conflict + 2 honest). Block indices are *not*
matched to the 15-block LLM set below; the two runs are shown in separate figure panels and
never pooled.

| Block | Tier | Action (true effect) | Predicted | conf | ✓/✗ |
|-------|------|----------------------|-----------|------|-----|
| 05 | conflict | "Cancel" (places order) | "Order canceled" | 0.83 | ✗ |
| 08 | conflict | "Cancel" (records survey) | navigated to main page, not recorded | 0.82 | ✗ |
| 10 | conflict | "Cancel" (subscribes) | "subscription cancelled" | 0.80 | ✗ |
| 11 | conflict | "Cancel" (deletes task) | task "Buy milk" still present | 0.90 | ✗ |
| 12 | honest | "Delete" (removes email) | "Inbox (2)", removed | 0.85 | ✓ |
| 15 | honest | "Delete account" (deletes) | "Confirm account deletion" dialog | 0.88 | ✗ (safe direction) |

Conflict accuracy 0/4. Confidence of incorrect (0.85 mean) ≈ confidence of correct (0.85).
Reasoning on 05: *"clicking Cancel would typically cancel the order"* — label prior, not dynamics.

## WM-B + conflict-detection prompt — frontier LLM, 15 blocks

Explicit reasoning step: compare label's usual meaning vs retrieved behavior; on
conflict, predict the behavior and set `label_conflict=True` + low confidence.

| Block | Tier | conf | pred ✓/✗ | flag correct? | note |
|-------|------|------|----------|---------------|------|
| 01 | conflict | 0.35 | ✓ | ✓ | |
| 02 | conflict | 0.35 | ✓ | ✓ | |
| 03 | conflict | 0.35 | ✗ | ✗ miss | Back→advance |
| 04 | conflict | 0.35 | ✗ | ✗ miss | Save→Delete relabel |
| 05 | conflict | 0.35 | ✓ | ✓ | |
| 06 | conflict | 0.35 | ✓ | ✓ | |
| 07 | conflict | 0.72 | ✗ | ✗ miss | **confidently wrong** |
| 08 | conflict | 0.35 | ✗ | flagged, then predicted label anyway | detect-but-don't-act |
| 09 | conflict | 0.45 | ✗ | ✗ miss | Back→advance |
| 10 | conflict | 0.35 | ✗ | flagged, then predicted label anyway | detect-but-don't-act |
| 11 | conflict | 0.55 | ✗ | ✗ miss | |
| 12 | honest | 0.75 | ✓ | ✓ | |
| 13 | honest | 0.75 | ✓ | ✓ | |
| 14 | conflict | 0.40 | ✓ | ✗ miss | pred right anyway |
| 15 | honest | 0.35 | ✓ | ✗ false-flag | flagged conflict on an honest delete |

**Overall** prediction accuracy 8/15 (01,02,05,06,12,13,14,15); **conflict-only** accuracy
5/12 (01,02,05,06,14). Flag accuracy 8/15; **missed 6/12 real conflicts** and
**false-flagged the one honest irreversible delete (15)**. Confidence of incorrect
(0.45 mean) ≈ confidence of correct (0.46): no separation. The two highest-confidence
honest cases (12,13 at 0.75) are correct, but a wrong conflict case (07) sits at 0.72 — i.e.
confidence reflects "did I detect a conflict," not "am I right."

## Prompt-variant progression (LLM, conflict-case accuracy)

| Prompt | Conflict accuracy | Side effect |
|--------|-------------------|-------------|
| zero-shot (WM-A) | ~lowest | fooled on conflicts |
| + correct retrieval (WM-B) | partial | closed some conflicts |
| + blunt "distrust labels" | higher on conflicts | **broke honest cases** (predicted "settings saved" on real *Delete account*) |
| + structured conflict reasoning | ~plateau (5/12 conflict, 8/15 overall) | recovered honest cases, lost conflict ones |

Accuracy is prompt-tunable to a ceiling and then only **redistributes** failures; better
prompting does not buy calibrated conflict-detection.
