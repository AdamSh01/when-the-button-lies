# When the Button Lies 🪤

**Can a world model tell a web agent that an action is unsafe — *before* it acts?**

A pre-registered pilot on whether LLMs and trained world models can flag **label–behavior conflicts** (a "Cancel" button that actually places the order) before irreversible web-agent actions. Short answer, in this pilot: **no — and they fail confidently.**

> ⚠️ **Scope up front (read this):** this is a **pilot** — n ≤ 15, single-step, hand-graded, synthetic interfaces. The finding is **directional, not definitive.** It is shared for transparency, method reuse, and as a cautionary signal — not as a finished study, and **not as a product.** See [§5 of the report](REPORT.md#5-what-would-make-this-conclusive) for exactly what would make it conclusive.

**What this is, in one line:** a reproducible *negative pilot* showing that current web world models can be confidently wrong about irreversible UI actions when the labels lie.

## The finding

A model-based web agent simulates "what happens if I click this?" before committing. For that to make it *safer*, the simulator must be **calibrated** — unsure when it's about to be wrong about an irreversible action. We tested that with interfaces whose labels contradict their behavior. In this pilot:

- A zero-shot LLM, a retrieval-augmented LLM, an explicit conflict-detection prompt, and a **trained open-weights world model (WebWorld-8B)** all **confidently mispredict** label–behavior conflicts.
- **Confidence does not track correctness** — incorrect predictions are not less confident than correct ones (see figure).
- WebWorld-8B got **every** conflict case wrong, at **0.80–0.90** confidence, reasoning *"clicking Cancel would typically cancel the order"* — and it is not a weak model (it scores near frontier on its own WebWorld-Bench; see [REPORT §3.2](REPORT.md#32-a-trained-world-model-is-fooled-the-same-way--confidently)).
- The conflict-detection prompt sometimes **flags the conflict and then predicts the label outcome anyway** — it writes the right reasoning and doesn't act on it.

![Confidence does not separate correct from incorrect predictions](results/calibration.png)

The failures are in the dangerous direction: the model confidently predicts *"nothing committed"* at the exact moment an irreversible action commits.

## What's in here

| Path | What |
|------|------|
| [`REPORT.md`](REPORT.md) | The full write-up: question, method, results, honest limitations, path to conclusive |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Hypotheses, conditions, endpoint, and decision rule — fixed before results were seen |
| [`src/wmeval/`](src/wmeval) | Evaluation harness (conditions, adversarial remap, calibration metrics, paired stats) with known-answer tests |
| [`scripts/run_webworld.py`](scripts/run_webworld.py) | Reproduces the WM-C0 run: WebWorld-8B in its native format, with geom-mean-token-prob confidence |
| [`results/raw/`](results/raw) | Raw per-block run records (JSON) — the single source of truth |
| [`results/`](results) | Figure + human-readable per-block results |
| [`make_figure.py`](make_figure.py) | Regenerates the headline figure from `results/raw/` |
| [`tests/`](tests) | Data-invariant tests tying the raw results to every number in the report |

## Reproduce

**Install (harness + tests + figure):**
```bash
pip install -e ".[dev]"
make test         # data-invariant tests + the wmeval known-answer tests
make reproduce    # regenerate results/calibration.png from results/raw/
```

**The trained-world-model run (WebWorld-8B)** — fits a single 16 GB GPU (free Colab T4) in 8-bit:
```bash
pip install -e ".[webworld]"
python scripts/run_webworld.py        # bundled demo transition; plug in the full set for all of §3.2
```
The script uses WebWorld's documented prompt format (system prompt + `Initial Page State: / First Action: / Next Page State:`, actions as `click([id])`, A11y-tree states) and greedy decoding, exactly as the [model card](https://huggingface.co/Qwen/WebWorld-8B) specifies. (The card loads in bf16, which OOMs a T4; 8-bit is why this ran on free hardware — pass `--bf16` on a larger GPU.)

## Method in one paragraph

Take honest `(state, action, next_state)` transitions; remap the *surface labels* of the observation so they contradict the true behavior (swap "Submit"↔"Cancel", "Save"↔"Delete"), leaving the recorded outcome unchanged. This dissociates **label priors** ("Cancel cancels") from **learned dynamics** (what the interface actually does). Score each condition on whether its **confidence separates right from wrong** — the property safe action-validation needs. No label leakage; retrieval never returns a transition for its own prediction. Full protocol in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Status & contributions

A **negative pilot**, deliberately. Its reusable parts:
1. the **adversarial-remap methodology** for probing whether a world model relies on labels or on dynamics;
2. a **pre-registered, reproducible harness** with calibration + paired-stats metrics, and raw data wired to the figure and tests;
3. a documented, one-directional **cautionary finding** that off-the-shelf prompting *and* a current trained web world model both fail calibrated conflict-detection.

Extensions welcome — the [path to a conclusive study](REPORT.md#5-what-would-make-this-conclusive) is spelled out.

---

*Model under test: `Qwen/WebWorld-8B` (Xiao et al., 2026, arXiv:2602.14721; Qwen3-8B base, Apache-2.0). This repository: Apache-2.0.*
