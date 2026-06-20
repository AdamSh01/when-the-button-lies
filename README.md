# When the Button Lies 🪤

**Can a world model tell a web agent that an action is unsafe — *before* it acts?**

A pre-registered pilot on whether LLMs and trained world models can flag **label–behavior conflicts** (a "Cancel" button that actually places the order) before irreversible web-agent actions. Short answer, in this pilot: **no — and they fail confidently.**

> ⚠️ **Scope up front (read this):** this is a **pilot** — n ≤ 15, single-step, hand-graded, synthetic interfaces. The finding is **directional, not definitive.** It is shared for transparency, method reuse, and as a cautionary signal — not as a finished study. See [§5 of the report](REPORT.md#5-what-would-make-this-conclusive) for exactly what would make it conclusive.

## The finding

A model-based web agent simulates "what happens if I click this?" before committing. For that to make it *safer*, the simulator must be **calibrated** — unsure when it's about to be wrong about an irreversible action. We tested that with interfaces whose labels contradict their behavior. In this pilot:

- A zero-shot LLM, a retrieval-augmented LLM, an explicit conflict-detection prompt, and a **trained open-weights world model (WebWorld-8B)** all **confidently mispredict** label–behavior conflicts.
- **Confidence does not track correctness** — incorrect predictions are not less confident than correct ones (see figure).
- WebWorld-8B got **every** conflict case wrong, at **0.80–0.90** confidence, reasoning *"clicking Cancel would typically cancel the order."*
- The conflict-detection prompt sometimes **flags the conflict and then predicts the label outcome anyway** — it writes the right reasoning and doesn't act on it.

![Confidence does not separate correct from incorrect predictions](results/calibration.png)

The failures are in the dangerous direction: the model confidently predicts *"nothing committed"* at the exact moment an irreversible action commits.

## What's in here

| Path | What |
|------|------|
| [`REPORT.md`](REPORT.md) | The full write-up: question, method, results, honest limitations, path to conclusive |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Hypotheses, conditions, endpoint, and decision rule — fixed before results were seen |
| [`src/wmeval/`](src/wmeval) | Evaluation harness (conditions, adversarial remap, calibration metrics, paired stats) with known-answer tests |
| [`make_figure.py`](make_figure.py) | Regenerates the headline figure from the run data |
| [`results/`](results) | Figure + per-block pilot results |

## Reproduce

**Harness + tests:**
```bash
pip install -e ".[dev]"
pytest                       # known-answer + smoke tests
```

**The trained-world-model run (WebWorld-8B)** — fits a single 16 GB GPU (free Colab T4) in 8-bit:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
tok = AutoTokenizer.from_pretrained("Qwen/WebWorld-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/WebWorld-8B", device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    trust_remote_code=True).eval()
# feed remapped (state, action) in WebWorld's a11y-tree + function-call format; see REPORT §2.1
```

**Figure:**
```bash
python make_figure.py        # -> results/calibration.png
```

## Method in one paragraph

Take honest `(state, action, next_state)` transitions; remap the *surface labels* of the observation so they contradict the true behavior (swap "Submit"↔"Cancel", "Save"↔"Delete"), leaving the recorded outcome unchanged. This dissociates **label priors** ("Cancel cancels") from **learned dynamics** (what the interface actually does). Score each condition on whether its **confidence separates right from wrong** — the property safe action-validation needs. No label leakage; retrieval never returns a transition for its own prediction. Full protocol in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Status & contributions

A **negative pilot**, deliberately. Its reusable parts:
1. the **adversarial-remap methodology** for probing whether a world model relies on labels or on dynamics;
2. a **pre-registered, reproducible harness** with calibration + paired-stats metrics;
3. a documented, one-directional **cautionary finding** that off-the-shelf prompting *and* a current trained web world model both fail calibrated conflict-detection.

Extensions welcome — the [path to a conclusive study](REPORT.md#5-what-would-make-this-conclusive) is spelled out.

---

*Model under test: `Qwen/WebWorld-8B` (Xiao et al., 2026, arXiv:2602.14721). License: see repo.*
