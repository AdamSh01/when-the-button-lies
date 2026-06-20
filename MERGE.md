# Merge notes (read once, then delete)

This bundle is everything I could finalize from the files you shared, with all three
reviews folded in. To make the public repo `AdamSh01/when-the-button-lies` complete,
there is exactly **one thing only you can do** — the rest is drop-in.

## The one action only you can do

The public repo is missing two things that live in your local/canonical copy (all three
reviewers flagged this — it is the single biggest issue, and it is just a `git add`):

```bash
git add src/wmeval/ PREREGISTRATION.md
```

- `src/wmeval/` — your harness with the 42 known-answer tests.
- `PREREGISTRATION.md` — the pre-registered hypotheses, conditions, endpoint, decision rule.

The README, REPORT, `pyproject.toml`, `Makefile`, and `tests/` all reference these paths.
Until they are committed, those links 404 and the "pre-registered / reproducible" claims
have nothing behind them.

## Drop-in (everything else)

These files overwrite or add to the repo — no decisions needed:

```
README.md                      (replaces — new file map, make-reproduce flow, positioning, license)
REPORT.md                      (replaces — §3.1 denominator fix, model-card citations, strong-model framing, BibTeX)
make_figure.py                 (replaces — now loads from results/raw/ instead of hardcoding)
results/pilot_results.md       (replaces — subset note + overall-vs-conflict accuracy split)
results/raw/wm_c0.json         (new — source of truth, WebWorld-8B run)
results/raw/llm_conflict_prompt.json  (new — source of truth, LLM run)
results/calibration.png        (regenerated from raw; identical figure)
scripts/run_webworld.py        (new — reproduces the WM-C0 run in WebWorld's native format)
tests/test_pilot_data.py       (new — ties raw data to every headline number in the report)
pyproject.toml                 (new — src-layout, [dev] + [webworld] extras matching the README)
Makefile                       (new — make reproduce / make test)
```

One reconciliation note: if your local repo already has a working `pyproject.toml` with the
42-test setup, keep yours — just make sure it declares the `[dev]` extra with `matplotlib`
and `pytest` so `pip install -e ".[dev]"` and `make reproduce` work on a clean clone. Don't
keep two.

## Verify before pushing (clean clone, fresh venv)

```bash
pip install -e ".[dev]"
make test          # green
make reproduce     # regenerates results/calibration.png
```

LICENSE: keep your existing Apache-2.0 file (it's correct). Set the GitHub "About" to:
> Pre-registered pilot: can LLMs and trained world models flag label–behavior conflicts before irreversible web-agent actions? In this pilot, no — and they fail confidently.
