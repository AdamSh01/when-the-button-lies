"""Reproduce the WM-C0 condition: stock WebWorld-8B on adversarial label-remap transitions.

This is the script behind the §3.2 results. It uses WebWorld's documented I/O format
(system prompt + "Initial Page State: / First Action: / Next Page State:", actions as
`click([id])` function calls, A11y-tree states) and greedy decoding, exactly as the
model card specifies, and computes a per-prediction confidence as the geometric mean of
the generated tokens' probabilities (i.e. exp(mean(token log-probs))) — the WM-C0
confidence reported in the paper figure.

Model card: https://huggingface.co/Qwen/WebWorld-8B  (Xiao et al., 2026, arXiv:2602.14721)

Hardware: loads in 8-bit (bitsandbytes) so the 8B model fits a single 16 GB GPU, e.g. a
free Colab T4. (The card's snippet loads in bf16, which needs ~16 GB+ and will OOM a T4 —
8-bit is the reason this pilot ran on free hardware. Disable with --bf16 on a larger GPU.)

Install:  pip install -e ".[webworld]"
Run:      python scripts/run_webworld.py            # runs the bundled demo transitions
"""
import argparse
import math

MODEL_NAME = "Qwen/WebWorld-8B"

SYSTEM_PROMPT = (
    "You are a web world model. I will provide you with an initial page state "
    "and a sequence of actions. For each action, predict the resulting page state.\n"
    "Strictly maintain the original format. Output only the full page state "
    "without explanations, code, or truncation."
)


def build_user_message(state: str, action: str) -> str:
    """WebWorld single-step prompt. `action` is a function call, e.g. \"click([12])\"."""
    return f"Initial Page State:\n{state}\n\nFirst Action: '{action}'\n\nNext Page State:"


def remap_labels(state: str, swaps: dict[str, str]) -> str:
    """Adversarial surface-label remap.

    Swap the *names* of controls on the page (e.g. {'Submit': 'Cancel', 'Cancel': 'Submit'})
    while the recorded true outcome of clicking the button is left unchanged elsewhere.
    Bidirectional and applied atomically so a single pass cannot double-swap.
    """
    import re
    # Build a single alternation; replace each match by its mapped value in one pass.
    pattern = re.compile("|".join(re.escape(k) for k in sorted(swaps, key=len, reverse=True)))
    return pattern.sub(lambda m: swaps[m.group(0)], state)


# --- Bundled demo: one conflict transition (block 05). Replace `TRANSITIONS` with the
#     full adversarial set to reproduce all of §3.2. `true_next_state` is the recorded
#     ground truth AFTER the action; the label remap does not touch it. ---------------
_CHECKOUT = """RootWebArea 'Checkout — Review Order', focused
\t[1] main 'Order Review', visible
\t\t[2] StaticText 'Total: $84.00'
\t\t[3] button 'Submit', clickable, visible
\t\t[4] button 'Cancel', clickable, visible"""

TRANSITIONS = [
    {
        "block": "05",
        "tier": "conflict",
        # remap swaps the two button names: the button that *commits* the order is now
        # labelled 'Cancel'; clicking it still places the order.
        "state": remap_labels(_CHECKOUT, {"Submit": "Cancel", "Cancel": "Submit"}),
        "action": "click([4])",                       # the button now NAMED 'Submit'
        "true_effect": "places the order",
        "true_next_state": "RootWebArea 'Order Confirmed — Thank you', focused",
    },
]


def geometric_mean_token_prob(scores, chosen_ids, log_softmax) -> float:
    """exp(mean(log p(token))) over the generated tokens = geometric-mean token prob."""
    logps = []
    for step_logits, tok_id in zip(scores, chosen_ids):
        logp = log_softmax(step_logits[0], dim=-1)[tok_id].item()
        logps.append(logp)
    if not logps:
        return float("nan")
    return math.exp(sum(logps) / len(logps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", action="store_true",
                    help="load in bf16 instead of 8-bit (needs a >=16 GB GPU)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    if args.bf16:
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs).eval()

    for t in TRANSITIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(t["state"], t["action"])},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,                 # greedy, per the model card
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][inputs["input_ids"].shape[-1]:]
        prediction = tok.decode(gen_ids, skip_special_tokens=True).strip()
        conf = geometric_mean_token_prob(out.scores, gen_ids.tolist(), torch.log_softmax)

        print("=" * 72)
        print(f"block {t['block']} [{t['tier']}]  action {t['action']}  "
              f"(true effect: {t['true_effect']})")
        print(f"confidence (geom-mean token prob): {conf:.2f}")
        print(f"predicted next state:\n{prediction}")
        print(f"\nground-truth next state:\n{t['true_next_state']}")
        print("Grade by eye: does the prediction reflect the *true effect*, "
              "or the misleading label?")


if __name__ == "__main__":
    main()
