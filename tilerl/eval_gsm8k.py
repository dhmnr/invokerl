"""GSM8K evaluation harness for Qwen3 — pure cuTile + CuPy.

Evaluates a Qwen3 model on the GSM8K math reasoning benchmark with greedy decoding.
Supports NumPy CPU fallback for hardware with CuPy compatibility issues (e.g. sm_120).

Usage:
    # Full test set (1319 problems), NumPy CPU mode:
    TILERL_BACKEND=numpy python eval_gsm8k.py

    # Quick 50-sample eval:
    TILERL_BACKEND=numpy python eval_gsm8k.py --num_samples 50

    # Resume interrupted eval:
    TILERL_BACKEND=numpy python eval_gsm8k.py --resume results/gsm8k_0.6b_50.jsonl

    # CuPy GPU mode (A100/H100):
    python eval_gsm8k.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

# Backend selection — mirrors qwen3.py pattern
_BACKEND = os.environ.get("TILERL_BACKEND", "").lower()

if _BACKEND == "numpy":
    import numpy as cp  # type: ignore[assignment]
else:
    try:
        import cupy as cp
    except ModuleNotFoundError:
        import numpy as cp  # type: ignore[assignment]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3 import Qwen3Config, Qwen3Model
from inference import load_hf_weights


# ---------------------------------------------------------------------------
# GSM8K data loading
# ---------------------------------------------------------------------------

def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test split.

    Returns list of {"question": str, "answer": str, "gold": str}.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")

    data = []
    for ex in ds:
        gold = _extract_gold(ex["answer"])
        data.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "gold": gold,
        })
    return data


def _extract_gold(answer_text: str) -> str:
    """Extract the numeric answer after #### from GSM8K answer text."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


# ---------------------------------------------------------------------------
# Answer extraction from model output
# ---------------------------------------------------------------------------

def extract_model_answer(response: str) -> str:
    """Extract the final numeric answer from model's generated response.

    Looks for (in priority order):
    1. #### <number>
    2. \\boxed{<number>}
    3. "the answer is <number>"
    4. Last number in the response
    """
    # #### pattern (trained models)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # \\boxed{} pattern
    match = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # "the answer is X" pattern
    match = re.search(r"[Tt]he\s+answer\s+is\s+\$?(-?[\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Last number in the response
    numbers = re.findall(r"-?\d+\.?\d*", response)
    if numbers:
        return numbers[-1].strip()

    return ""


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer (numeric comparison)."""
    try:
        pred_val = float(predicted.replace(",", ""))
        gold_val = float(gold.replace(",", ""))
        return abs(pred_val - gold_val) < 1e-5
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(question: str) -> str:
    """Format a GSM8K question as a Qwen3 chat prompt."""
    return (
        "<|im_start|>system\n"
        "You are a helpful math assistant. Solve the problem step by step. "
        "End your response with #### followed by the numeric answer.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: Qwen3Model,
    tokenizer,
    test_data: list[dict],
    max_new_tokens: int = 512,
    output_path: str | None = None,
    completed_indices: set[int] | None = None,
) -> dict:
    """Run GSM8K evaluation, one problem at a time.

    Saves results incrementally to output_path (JSONL) for resume capability.

    Returns:
        {"accuracy": float, "correct": int, "total": int}
    """
    completed_indices = completed_indices or set()
    correct = 0
    total = 0
    start_time = time.time()

    out_file = open(output_path, "a") if output_path else None

    try:
        for idx, ex in enumerate(test_data):
            if idx in completed_indices:
                continue

            prompt = format_prompt(ex["question"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

            t0 = time.time()
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                eos_token_id=151645,  # <|im_end|>
            )
            gen_time = time.time() - t0

            # Decode generated tokens (skip prompt)
            gen_ids = output_ids[len(prompt_ids):]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predicted = extract_model_answer(response)
            is_correct = check_answer(predicted, ex["gold"])

            correct += int(is_correct)
            total += 1
            elapsed = time.time() - start_time

            result = {
                "index": idx,
                "question": ex["question"],
                "gold": ex["gold"],
                "predicted": predicted,
                "correct": is_correct,
                "response": response,
                "gen_tokens": len(gen_ids),
                "gen_time_s": round(gen_time, 2),
            }

            # Incremental save
            if out_file:
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()

            # Progress
            acc_so_far = correct / total
            n_gen = len(gen_ids)
            tok_s = n_gen / max(gen_time, 0.001)
            remaining = len(test_data) - len(completed_indices) - total
            eta_s = (elapsed / total) * remaining if total > 0 else 0

            mark = "+" if is_correct else "-"
            print(
                f"[{total:>4d}/{len(test_data) - len(completed_indices)}] "
                f"{mark} gold={ex['gold']:>8s} pred={predicted:>8s}  "
                f"acc={acc_so_far:.1%}  "
                f"{n_gen} tok in {gen_time:.1f}s ({tok_s:.1f} t/s)  "
                f"ETA {_fmt_time(eta_s)}",
                flush=True,
            )

    except KeyboardInterrupt:
        print(f"\nInterrupted after {total} problems.")

    finally:
        if out_file:
            out_file.close()

    accuracy = correct / max(total, 1)
    return {"accuracy": accuracy, "correct": correct, "total": total}


def _fmt_time(seconds: float) -> str:
    """Format seconds as h:mm:ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def load_completed(path: str) -> tuple[set[int], int, int]:
    """Load completed results from a JSONL file for resume.

    Returns (completed_indices, correct_count, total_count).
    """
    completed: set[int] = set()
    correct = 0
    total = 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                completed.add(rec["index"])
                correct += int(rec["correct"])
                total += 1
    return completed, correct, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 on GSM8K (greedy decoding)"
    )
    parser.add_argument(
        "--model_repo", type=str, default="Qwen/Qwen3-0.6B",
        help="HuggingFace model repo ID",
    )
    parser.add_argument(
        "--model_size", type=str, default="0.6b",
        choices=["0.6b", "1.7b", "4b"],
    )
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help="Number of test samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max tokens to generate per problem",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing JSONL results file to resume from",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint dir (with model.npz) to evaluate instead of HF weights",
    )
    args = parser.parse_args()

    backend_name = "NumPy CPU" if cp is np else "CuPy GPU"
    print(f"=== GSM8K Evaluation ({backend_name}) ===\n")

    # --- Model ---
    print("[1/4] Initializing model...")
    model_configs = {
        "0.6b": Qwen3Config.qwen3_0_6b,
        "1.7b": Qwen3Config.qwen3_1_7b,
        "4b": Qwen3Config.qwen3_4b,
    }
    cfg = model_configs[args.model_size]()
    model = Qwen3Model(cfg)
    n_params = sum(p.size for p in model.params.values())
    print(f"  Qwen3-{args.model_size}: {n_params:,} params")

    # --- Weights ---
    print("[2/4] Loading weights...")
    if args.checkpoint:
        # Load from training checkpoint (.npz)
        npz_path = os.path.join(args.checkpoint, "model.npz")
        data = np.load(npz_path)
        for k in data.files:
            if k in model.params:
                model.params[k] = cp.asarray(data[k])
        print(f"  Loaded checkpoint from {args.checkpoint}")
        # Still need tokenizer from HF
        local_dir = None
    else:
        local_dir = load_hf_weights(model, args.model_repo)

    # --- Tokenizer ---
    print("[3/4] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer_source = local_dir if local_dir else args.model_repo
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    print(f"  {tokenizer.__class__.__name__}, vocab={len(tokenizer)}")

    # --- Data ---
    print("[4/4] Loading GSM8K test set...")
    test_data = load_gsm8k_test()
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    print(f"  {len(test_data)} problems\n")

    # --- Resume ---
    os.makedirs(args.output_dir, exist_ok=True)
    if args.checkpoint:
        ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
        default_output = os.path.join(
            args.output_dir,
            f"gsm8k_{args.model_size}_{ckpt_name}_{len(test_data)}.jsonl",
        )
    else:
        default_output = os.path.join(
            args.output_dir,
            f"gsm8k_{args.model_size}_{len(test_data)}.jsonl",
        )
    output_path = args.resume or default_output
    completed_indices, prev_correct, prev_total = set(), 0, 0
    if args.resume or os.path.exists(output_path):
        completed_indices, prev_correct, prev_total = load_completed(output_path)
        if completed_indices:
            print(
                f"  Resuming: {prev_total} already done "
                f"({prev_correct}/{prev_total} = "
                f"{prev_correct / max(prev_total, 1):.1%})"
            )
            remaining = len(test_data) - len(completed_indices)
            print(f"  Remaining: {remaining} problems\n")

    # --- Run ---
    print(f"Saving results to: {output_path}\n")
    t0 = time.time()
    result = evaluate(
        model, tokenizer, test_data,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
        completed_indices=completed_indices,
    )
    total_time = time.time() - t0

    # Combine with resumed results
    total_correct = prev_correct + result["correct"]
    total_done = prev_total + result["total"]
    total_acc = total_correct / max(total_done, 1)

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"GSM8K Evaluation Complete")
    print(f"{'=' * 50}")
    print(f"  Model:    Qwen3-{args.model_size}")
    print(f"  Backend:  {backend_name}")
    print(f"  Problems: {total_done}/{len(test_data)}")
    print(f"  Correct:  {total_correct}")
    print(f"  Accuracy: {total_acc:.2%}")
    print(f"  Time:     {_fmt_time(total_time)} (this session)")
    print(f"  Results:  {output_path}")

    # Save summary JSON
    summary_path = output_path.replace(".jsonl", "_summary.json")
    summary = {
        "model": f"Qwen3-{args.model_size}",
        "model_repo": args.model_repo,
        "backend": backend_name,
        "total_problems": len(test_data),
        "evaluated": total_done,
        "correct": total_correct,
        "accuracy": round(total_acc, 4),
        "max_new_tokens": args.max_new_tokens,
        "session_time_s": round(total_time, 1),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:  {summary_path}")


if __name__ == "__main__":
    main()
