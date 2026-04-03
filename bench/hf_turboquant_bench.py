#!/usr/bin/env python3
"""Benchmark TurboQuant KV cache compression on HuggingFace models.

Compares FP16 baseline vs TurboQuant 3-bit (Paper) vs TQ 4-bit (MSE-only)
on quality, VRAM, and throughput.

Usage:
    python hf_turboquant_bench.py --model Qwen/Qwen2.5-14B-Instruct
    python hf_turboquant_bench.py --model meta-llama/Llama-3.1-8B-Instruct

Requires: torch, transformers, scipy (and a GPU with sufficient VRAM)
"""

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add turboquant to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from turboquant import (
    TurboQuantGenerationCache,
    TurboQuantPaperCache,
    patch_model_for_paper_generation,
    unpatch_model_for_paper_generation,
)

EVAL_PROMPTS = Path(__file__).parent / "eval_prompts.jsonl"


def load_prompts() -> list[dict]:
    prompts = []
    with open(EVAL_PROMPTS) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def check_quality(response: str, expected_pattern: str) -> bool:
    try:
        return bool(re.search(expected_pattern, response, re.IGNORECASE))
    except re.error:
        return expected_pattern.lower() in response.lower()


def reset_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    cache=None,
) -> dict:
    """Run a single generation and return metrics."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    reset_vram()
    start = time.time()

    with torch.no_grad():
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if cache is not None:
            kwargs["past_key_values"] = cache

        outputs = model.generate(**inputs, **kwargs)

    elapsed = time.time() - start
    vram = get_vram_mb()

    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "text": response.strip(),
        "new_tokens": new_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_per_sec": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
        "vram_mb": round(vram, 1),
    }


def benchmark_config(
    model,
    tokenizer,
    config_name: str,
    cache_factory,
    prompts: list[dict],
    patched: bool = False,
) -> dict:
    """Run benchmark for a single configuration."""
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    if patched and not getattr(model, "_turboquant_patched", False):
        patch_model_for_paper_generation(model)
        model._turboquant_patched = True

    correct = 0
    total_tokens = 0
    total_time = 0
    vram_samples = []
    results = []

    for i, p in enumerate(prompts):
        cache = cache_factory() if cache_factory else None
        r = run_generation(model, tokenizer, p["prompt"], max_new_tokens=256, cache=cache)

        match = check_quality(r["text"], p["expected_pattern"])
        if match:
            correct += 1
        total_tokens += r["new_tokens"]
        total_time += r["elapsed_s"]
        vram_samples.append(r["vram_mb"])

        status = "PASS" if match else "FAIL"
        print(f"  [{i+1}/{len(prompts)}] {status} ({r['tok_per_sec']} tok/s, {r['vram_mb']:.0f}MB) {p['id']}")

        results.append({
            "id": p["id"],
            "match": match,
            "tokens": r["new_tokens"],
            "tok_per_sec": r["tok_per_sec"],
            "vram_mb": r["vram_mb"],
        })

    avg_tok_s = round(total_tokens / total_time, 1) if total_time > 0 else 0
    quality_pct = round(correct / len(prompts) * 100, 1) if prompts else 0
    avg_vram = round(sum(vram_samples) / len(vram_samples), 1) if vram_samples else 0
    peak_vram = max(vram_samples) if vram_samples else 0

    summary = {
        "config": config_name,
        "quality_pct": quality_pct,
        "correct": correct,
        "total_prompts": len(prompts),
        "avg_tok_per_sec": avg_tok_s,
        "avg_vram_mb": avg_vram,
        "peak_vram_mb": round(peak_vram, 1),
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
        "results": results,
    }

    print(f"\n  SUMMARY: {quality_pct}% quality, {avg_tok_s} tok/s, {avg_vram:.0f}MB avg VRAM, {peak_vram:.0f}MB peak")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant KV cache")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4], help="Bit widths to test")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 baseline")
    parser.add_argument("--max-prompts", type=int, default=50, help="Max prompts to eval")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device
    )

    # Suppress generation config warnings
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    prompts = load_prompts()[:args.max_prompts]
    print(f"Loaded {len(prompts)} eval prompts")

    all_results = {}

    # FP16 baseline (no cache quantization)
    if not args.skip_fp16:
        all_results["fp16"] = benchmark_config(
            model, tokenizer, "FP16 Baseline", None, prompts, patched=False
        )

    # Patch model for TurboQuant
    patch_model_for_paper_generation(model)
    model._turboquant_patched = True

    # TurboQuant configs
    for bits in args.bits:
        if bits <= 3:
            config_name = f"TQ {bits}-bit (Paper: MSE+QJL)"
            factory = lambda b=bits: TurboQuantPaperCache.from_model_config(model.config, bits=b)
        else:
            config_name = f"TQ {bits}-bit (MSE-only)"
            factory = lambda b=bits: TurboQuantGenerationCache.from_model_config(model.config, bits=b)

        all_results[f"tq_{bits}bit"] = benchmark_config(
            model, tokenizer, config_name, factory, prompts, patched=True
        )

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Config':<25} {'Quality':>8} {'tok/s':>8} {'VRAM(MB)':>10} {'Peak(MB)':>10}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        print(f"{r['config']:<25} {r['quality_pct']:>7.1f}% {r['avg_tok_per_sec']:>7.1f} {r['avg_vram_mb']:>9.0f} {r['peak_vram_mb']:>9.0f}")
    print(f"{'='*70}")

    # Save results
    out_path = args.output or f"bench_turboquant_{args.model.split('/')[-1]}.json"
    with open(Path(__file__).parent / out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
