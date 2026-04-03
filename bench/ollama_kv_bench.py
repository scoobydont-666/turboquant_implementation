#!/usr/bin/env python3
"""Benchmark Ollama KV cache modes: q8_0 vs q4_0.

Measures VRAM, throughput (tok/s), time-to-first-token, and quality
on a standardized 50-prompt eval set.

Usage:
    python ollama_kv_bench.py --host giga --model qwen3:14b
    python ollama_kv_bench.py --host giga --model qwen3:14b --kv-type q4_0
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

EVAL_PROMPTS = Path(__file__).parent / "eval_prompts.jsonl"

HOST_MAP = {
    "giga": "192.168.200.163",
    "mega": "192.168.200.141",
    "mecha": "192.168.200.87",
}


def resolve_host(host: str) -> str:
    return HOST_MAP.get(host.lower(), host)


def get_vram_mb(host: str) -> float:
    """Get VRAM usage via nvidia-smi over SSH."""
    ip = resolve_host(host)
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", ip,
             "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        # Sum all GPUs
        return sum(float(line.strip()) for line in result.stdout.strip().split("\n") if line.strip())
    except Exception as e:
        print(f"  WARNING: nvidia-smi failed: {e}")
        return 0.0


def get_ollama_ps(host: str) -> list[dict]:
    """Get loaded models from Ollama."""
    ip = resolve_host(host)
    try:
        resp = httpx.get(f"http://{ip}:11434/api/ps", timeout=10)
        return resp.json().get("models", [])
    except Exception:
        return []


def generate(host: str, model: str, prompt: str, num_predict: int = 256) -> dict:
    """Run a single generation and return timing metrics."""
    ip = resolve_host(host)
    url = f"http://{ip}:11434/api/generate"

    start = time.time()
    first_token_time = None
    tokens = 0
    response_text = ""

    try:
        with httpx.stream("POST", url, json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": num_predict, "temperature": 0},
        }, timeout=120) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    if first_token_time is None:
                        first_token_time = time.time()
                    response_text += data["response"]
                    tokens += 1
                if data.get("done"):
                    break
    except Exception as e:
        return {"error": str(e), "text": "", "tokens": 0, "ttft": 0, "tok_per_sec": 0}

    elapsed = time.time() - start
    ttft = (first_token_time - start) if first_token_time else elapsed

    return {
        "text": response_text,
        "tokens": tokens,
        "elapsed_s": round(elapsed, 2),
        "ttft_s": round(ttft, 3),
        "tok_per_sec": round(tokens / elapsed, 1) if elapsed > 0 else 0,
    }


def check_quality(response: str, expected_pattern: str) -> bool:
    """Check if response matches expected pattern (regex)."""
    try:
        return bool(re.search(expected_pattern, response, re.IGNORECASE))
    except re.error:
        return expected_pattern.lower() in response.lower()


def load_prompts() -> list[dict]:
    """Load eval prompts from JSONL file."""
    prompts = []
    with open(EVAL_PROMPTS) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def run_benchmark(host: str, model: str, kv_type: str, prompts: list[dict]) -> dict:
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {model} on {host} with KV cache: {kv_type}")
    print(f"{'='*60}")

    # Pre-benchmark VRAM
    vram_before = get_vram_mb(host)
    print(f"  VRAM before: {vram_before:.0f} MB")

    # Ensure model is loaded
    print(f"  Loading model...")
    gen = generate(host, model, "Hello", num_predict=1)
    if gen.get("error"):
        print(f"  ERROR: {gen['error']}")
        return {"error": gen["error"]}

    vram_after = get_vram_mb(host)
    print(f"  VRAM after load: {vram_after:.0f} MB (delta: {vram_after - vram_before:.0f} MB)")

    # Check Ollama model info
    models = get_ollama_ps(host)
    for m in models:
        if model in m.get("name", ""):
            vram_gb = m.get("size_vram", 0) / 1e9
            print(f"  Model VRAM (Ollama): {vram_gb:.1f} GB")

    # Run eval prompts
    results = []
    correct = 0
    total_tokens = 0
    total_time = 0
    ttfts = []

    for i, p in enumerate(prompts):
        r = generate(host, model, p["prompt"], num_predict=256)
        if r.get("error"):
            print(f"  [{i+1}/{len(prompts)}] ERROR: {r['error']}")
            continue

        match = check_quality(r["text"], p["expected_pattern"])
        if match:
            correct += 1
        total_tokens += r["tokens"]
        total_time += r["elapsed_s"]
        ttfts.append(r["ttft_s"])

        status = "PASS" if match else "FAIL"
        print(f"  [{i+1}/{len(prompts)}] {status} ({r['tok_per_sec']} tok/s, TTFT {r['ttft_s']}s) {p['id']}")

        results.append({
            "id": p["id"],
            "category": p["category"],
            "match": match,
            "tokens": r["tokens"],
            "tok_per_sec": r["tok_per_sec"],
            "ttft_s": r["ttft_s"],
        })

    avg_tok_s = round(total_tokens / total_time, 1) if total_time > 0 else 0
    avg_ttft = round(sum(ttfts) / len(ttfts), 3) if ttfts else 0
    quality_pct = round(correct / len(prompts) * 100, 1) if prompts else 0

    summary = {
        "host": host,
        "model": model,
        "kv_type": kv_type,
        "vram_mb": vram_after,
        "vram_delta_mb": round(vram_after - vram_before),
        "prompts": len(prompts),
        "correct": correct,
        "quality_pct": quality_pct,
        "avg_tok_per_sec": avg_tok_s,
        "avg_ttft_s": avg_ttft,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 1),
        "results": results,
    }

    print(f"\n  SUMMARY: {quality_pct}% quality, {avg_tok_s} tok/s avg, {avg_ttft}s avg TTFT, {vram_after:.0f} MB VRAM")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama KV cache modes")
    parser.add_argument("--host", default="giga", help="Ollama host (giga/mega/mecha or IP)")
    parser.add_argument("--model", default="qwen3:14b", help="Model name")
    parser.add_argument("--kv-type", default="q8_0", help="KV cache type (q8_0, q4_0, f16)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} eval prompts")
    print(f"NOTE: KV cache type is set via OLLAMA_KV_CACHE_TYPE env var on the host.")
    print(f"      Current run assumes host is configured with: {args.kv_type}")

    summary = run_benchmark(args.host, args.model, args.kv_type, prompts)

    out_path = args.output or f"bench_ollama_{args.kv_type}_{args.host}.json"
    with open(Path(__file__).parent / out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
