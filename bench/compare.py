#!/usr/bin/env python3
"""Compare benchmark results across Ollama KV modes and TurboQuant configs.

Reads bench_*.json files and produces a decision matrix in Markdown.

Usage:
    python compare.py
"""

import json
from pathlib import Path

BENCH_DIR = Path(__file__).parent


def load_results() -> dict:
    results = {}
    for f in sorted(BENCH_DIR.glob("bench_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        # Handle both ollama (single dict) and turboquant (nested dict) formats
        if "kv_type" in data:
            key = f"ollama_{data['kv_type']}"
            results[key] = data
        else:
            for config_key, config_data in data.items():
                results[config_key] = config_data
    return results


def main():
    results = load_results()
    if not results:
        print("No benchmark results found. Run ollama_kv_bench.py and hf_turboquant_bench.py first.")
        return

    print("# TurboQuant Benchmark Results\n")
    print(f"| Config | Quality | tok/s | VRAM (MB) | Infra Cost |")
    print(f"|--------|---------|-------|-----------|------------|")

    for key, r in results.items():
        name = r.get("config", r.get("kv_type", key))
        quality = r.get("quality_pct", 0)
        tok_s = r.get("avg_tok_per_sec", r.get("avg_tok_per_sec", 0))
        vram = r.get("peak_vram_mb", r.get("vram_mb", 0))

        if "ollama" in key:
            cost = "zero (current)" if "q8_0" in key else "1-line config"
        elif "fp16" in key:
            cost = "reference only"
        else:
            cost = "sidecar server"

        print(f"| {name:<20} | {quality:>6.1f}% | {tok_s:>5.1f} | {vram:>9.0f} | {cost} |")

    # Decision
    print("\n## Decision")
    ollama_q8 = results.get("ollama_q8_0", {})
    ollama_q4 = results.get("ollama_q4_0", {})
    tq_3 = results.get("tq_3bit", {})

    if ollama_q4:
        q8_quality = ollama_q8.get("quality_pct", 0)
        q4_quality = ollama_q4.get("quality_pct", 0)
        delta = abs(q8_quality - q4_quality)
        if delta <= 3.0:  # Within 3 percentage points
            print(f"- Ollama q4_0 quality ({q4_quality}%) is within {delta:.1f}pp of q8_0 ({q8_quality}%)")
            print(f"- **RECOMMENDATION: Deploy q4_0 via Ansible. TurboQuant sidecar not needed.**")
        else:
            print(f"- Ollama q4_0 quality drop is {delta:.1f}pp — significant")
            if tq_3:
                tq_quality = tq_3.get("quality_pct", 0)
                tq_vram = tq_3.get("peak_vram_mb", 0)
                q4_vram = ollama_q4.get("vram_mb", 0)
                print(f"- TurboQuant 3-bit quality: {tq_quality}%, VRAM: {tq_vram:.0f}MB")
                if tq_quality > q4_quality:
                    print(f"- **RECOMMENDATION: Proceed to Phase 2 — TurboQuant server.**")
                else:
                    print(f"- **RECOMMENDATION: TurboQuant quality not better. Stay on q8_0.**")


if __name__ == "__main__":
    main()
