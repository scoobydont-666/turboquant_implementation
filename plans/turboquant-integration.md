# TurboQuant KV Cache Quantization — Integration Plan

## Context

Josh found kumar045's TurboQuant implementation (Google ICLR 2026 paper) — PyTorch-native KV cache compression to 3-4 bits using vector quantization. Repo cloned to `/opt/turboquant/`, forked to GitHub.

**Problem:** Our 16GB RTX 5080s are VRAM-constrained running qwen3:14b at 32K context with `num_parallel=2`. KV cache is the bottleneck.

**Key constraint:** TurboQuant is HuggingFace-only. Ollama uses llama.cpp (C++). No direct injection possible. However, Ollama already supports native `q4_0` KV cache — that's the zero-effort baseline to beat.

---

## Phase 1: Benchmark (1 day, 1.5 GPU-hours)

**Goal:** Determine if TurboQuant beats Ollama's native q4_0 enough to justify a sidecar server.

### Step 1a: Ollama q8_0 vs q4_0 (30 min, GIGA)

Create `/opt/turboquant/bench/ollama_kv_bench.py`:
- Load qwen3:14b with `OLLAMA_KV_CACHE_TYPE=q8_0` and `q4_0`
- Run 50-prompt eval (reasoning, code, factual QA) via `/api/generate`
- Measure: VRAM delta, tok/s, TTFT at 16K/32K context, quality (LLM-judge 0-5)
- Config change: `/opt/hydra-project/ansible/inventory/group_vars/gpu_nodes.yml` (`ollama_kv_cache_type`)

### Step 1b: TurboQuant on HuggingFace (1 hr, MEGA)

Create `/opt/turboquant/bench/hf_turboquant_bench.py`:
- Load Qwen2.5-14B-Instruct via HuggingFace in FP16
- Apply `patch_model_for_paper_generation()` from turboquant.py
- Test: FP16 baseline, TQ 3-bit (Paper), TQ 4-bit (MSE-only)
- Same 50-prompt eval set
- Measure VRAM via `torch.cuda.max_memory_allocated()` at 8K/16K/32K

### Step 1c: Decision matrix

| Config | VRAM (32K) | tok/s | Quality | Infra Cost |
|--------|-----------|-------|---------|------------|
| Ollama q8_0 | baseline | baseline | baseline | zero |
| Ollama q4_0 | ? | ? | ? | 1-line config |
| TQ 3-bit | ? | ? | ? | sidecar server |
| TQ 4-bit | ? | ? | ? | sidecar server |

**Go/No-Go:**
- If q4_0 quality within 0.3 of q8_0 → **deploy q4_0, stop here**
- If TQ 3-bit quality within 0.5 of FP16 AND saves >1.5GB over q4_0 → Phase 2
- If TQ throughput <50% of Ollama → Phase 2 only if VRAM gain enables critical use case

---

## Phase 2: Wrapper Service (1 week, if Phase 1 justifies)

**Goal:** FastAPI server with OpenAI-compatible API running TurboQuant.

Port: **8560** | Host: GIGA or MEGA | Register in port-registry.md

### Files to create

```
/opt/turboquant/
  server/
    app.py           — FastAPI: POST /v1/chat/completions, GET /v1/models, GET /health, GET /metrics
    config.py        — Model name, bits, max_context, VRAM budget
    model_manager.py — HF model loading + TurboQuant patching + cache lifecycle
  docker/
    Dockerfile       — nvidia/cuda + torch + transformers + scipy + fastapi
    docker-compose.yml
  bench/             — Phase 1 scripts (already created)
```

### Key integration point
Christi's vLLM backend client (`/opt/christi-project/christi/inference/vllm_backend.py`) already speaks OpenAI-compatible protocol — connects without code changes by setting `vllm_host: http://127.0.0.1:8560`.

### Performance concern
`_pack_unsigned()` (turboquant.py:116) has a Python-level `for i in range(width)` loop — will be slow. Wrap with `@torch.compile` as a first pass.

---

## Phase 3: Fleet Integration (3-5 days)

- **GPU Guardian** (port 9095): Add TurboQuant health check to probe, report cache compression stats in `/status`
- **Model Routing Proxy** (port 8550): Route long-context requests (>16K) to TurboQuant server
- **Claude-Swarm**: Register "turboquant" as an inference capability for dispatch matching
- **Prometheus/Grafana**: Scrape `/metrics` from TurboQuant server

---

## Phase 4: Optimization (2-4 weeks, ongoing)

- Triton kernels for bit pack/unpack (10-50x speedup)
- `torch.bucketize()` for Lloyd-Max lookup (replace O(n*k) with O(n log k))
- Continuous batching for `num_parallel >= 2`
- Per-request cache isolation + LRU eviction
- Monitor llama.cpp for native 3-bit KV cache support (would obsolete this)

---

## Critical Files

| File | Role |
|------|------|
| `/opt/turboquant/turboquant.py` | Core library — compression, caching, model patching |
| `/opt/hydra-project/ansible/inventory/group_vars/gpu_nodes.yml` | `ollama_kv_cache_type` setting |
| `/opt/hydra-project/ansible/roles/ollama/templates/ollama-container.service.j2` | Template for TQ service |
| `/opt/christi-project/christi/inference/vllm_backend.py` | OpenAI-compatible client (reuse) |
| `/opt/hydra-project/libs/gpu_guardian/gpu_guardian/sdk.py` | GPU pre-flight SDK (extend) |
| `/opt/hydra-project/docs/port-registry.md` | Register port 8560 |

## Verification

1. Phase 1: benchmark results table in `/opt/turboquant/bench/results.md`
2. Phase 2: `curl http://127.0.0.1:8560/v1/models` returns loaded model list
3. Phase 2: `curl -X POST http://127.0.0.1:8560/v1/chat/completions -d '{"model":"qwen3:14b-tq3","messages":[{"role":"user","content":"Hello"}]}'` returns coherent response
4. Phase 3: GPU Guardian `/status` shows TurboQuant server in fleet
5. Phase 3: Christi connects via factory with `backend: vllm, host: http://127.0.0.1:8560`
