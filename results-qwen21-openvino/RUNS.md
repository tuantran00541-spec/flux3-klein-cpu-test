# Qwen-Image-2.1 extra-speedup explore — run tracker (2026-11-27)

Baseline for comparison: R1–R5 speed test (results-qwen21-speed/REPORT.md) —
best quality-preserving config Q6_K 20 steps + EasyCache 716.7s, SSIM 0.9242;
Turbo + EasyCache fastest 367.6s but reframes. Same test here: swordsman
prompt, seed 42, 512², SSIM/PSNR vs Q6_K reference base-20.png.

Workflow: `.github/workflows/qwen21-openvino.yml` (NEW — does NOT touch
qwen21-quant-ladder.yml, which R6 owns). Driver: `qwen21-ov/run_explore.py`
(+ `ov_attempt.py` for the OpenVINO path). Budget: max 4 runs total.

| # | Run ID | Path | Config | Status |
|---|---|---|---|---|
| OV-1 | 46016329942 | viggle | Viggle Turbo 6-step fused Q6_K (cfg 1.0, Viggle sigmas) base + easycache; Lightning 4-step fused Q6_K base | ✅ success (14:51 UTC, elapsed 608.6s). viggle-base 167.5s SSIM 0.4371 — framing KEPT, face blurred/featureless, softer detail; viggle-easycache 152.6s SSIM 0.4537 — framing kept, MORE degraded (blank face, heavy grain); lightning-base 187.7s SSIM 0.3771 — framing kept, cleanest of the three by eye (face + robes readable), lowest SSIM is pose/position shift, not a reframe. None reframed to close-up. |
| OV-2 | 46018091029 | cache | sd.cpp --cache-mode sweep, 20 steps Q6_K: easycache (baseline) + taylorseer + cache-dit + dbcache + spectrum + ucache | ❌ CANCELLED — no results. 120-min workflow timeout killed the driver step (~16:40 UTC, ~119 min in); job conclusion `cancelled`, **0 artifacts**. The driver saves results.json after each probe, so completed probes' PNGs + results.json existed on the runner, but the upload step was cut off by the same timeout before anything landed. Run logs unrecoverable (`gh run view --log`: "log not found"; jobs API: BlobNotFound 404) — how many of the 6 probes finished is UNKNOWN. Cause: 6 probes × ~700–1000s sampling on a slow host + ~15 min setup > 120 min; the driver's 100-min JOB_BUDGET only stops *starting* new probes, it cannot save a probe already running when the workflow timeout fires. NOT re-dispatched (4-run budget used). |
| OV-3 | 46018110199 | res384 | Q4_K_M + EasyCache 20 steps; 384² + Lanczos upscale to 512² (plain + easycache) | ✅ success (15:56 UTC, elapsed 3307.9s). q4-easycache 820.3s (sampling 678.3s) SSIM 0.6782 — eye CLEAN/near-ref, but SLOWER than the 716.7s Q6_K best: Q4 buys no speed. res384-lanczos 940.6s (sampling 824.7s!) SSIM 0.2385 — eye clean quality but REFRAMED (tighter framing, swordsman fills frame); 384² sampling slower than 512² on this host (host variance). res384-lanczos-easycache 563.5s (sampling 465.9s) SSIM 0.2436 — eye clean, same reframe; within-host EasyCache saves 44% at 384² (825→466s), but reframed → lane disqualified on framing. |
| OV-4 | 46018129474 | openvino | blaj INT4 via OVDiffusionPipeline on CPU, 10 steps, SPEED-ONLY (abliterated TE — SSIM not comparable); released stack first, nightly retry once | ❌ FAILURE (14:41:38 UTC, ~30s in). Died in workflow step "Install OpenVINO stack": pip ResolutionImpossible — `optimum-intel==2.2.0` conflicts with `transformers==5.15.1`. Driver never ran (nightly retry lives in the driver, never reached). No artifact. Fix needs a compatible pin set in the workflow; NOT re-dispatched (4-run budget used). |

## Research verdicts (web/docs verified 2026-11-27, research subagent)

- **Viggle Turbo (top bet):** DMD-distilled 6-step LoRA fused to GGUF
  (`Abiray/...-viggle-turbo-v0.3-6step-GGUF`, Q6_K 5.88 GB), claimed ~4.5×.
  Different distillation from official Turbo (already tested: reframes).
  Sigmas 512²: 1.000000,0.962557,0.923057,0.837170,0.631512,0.363570,0.0.
  Lightning 4-step fused Q6_K (`darioooooo0o/...`, 6.27 GB), 4 steps cfg 1.0
  euler simple, sigmas 1.0/0.857/0.667/0.400, LPIPS Q6_K 0.101 vs bf16.
- **OpenVINO:** GenAI table DOES list Qwen-Image-2.1 (T2I/I2I ✅); optimum-intel
  2.2.0 lists it too. NO official pre-converted model (OpenVINO org only
  publishes Qwen3.x VLMs). blaj INT4: transformer stock, TE abliterated,
  12.9s/512²/10 steps on Intel Arc iGPU (NOT CPU), needs nightly stack
  (diffusers + optimum-intel main + OV 2026.5.0 nightly, py 3.12).
  Full-pipeline export OOM-killed on 30 GB — never attempt as one export.
  The "unsupported QwenImage21Pipeline" error could not be verified live;
  workaround if it recurs: OVDiffusionPipeline instead of GenAI Text2ImagePipeline.
- **DROPPED (with reason):** TeaCache (no flag in sd.cpp; vLLM-Omni marks it
  ❌ for 2.1); ToMe token merging (no patch for QwenImage21); IPEX (no 2.1
  recipe/number; runner CPUs mostly AMD, no AMX; doesn't fix 33 GB RAM);
  text-encoder quant for speed (TE ~0.4s warm vs ~10s/DiT step, <2% share);
  TinyVAE as final decoder (ComfyUI preview only, RGB-only; VAE ~2% of wall).
