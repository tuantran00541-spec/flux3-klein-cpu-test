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
| OV-1 | - | viggle | Viggle Turbo 6-step fused Q6_K (cfg 1.0, Viggle sigmas) base + easycache; Lightning 4-step fused Q6_K base | pending |
| OV-2 | - | cache | sd.cpp --cache-mode sweep, 20 steps Q6_K: easycache (baseline) + taylorseer + cache-dit + dbcache + spectrum + ucache | pending |
| OV-3 | - | res384 | Q4_K_M + EasyCache 20 steps; 384² + Lanczos upscale to 512² (plain + easycache) | pending |
| OV-4 | - | openvino | blaj INT4 via OVDiffusionPipeline on CPU, 10 steps, SPEED-ONLY (abliterated TE — SSIM not comparable); released stack first, nightly retry once | pending |

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
