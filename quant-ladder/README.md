# FLUX 3 Klein quant ladder

Staged compression ladder on the **BASE** checkpoint (not the distilled one
everything before used). Base first, compress gradually, find the most
stable version, then optimize its speed.

## Design

- **Stage 0 — base reference.** BASE checkpoint at BF16
  (`ml-resources/flux-3-image-klein-comfy`,
  `diffusion_models/flux3-image-klein-base-bf16.safetensors`, ungated mirror
  of the gated `black-forest-labs/flux-3-image-klein`). Generated with sd.cpp
  at 50 steps CFG 3.5 (BFL spec, true reference) and at the ladder steps/CFG
  (metric reference, generated first so it exists even if 50 steps times out).
  Same swordsman prompt + seed 42 as runs 1-2, verbatim from `client.py`.
- **Stage 1 — ladder.** From base BF16, `sd-cli -M convert` one level at a
  time: FP8 (`f8_e4m3`, probe — skip if sd-cli rejects it) -> Q8_0 -> Q6_K
  -> Q5_K_M -> Q4_K_M -> Q3_K -> Q2_K -> TQ2. Same tensor-type-rules for all
  (`modulations=f16,emb_in=f16`, as in the Bonsai jobs). Each level: generate
  512^2 at fixed steps/CFG, SSIM + PSNR + mean abs diff vs the metric
  reference, eyeball, then delete the GGUF (disk).
- **Most stable** = lowest precision with eyeball PASS and SSIM >= 0.85 vs
  the base reference (rule recorded in the final report).
- **Stage 2 — speed.** Only on the winner: threads 2 vs 4, steps 12 / 8 vs
  ladder steps, EasyCache off vs on, `--conv-direct on`, one variable at a
  time, metrics vs the same reference. Probes are configurable via the
  workflow `probes` input (`base,s12,s8,nocache,t2,convdirect`).

## Dispatch plan (max 4 runs, 120-min cap each)

1. `stage0` (steps 16, cfg 3.5, stage0_steps 50) — base reference.
   DONE 2026-11-27, run 45917327154: base-16 SUCCESS 1653.3s (metric ref);
   base-50 TIMEOUT at 3000s (BF16 ~5x slower/step than Q4; 50 steps
   impractical on CPU — ladder fixed at 16 steps CFG 3.5).
2. `ladder` levels `fp8,q8_0,q6_k` (steps 16, cfg 3.5).
   DONE 2026-11-27, run 45927331511: FP8 convert OK but gen TIMEOUT 1800s;
   Q8_0 TIMEOUT 1800s; **Q6_K SUCCESS 516.4s, SSIM 0.9621, PSNR 29.26 dB**,
   eyeball PASS.
3. `ladder` levels `q5_k_m,q4_k_m,q3_k` (steps 16, cfg 3.5).
   DONE 2026-11-27, run 45937997623: **Q5_K_M 467.2s SSIM 0.9570 PSNR 27.42
   PASS**; **Q4_K_M 457.9s SSIM 0.9045 PSNR 24.69 PASS**; Q3_K 489.5s
   SSIM 0.8182 PSNR 20.60 — below the 0.85 rule, FAIL (eyeball: coherent
   swordsman but composition drifted, katana rendering off).
   **Stage-1 verdict by the rule: Q4_K_M is the most stable** (lowest
   precision with eyeball PASS + SSIM >= 0.85). Q5_K_M is the user's
   pick for the speed stage: nearly as fast as Q4 (467.2s vs 457.9s)
   with much better fidelity.
4. `speed` winner `q5_k_m` (steps 16, cfg 3.5,
   probes `base,s12,s8,nocache,t2,convdirect`) — user order 2026-11-27
   "Q5km thêm cách tăng tốc vào để chạy thử xem". DONE 2026-11-27, run
   45946274690 (success, ~121 min — nearly hit the 120-min cap because
   this host was ~2x slower than run 2's host; only WITHIN-run wall
   times are comparable):
   base 1037.4s SSIM 0.9485; **s12 771.3s SSIM 0.9116 — 26% faster than
   in-run base, image still clean (best speed config)**; s8 767.7s
   SSIM 0.8762 — no faster than s12, lower fidelity, not worth it;
   nocache 1477.3s SSIM 0.9354 — EasyCache off costs +42%, keep it on;
   t2 1110.0s — 2 threads only +7% slower (byte-identical image);
   convdirect 1039.6s — `--conv-direct on` does nothing on CPU
   (byte-identical image). All six probes eyeball PASS.
   The Q2_K/TQ2 ladder run (run 4 of 4) only happens if the user orders it.

Ladder/speed runs fetch the stage-0 reference from the `quant-ladder-stage0`
artifact via the GitHub API (no repo bloat, no run id needed).

## Files

- `quant-ladder/run_ladder.py` — the driver (sd.cpp only, numpy/Pillow metrics).
- `.github/workflows/flux3-quant-ladder.yml` — the workflow.
- Results (local): `~/workspace/flux3-klein-cpu-test/results-quant-ladder/`.

## Known context

- TQ2 straight from BF16 already failed once (Bonsai amax diagnostic, run
  45865137246: pure noise) — expect the TQ2 rung to fail; it is in the
  ladder to document it with numbers, not because it is expected to work.
- Q8_0 is slow on CPU (1291s at 16 steps in run 2 vs Q4 241s) — ladder run 2
  (fp8/q8/q6) is the slowest run; watch the 120-min cap.
- sd-cli `--type` for generation lists only f32/f16/q4_0/q4_1/q5_0/q5_1/
  q8_0/q2_k/q3_k/q4_k in the README, but `-M convert` accepted `tq2_0`
  (Bonsai round 1). The driver tries each level's types in order and records
  what was actually used; a level whose types all fail is skipped, never fatal.
