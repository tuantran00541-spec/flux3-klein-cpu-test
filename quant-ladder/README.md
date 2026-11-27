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
- **Stage 2 — speed.** Only on the winner: threads 2 vs 4, steps 12 vs
  ladder steps, EasyCache off vs on, one variable at a time, metrics vs the
  same reference.

## Dispatch plan (max 4 runs, 120-min cap each)

1. `stage0` (steps 16, cfg 3.5, stage0_steps 50) — base reference.
2. `ladder` levels `fp8,q8_0,q6_k` (steps 16, cfg 3.5).
3. `ladder` levels `q5_k_m,q4_k_m,q3_k` (steps 16, cfg 3.5).
4. `ladder` levels `q2_k,tq2` + `run_speed=true` winner=<best of runs 2-3>.

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
