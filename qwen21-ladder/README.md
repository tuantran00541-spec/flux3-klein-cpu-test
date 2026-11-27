# Qwen-Image-2.1 quant ladder

Staged compression ladder on the **BASE** Qwen-Image-2.1 checkpoint, same
treatment as the FLUX 3 Klein ladder (`quant-ladder/`): base reference
first, compress gradually, find the most stable version, then optimize
its speed.

## Design

- **Stage 0 — base reference.** Merged BF16 DiT (`Qwen/Qwen-Image-2.1`,
  `transformer/` 2 shards, ~14 GB) + Q8_0 text encoder
  (`Qwen/Qwen3-VL-8B-Instruct-GGUF`, `Qwen3VL-8B-Instruct-Q8_0.gguf`) +
  VAE (`Comfy-Org/Qwen-Image-2.1`, `vae/qwen_image_2.1_vae_bf16.safetensors`).
  Single generation at 20 steps / CFG 6.0 / euler / seed 42 / 512^2 — the
  proven sd.cpp data point for base 2.1 (Frosty40 / DogukanUrker /
  AtomicChat). Same swordsman prompt + seed as the FLUX ladder so the two
  models are comparable. No 50-step second reference: unaffordable on CPU.
- **Stage 1 — ladder.** From merged BF16, `sd-cli -M convert` one level at
  a time: Q8_0 -> Q6_K -> Q5_K_M -> Q4_K_M -> Q3_K -> Q2_K. Each level:
  generate 512^2 at fixed steps/CFG, SSIM + PSNR + mean abs diff vs the
  metric reference, eyeball, then delete the GGUF (disk).
- **Most stable** = lowest precision with eyeball PASS and SSIM >= 0.85 vs
  the base reference (same rule as the FLUX ladder).
- **Stage 2 — speed.** Only on the winner, and only if a run remains:
  threads 2 vs 4, steps 16 / 12 vs ladder steps, `--conv-direct on`, one
  variable at a time, metrics vs the same reference.

Deliberate deviations from the FLUX ladder: steps 20 / CFG 6.0 (vs 16 /
3.5); EasyCache OFF everywhere (unproven for Qwen flow-matching); no
`--llm_vision` / mmproj (proven invocations pass only `--llm`); shards are
merged with `safetensors.numpy` (no torch) and deleted after merging;
no fp8 / tq2 rungs.

## Dispatch plan (max 4 runs, 120-min cap each)

1. `stage0` (steps 20, cfg 6.0) — base reference. Validates the whole
   pipeline (download -> merge -> generate).
2. `ladder` levels `q6_k,q8_0` (steps 20, cfg 6.0) — Q6 first so the
   plausible-stable rung lands even if Q8_0 eats the budget.
3. `ladder` levels `q5_k_m,q4_k_m` (steps 20, cfg 6.0).
4. `ladder` levels `q3_k,q2_k` (steps 20, cfg 6.0).
   Speed stage only if a run remains after stage 1 (it does not in this
   plan — OPEN unless a ladder run is skipped).

Ladder/speed runs fetch the stage-0 reference from the
`qwen21-ladder-stage0` artifact via the GitHub API (no repo bloat, no run
id needed).

## Files

- `qwen21-ladder/run_ladder.py` — the driver (sd.cpp only, numpy/Pillow
  metrics, safetensors.numpy shard merge).
- `.github/workflows/qwen21-quant-ladder.yml` — the workflow.
- Results (local): `~/workspace/flux3-klein-cpu-test/results-qwen21-ladder/`.

## License

Qwen-Image-2.1 is under the **Qwen Research License** — research /
evaluation only, commercial use needs a paid license.

## FLUX ladder reference numbers (same prompt/seed, 512^2, 16 steps)

Base BF16 1653s; FP8 + Q8_0 both timeout 1800s; Q6_K 516s SSIM 0.9621;
Q5_K_M 467s SSIM 0.957; Q4_K_M 458s SSIM 0.9045; Q3_K 490s SSIM 0.8182.
Most stable: Q4_K_M.
