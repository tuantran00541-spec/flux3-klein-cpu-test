# Qwen-Image-2.1 quant ladder

Staged compression ladder on the **BASE** Qwen-Image-2.1 checkpoint, same
treatment as the FLUX 3 Klein ladder (`quant-ladder/`): base reference
first, compress gradually, find the most stable version, then optimize
its speed.

## Design

- **Stage 0 — base reference.** Merged F16 DiT (`Qwen/Qwen-Image-2.1`,
  `transformer/` 2 shards, ~14 GB BF16 -> ~7.1 GB F16; the merged BF16
  is 14.23 GB and CANNOT load on the 15 GiB runner — sd.cpp's weight
  manager allows ~13.27 GB, "RAM weight capacity exceeded", run 2)
  + Q8_0 text encoder
  (`Qwen/Qwen3-VL-8B-Instruct-GGUF`, `Qwen3VL-8B-Instruct-Q8_0.gguf`) +
  VAE (`Comfy-Org/Qwen-Image-2.1`, `vae/qwen_image_2.1_vae_bf16.safetensors`).
  Single generation at 20 steps / CFG 6.0 / euler / seed 42 / 512^2 — the
  proven sd.cpp data point for base 2.1 (Frosty40 / DogukanUrker /
  AtomicChat). Same swordsman prompt + seed as the FLUX ladder so the two
  models are comparable. No 50-step second reference: unaffordable on CPU.
  The BF16->F16 cast is exact except F16-subnormal tiny weights
  (|x| < 6.1e-5) — negligible vs quant noise; merge streams one tensor
  at a time (peak RAM ~7 GB).
- **Stage 1 — ladder.** From merged F16, `sd-cli -M convert` one level at
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
   Run 1 (45954967560, 10m10s): FAILED fast at merge — `safetensors.numpy`
   `TypeError: data type 'bfloat16' not understood` (numpy needs
   `ml_dtypes` for BF16). Shard names/sizes confirmed
   (9.97 GB + 4.26 GB, 80s download, 78.1 GB free disk). Fix: `ml_dtypes`
   added to pip install + imported in `merge_shards`; driver now writes
   `results.json` even on early failure. Run 2 = stage0 retry.
   Run 2 (45956491363, 11m21s): merge FIXED (297 tensors, 14,230,284,584 B
   in 110s; TE 8,709,519,456 B; VAE 675,509,688 B; total driver 308s) and
   sd.cpp DOES detect Qwen-Image-2.1 (32 layers, prompt encoded in 26s)
   — but generation FAILED in 29.4s: `[E] RAM weight capacity exceeded.
   Requested 14230274048 bytes, 13270585344 bytes available` — runner
   has 15 GiB total, 14 GiB available; the BF16 DiT alone needs 14.23 GB.
   Fix: merge casts BF16->F16 (streaming, one tensor at a time) and the
   reference + ladder converts use the ~7.1 GB F16 file. Run 3 = stage0
   + chained ladder.
2. `stage0` + chained `ladder` levels `q6_k,q5_k_m,q4_k_m`
   (steps 20, cfg 6.0, `run_ladder=true`) — the plausible-stable range
   first (run budget is now 2 runs left, not 4; chaining saves a run).
3. `ladder` levels `q3_k,q2_k,q8_0` (steps 20, cfg 6.0, ref from the
   stage0 artifact) — Q3/Q2 first, Q8_0 last (likely timeout, as in FLUX).
   Speed stage only if a run remains after stage 1 (OPEN unless a ladder
   run is skipped).
4. (no 4th run remains in the plan — runs 1+2 were spent on the merge
   and RAM failures; see 2–3).

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
