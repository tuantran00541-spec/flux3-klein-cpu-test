# Qwen-Image-2.1 quant ladder

Staged compression ladder on the **BASE** Qwen-Image-2.1 checkpoint, same
treatment as the FLUX 3 Klein ladder (`quant-ladder/`): base reference
first, compress gradually, find the most stable version, then optimize
its speed.

## Design

- **Stage 0 — base reference.** Merged F16 DiT (`Qwen/Qwen-Image-2.1`,
  `transformer/` 2 shards, ~14 GB BF16 (merged F16 is the same size —
  both are 2 bytes/element); the merged BF16
  is 14.23 GB and CANNOT load on the 15 GiB runner — sd.cpp's weight
  manager allows ~13.27 GB, "RAM weight capacity exceeded", run 2)
  + Q8_0 text encoder
  (`Qwen/Qwen3-VL-8B-Instruct-GGUF`, `Qwen3VL-8B-Instruct-Q8_0.gguf`) +
  VAE (`Comfy-Org/Qwen-Image-2.1`, `vae/qwen_image_2.1_vae_bf16.safetensors`).
  CORRECTION (run 3): F16 does NOT halve anything — BF16 and F16 are
  both 2 bytes/element, so the merged F16 file is 14,230,284,288 bytes
  and fails to load the same way ("Requested 14230274048 bytes,
  12774645760 bytes available"). Stage 0 as defined CANNOT succeed on
  this runner. The working reference stage is `q8ref`: convert the
  merged F16 -> Q8_0 GGUF (`sd-cli -M convert` streams tensors, ~2.6 GB
  RAM per the FLUX.3 Klein recipe) and generate the reference from it;
  fallback Q6_K if Q8_0 generation fails. Ladder SSIMs are therefore
  vs a Q8_0 (or Q6_K) reference, NOT true BF16 — the report must say so.
  Single generation at 20 steps / CFG 6.0 / euler / seed 42 / 512^2 — the
  proven sd.cpp data point for base 2.1 (Frosty40 / DogukanUrker /
  AtomicChat). Same swordsman prompt + seed as the FLUX ladder so the two
  models are comparable. No 50-step second reference: unaffordable on CPU.
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
   reference + ladder converts were to use the F16 file (then believed
   to be ~7.1 GB — wrong, see run 3). Run 3 = stage0
   + chained ladder.
   Run 3 (45958588714, 10m36s): merge streamed fine (297 tensors F16 in
   157s) — but the merged F16 file is 14,230,284,288 bytes, i.e. the
   SAME size as BF16 (both are 2 bytes/element; the "halves memory"
   assumption was wrong), and generation failed identically in 16.4s:
   `[E] RAM weight capacity exceeded. Requested 14230274048 bytes,
   12774645760 bytes available`. Stage 0 is impossible on this runner.
   Fix for run 4: new `q8ref` stage — convert F16 -> Q8_0 GGUF with
   `sd-cli -M convert` (streaming, ~2.6 GB RAM) and use THAT as the
   reference, Q6_K fallback; ladder rungs convert once from F16.
2. `q8ref` + chained `ladder` levels `q6_k,q5_k_m,q4_k_m,q3_k,q2_k`
   (steps 20, cfg 6.0, `run_ladder=true`) — LAST run (4 of 4).
   Run 4 (45960665797, 2h1m7s, conclusion cancelled = 120-min cap hit
   after upload): convert F16 -> Q8_0 OK in 34s (7,591,558,304 B) but
   Q8_0 generation TIMEOUT at 3600s; fallback Q6_K convert OK in 96s
   (5,876,563,104 B), generation SUCCESS in 1233.5s — reference is Q6_K.
   Ladder: Q5_K (converted as `q5_K`, 326s, 4,936,078,496 B) SUCCESS in
   1192.9s, SSIM 0.7191 / PSNR 15.46 dB vs Q6_K; Q4/Q3/Q2 SKIPPED
   ("out of time", 6686s elapsed — the Q8_0 timeout ate the budget).
   See "Results" below. No runs remain.

## Results (run 4, 2026-11-27; 512^2, 20 steps, CFG 6.0, euler, seed 42)

| Level | Type used | Time | SSIM vs Q6_K ref | Eyeball |
|---|---|---|---|---|
| Q8_0 (ref candidate) | q8_0 | TIMEOUT 3600s | — | — |
| **Q6_K (reference)** | q6_K | 1233.5s | — (ref) | PASS, clean swordsman |
| Q5_K ("q5_k_m") | q5_K | 1192.9s | 0.7191 / 15.46 dB | PASS, clean — composition drifted, not degraded |
| Q4_K_M | — | not run | — | OPEN |
| Q3_K | — | not run | — | OPEN |
| Q2_K | — | not run | — | OPEN |

**Verdict by the pre-registered rule (eyeball PASS + SSIM >= 0.85):
Q6_K is the most stable** — it is the only rung that meets both legs,
because it IS the reference. Q5_K passes the eyeball (clean,
on-prompt, no artifacts) but fails the SSIM leg (0.7191). The low
SSIM is composition drift, not degradation: same prompt/seed, the
Q5_K swordsman re-poses (body more frontal, katana on the other hip)
while staying fully coherent. Qwen-Image-2.1's diffusion trajectory
is far less stable under quantization than FLUX 3 Klein's (FLUX Q5_K
vs BF16: SSIM 0.957) — so for Qwen, pixel metrics overstate quality
loss and the eyeball is the deciding evidence. No claim is made about
Q4/Q3/Q2 or about true-BF16 fidelity: the true BF16 base was never
generated (14.23 GB cannot load on the 15 GiB runner), and Q4/Q3/Q2
never ran (no runs left). Speed stage: OPEN (no run left).

Notes: sd-cli `-M convert` accepts only the uppercase-K type names
(`q6_K`, `q5_K`, `q4_K`, `q3_K`); the `_M`/lowercase variants
(`q5_k_m`, `q5_K_M`, `q5_k`, `q6_k`) are rejected instantly — the
same was true in the FLUX ladder (its "Q5_K_M" was actually `q5_K`),
so the two ladders used identical types. Convert is cheap and
streaming (Q8_0 34s, Q6_K 96s, Q5_K 326s, ~2.6 GB RAM); generation
dominates (~20 min/image at Q6/Q5). Q8_0 generation is unusable on
CPU (timeout, as in FLUX). The artifact `qwen21-ladder-q8ref`
(5.84 GB, mostly the Q6_K GGUF) is on GitHub; local copies of the
two PNGs + `results.json` (reconstructed verbatim from the job log —
the 5.8 GB artifact download was cut short by the VM disk quota)
are in `~/workspace/flux3-klein-cpu-test/results-qwen21-ladder/q8ref/`.

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
