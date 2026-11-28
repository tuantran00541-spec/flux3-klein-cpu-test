# Qwen-Image-2.1 deep-quant sweep — final report (2026-11-27/28)

Repo: tuantran00541-spec/flux3-klein-cpu-test, workflow `qwen21-deepquant.yml`,
driver `qwen21-ladder/run_deepquant.py`, probes
`qwen21-ladder/deepquant-probes.json`. Same test as the ladder + R1–R6 speed
test: swordsman prompt, seed 42, 512², SSIM/PSNR vs the Q6_K reference
base-20.png. Baseline = 20 steps / CFG 6.0 / 4 threads. Stacked = EasyCache +
12 steps (R6 best combo). Turbo probes = 8 steps / CFG 1.0 / sd.cpp preset
`qwen-image-2.1-turbo`.

**6/6 runs used, 21/23 probes success, 2 FP8 probes failed (load error).
All 6 runs completed success.** Results in `results-qwen21-deepquant/D1/`–`D6/`
(results.json + PNGs + sd-cli logs). Driver commit `15294ed`.

## Per-group results

### D1 — own Q4 (run 46076596744, success, 3/3, elapsed 2856.5s)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| own-q4-base | 20/6.0 | 1155.6s | 0.6774 | 13.39 | clean, composition preserved (drift is tonal/detail) |
| own-q4-stacked | 12/6.0 + EasyCache | 527.9s | 0.5709 | 11.59 | clean, darker/muted (s12 tonal cost) |
| q6-stacked-control | 12/6.0 + EasyCache | 564.0s | 0.5933 | 12.29 | clean, s12 look — SSIM byte-identical to R6's control |

Within-run: Q4-stacked 6.4% faster than the Q6 control (527.9 vs 564.0s).
Q4 base is only slightly faster than the Q6_K ref (1233.5s ladder) but drifts
more than Q5_K_M did (0.6774 vs 0.7191).

### D2 — own Q3 (run 46082424954, success, 3/3, elapsed 4197.0s)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| own-q3-base | 20/6.0 | 1959.8s | 0.4911 | 9.84 | coherent, composition preserved, pose shifted (faces more forward, katana angle odd) |
| own-q3-stacked | 12/6.0 + EasyCache | 876.1s | 0.5291 | 10.76 | clean, darker/muted s12 look, face in shadow |
| q6-stacked-control | 12/6.0 + EasyCache | 880.8s | 0.5933 | 12.29 | clean, s12 look — SSIM byte-identical (slow host) |

Within-run: Q3-stacked 0.5% faster than the Q6 control — no gain. Q3 base
is much SLOWER than Q6 base (1959.8s vs 1233.5s) — dequant overhead outweighs
memory savings (same pattern as FLUX, where Q3_K was slower than Q4_K_M).
Q3 is a dead end for speed.

### D3 — own Q2 (run 46090176956, success, 3/3, elapsed 3447.6s)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| own-q2-base | 20/6.0 | 1371.1s | 0.5236 | 10.28 | coherent (NOT blank, unlike FLUX Q2), pose drifted (forward-facing, katana angle odd) |
| own-q2-stacked | 12/6.0 + EasyCache | 615.7s | 0.5192 | 10.00 | clean, darker/muted, same pose drift |
| q6-stacked-control | 12/6.0 + EasyCache | 594.6s | 0.5933 | 12.29 | clean, s12 look — SSIM byte-identical |

Within-run: Q2-stacked 3.5% SLOWER than the Q6 control. Q2 base slower than
Q6 and Q4, faster only than Q3.

### D4 — unsloth Q4 + Q3_XL (run 46096029233, success, 4/4, elapsed 3769.6s)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| unsloth-q4-base | 20/6.0 | 1239.6s | 0.8451 | 20.72 | **clean, closest to the reference of all deep quants** — same pose/composition as ref |
| unsloth-q4-stacked | 12/6.0 + EasyCache | 573.9s | 0.5642 | 12.09 | clean, s12 look |
| unsloth-q3xl-base | 20/6.0 | 1283.5s | 0.5785 | 11.93 | clean, close to reference pose, better than own Q3 |
| unsloth-q3xl-stacked | 12/6.0 + EasyCache | 580.6s | 0.5828 | 12.16 | clean, s12 look |

Third-party quants beat our own at the same bit-width — Unsloth Q4 0.8451 vs
own Q4 0.6774; Unsloth Q3_XL 0.5785 vs own Q3 0.4911. (Likely Dynamic 2.0's
better scale/importance handling vs sd-cli's plain convert.) Speed: no gain —
unsloth-q4-base ≈ Q6 base; stacked ≈ Q6 control cross-run. Quality win, not
a speed win.

### D5 — unsloth Q2 + leejet Q2 (run 46102218971, success, 4/4, elapsed 4005.6s)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| unsloth-q2-base | 20/6.0 | 1354.8s | 0.5064 | 10.67 | clean, forward-facing pose drift |
| unsloth-q2-stacked | 12/6.0 + EasyCache | 568.7s | 0.5019 | 10.31 | clean, s12 look |
| leejet-q2-base | 20/6.0 | 1379.7s | 0.5368 | 10.63 | clean, forward-facing, katana slightly odd (blade looks doubled) |
| leejet-q2-stacked | 12/6.0 + EasyCache | 621.1s | 0.5230 | 10.59 | clean, s12 look |

Both third-party Q2s coherent like own Q2; leejet marginally better than
unsloth on SSIM (0.5368 vs 0.5064). No speed gain.

### D6 — Turbo deep + FP8 (run 46108348394, success, 4/6 success + 2 FP8 failed, elapsed 1557.7s)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| turbo-q6-easycache | 8/1.0 + EasyCache | 473.1s | 0.4356 | 9.63 | clean bust close-up portrait (Turbo reframe, same as R5) |
| turbo-adq4-easycache | 8/1.0 + EasyCache | 282.2s | 0.4488 | 9.82 | **clean, best-looking Turbo — detailed face, fastest Turbo probe**; reframed to bust close-up |
| turbo-q2-easycache | 8/1.0 + EasyCache | 301.8s | 0.5124 | 11.55 | clean, reframed to close-up, face softer than AD-Q4; highest Turbo SSIM |
| turbo-q3xl-easycache | 8/1.0 + EasyCache | 319.3s | 0.5044 | 10.56 | clean, reframed to close-up |
| fp8-base | 20/6.0 | 0.1s | — | — | **FAILED — sd-cli exit 1**: Unsloth FP8 safetensors uses diffusers-style tensor names (`model.diffusion_model.transformer_blocks.*`); sd.cpp metadata validation rejects them ("not in model metadata") |
| fp8-stacked | 12/6.0 + EasyCache | 0.0s | — | — | **FAILED — same sd-cli exit 1**, same cause |

Turbo deep quants are the fastest probes in the whole sweep (282–473s), but
all reframe to a bust/upper-body close-up — separate lane, deliberate accept
only. FP8 direct load is dead as attempted; a convert-first run would be
needed to test FP8 at all.

## Final verdicts

- **Lowest stable deep quant (eye-clean at baseline): Unsloth Q4_K_M
  (SSIM 0.8451)** — the only deep quant that stays close to the Q6 reference
  at 20 steps. Own Q4 (0.6774) is eye-clean but drifted; Q3/Q2 (own and
  third-party, 0.49–0.58) are coherent but pose-drifted.
- **Fastest acceptable stacked combo (same full-body composition): Q6_K +
  EasyCache + 12 steps, 681.2s within-run (R6)** — no deep quant beats it.
  Own Q4 stacked 527.9s is faster but drifts (SSIM 0.5709); Unsloth Q4
  stacked 573.9s is close to the Q6 control and drifts less in absolute
  terms only because the s12 tonal cost dominates both.
- **Fastest overall (different framing): Turbo AD-Q4_K + EasyCache,
  282.2s** — bust close-up, detailed face. Turbo Q2 (301.8s) and Q3_XL
  (319.3s) are close behind.
- **Third-party vs own:** third-party wins on quality at the same bit-width
  — Unsloth Q4 0.8451 vs own Q4 0.6774; Unsloth Q3_XL 0.5785 vs own Q3
  0.4911; leejet Q2 0.5368 vs own Q2 0.5236 (marginal). No third-party build
  wins on speed.
- **Deep quants do NOT speed up Qwen on CPU.** Baseline: Q4 1155–1240s ≈
  Q6 1233s; Q3 1284–1960s; Q2 1355–1380s — all flat or slower. Stacked:
  deep quants land in the same 528–621s band as the Q6 control (564–881s
  cross-host). Only EasyCache + fewer steps + Turbo distillation move the
  time; quantization only shrinks disk/RAM.
- **FP8:** direct safetensors load FAILED (metadata validation) — untested,
  needs a convert-first attempt.

## All-probe table (SSIM vs Q6_K ref)

| Probe | Wall | SSIM |
|---|---|---|
| own-q4-base | 1155.6s | 0.6774 |
| own-q4-stacked | 527.9s | 0.5709 |
| own-q3-base | 1959.8s | 0.4911 |
| own-q3-stacked | 876.1s | 0.5291 |
| own-q2-base | 1371.1s | 0.5236 |
| own-q2-stacked | 615.7s | 0.5192 |
| unsloth-q4-base | 1239.6s | 0.8451 |
| unsloth-q4-stacked | 573.9s | 0.5642 |
| unsloth-q3xl-base | 1283.5s | 0.5785 |
| unsloth-q3xl-stacked | 580.6s | 0.5828 |
| unsloth-q2-base | 1354.8s | 0.5064 |
| unsloth-q2-stacked | 568.7s | 0.5019 |
| leejet-q2-base | 1379.7s | 0.5368 |
| leejet-q2-stacked | 621.1s | 0.5230 |
| turbo-q6-easycache | 473.1s | 0.4356 |
| turbo-adq4-easycache | 282.2s | 0.4488 |
| turbo-q2-easycache | 301.8s | 0.5124 |
| turbo-q3xl-easycache | 319.3s | 0.5044 |
| fp8-base | failed | — |
| fp8-stacked | failed | — |
| (controls) q6-stacked D1/D2/D3 | 564.0 / 880.8 / 594.6s | 0.5933 (identical) |

Controls: the Q6 stacked control reproduces SSIM 0.5933 exactly in D1, D2
and D3 (deterministic across runs); only its wall time varies with the host
(564–881s) — host variance is real, within-run deltas only.

## OPEN / obstacles

- **Budget exhausted (6/6 runs used).** Nothing further in this series
  without a new order.
- **FP8 untested** — direct load dead; needs a convert-first attempt
  (FP8 safetensors → GGUF or sd.cpp-format safetensors) in a new run.
- **Abiray base Q3_K_M** (secondary vendor) was dropped in research and never
  tested — OPEN if budget remains (unlikely to beat Unsloth's Q3_XL).
- **Disk quota** on the VM remains near-full: one D1 commit hit EDQUOT and
  needed ~13 MB of regenerable caches cleared (thumbnails, gh CLI cache).
  HF cache (3.3 GB) is the remaining candidate if a future commit blocks.
- Execution history: the run phase was blocked twice by execution-risk
  flags on fetched remote content (first on 515269b, then on f92257f); the
  user explicitly confirmed proceeding both times (05:20 "Cho chạy tiếp đi",
  05:25 "Cứ chạy đi, tôi chịu" for the f92257f flag specifically).
