# Qwen-Image-2.1 deep-quant sweep — run tracker (2026-11-27)

User order 2026-11-27 22:56: install Qwen-Image-2.1 on the GitHub runner,
test from base down through the compressed versions, find deep-compression
builds from other well-known parties, and test speed with many CPU speedup
methods STACKED together.

Same test as the ladder + R1–R6 speed test: swordsman prompt, seed 42,
512², SSIM/PSNR vs the Q6_K reference base-20.png.
Baseline = 20 steps / CFG 6.0 / 4 threads. Stacked = EasyCache + 12 steps
(R6 best combo: Q6_K 681.2s within-run, SSIM 0.5933, darker/muted tone).
Turbo probes = 8 steps / CFG 1.0 / sd.cpp preset `qwen-image-2.1-turbo`.

Workflow: `.github/workflows/qwen21-deepquant.yml` (NEW — touches neither
qwen21-quant-ladder.yml nor qwen21-openvino.yml). Driver:
`qwen21-ladder/run_deepquant.py`, probes: `qwen21-ladder/deepquant-probes.json`.
Budget: max 6 runs total (D1–D6).

## What is already proven (do NOT re-run)

- Base BF16/F16 DiT (14.23 GB) OOMs on the 15 GiB runner (stage0 runs 2–3).
- Q8_0 sampling times out at 3600 s, no image (q8ref run 4).
- Q6_K reference: 1233.5 s, clean. Q5_K_M: 1192.9 s, SSIM 0.7191 (pose drift).
- Q4/Q3/Q2 own quants: NEVER run (q8ref run 4 skipped them, out of time).
- Speed test R1–R6 (6/6 budget exhausted): best quality-preserving Q6_K +
  EasyCache 814.1 s within-run (SSIM 0.9242); EasyCache + 12 steps stacks
  sub-additively 681.2 s; Q5 + EasyCache + 12 steps fastest 628.8 s but
  stacks both quality costs; Turbo + EasyCache fastest 367.6 s but reframes.
- OpenVINO lane (separate coordinator, still running): OV-1 Viggle Turbo
  6-step fused Q6_K 167.5 s / Lightning 4-step 187.7 s (framing kept, face
  degraded); OV-2 cache-mode sweep in progress; OV-3 Q4_K_M + EasyCache
  820.3 s SSIM 0.6782 — Q4 buys no speed vs Q6_K; res384 reframed,
  disqualified; OV-4 OpenVINO INT4 FAILED in 30 s (pip ResolutionImpossible:
  optimum-intel 2.2.0 vs transformers 5.15.1), not re-dispatched.
  Do NOT duplicate the OV lane's OpenVINO work.

## Third-party deep builds (research 2026-11-27, HF API + READMEs)

| Build | Vendor | File | Size | Load verdict on sd.cpp CPU |
|---|---|---|---|---|
| unsloth Q4_K_M | Unsloth (Dynamic 2.0, 1.4M dl) | `qwen-image-2.1-Q4_K_M.gguf` | 4.20 GB | ✅ GGUF, direct `--diffusion-model` |
| unsloth Q3_K_XL | Unsloth (recommended <4 GB pick) | `qwen-image-2.1-Q3_K_XL.gguf` | 3.61 GB | ✅ GGUF, direct |
| unsloth Q2_K | Unsloth | `qwen-image-2.1-Q2_K.gguf` | 2.47 GB | ✅ GGUF, direct |
| leejet Q2_K | leejet (sd.cpp author, 187k dl) | `qwen_image_2.1-Q2_K.gguf` | 2.56 GB | ✅ GGUF, direct |
| unsloth Turbo Q6_K | Unsloth Turbo (2.18M dl) | `qwen-image-2.1-turbo-Q6_K.gguf` | 6.27 GB | ✅ GGUF, direct, turbo sigmas |
| unsloth Turbo Q3_K_XL | Unsloth Turbo | `qwen-image-2.1-turbo-Q3_K_XL.gguf` | 3.61 GB | ✅ GGUF, direct, turbo sigmas |
| unsloth Turbo Q2_K | Unsloth Turbo | `qwen-image-2.1-turbo-Q2_K.gguf` | 2.47 GB | ✅ GGUF, direct, turbo sigmas |
| AtomicChat AD-Q4_K | AtomicChat (AD importance-matrix, 612k dl) | `Qwen-Image-2.1-Turbo-AD-Q4_K.gguf` | 4.20 GB | ✅ GGUF, direct, turbo sigmas. Vendor SSIM 0.850 vs BF16 (8 steps) |
| unsloth FP8 | Unsloth (673k dl) | `Qwen-Image-2.1-FP8.safetensors` | 7.12 GB | ⚠️ ATTEMPT direct load as `--diffusion-model` (sd.cpp loads safetensors; T86 verified FP8 files with sd.cpp). If the load fails, probe records the error — no retry in this series |

Dropped, with reason: IQ types (not in `sd-cli -M convert`'s supported
type list); INT4 ConvRot / Nunchaku / MNN / ONNX (not sd.cpp-loadable);
blaj OpenVINO INT4 (OV lane, not sd.cpp-loadable); ternary (no
Qwen-Image-2.1 ternary exists — Bonsai is FLUX.2 Klein); Abiray base
Q3_K_M (secondary vendor — OPEN if budget remains).

## Run plan

| # | Group | Probes (in order) | Status |
|---|---|---|---|
| D1 | own Q4 | own-q4-base, own-q4-stacked, q6-stacked-control | ✅ done — run 46076596744, success, 3/3 probes (see Results) |
| D2 | own Q3 | own-q3-base, own-q3-stacked, q6-stacked-control | ✅ done — run 46082424954, success, 3/3 probes (see Results) |
| D3 | own Q2 | own-q2-base, own-q2-stacked, q6-stacked-control | ✅ done — run 46090176956, success, 3/3 probes (see Results) |
| D4 | unsloth base | unsloth-q4-base, unsloth-q4-stacked, unsloth-q3xl-base, unsloth-q3xl-stacked | pending |
| D5 | deep 2-bit | unsloth-q2-base, unsloth-q2-stacked, leejet-q2-base, leejet-q2-stacked | pending |
| D6 | Turbo deep + FP8 | turbo-q6-easycache, turbo-adq4-easycache, turbo-q2-easycache, turbo-q3xl-easycache, fp8-base, fp8-stacked | pending |

Per-run time estimate: setup ~15–20 min (sd.cpp build + downloads; own
runs also merge 14 GB shards) + 3–4 gens. Own Q4/Q3/Q2 gens ~1200–1500 s
baseline, ~600–800 s stacked; Turbo gens ~150–400 s; FP8 ~1500–2500 s.
Driver enforces the 6000 s job budget and records skipped probes.

## Results (fill in per group)

### D1 — own Q4 (run 46076596744, success, 3/3, elapsed 2856.5s; dispatched 22:25 UTC, completed ~23:24 UTC)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| own-q4-base | 20/6.0 | 1155.6s | 0.6774 | 13.39 | clean, composition preserved (face/profile slightly different; SSIM understates similarity) |
| own-q4-stacked | 12/6.0 + EasyCache | 527.9s | 0.5709 | 11.59 | clean, composition preserved, darker/muted (s12 tonal cost) |
| q6-stacked-control | 12/6.0 + EasyCache | 564.0s | 0.5933 | 12.29 | clean, s12 look — SSIM byte-identical to R6's 681.2s control (fast host this run) |

Within-run: Q4-stacked is 6.4% faster than the Q6 stacked control on the same host (527.9 vs 564.0s). Q4 base (1155.6s) is only slightly faster than the Q6_K ref (1233.5s ladder) but drifts more than Q5_K_M did (SSIM 0.6774 vs 0.7191) — Q4's drift is tonal/detail, not a pose break. Q4 buys a little speed at stacked config, no speed at baseline.

### D2 — own Q3 (run 46082424954, success, 3/3, elapsed 4197.0s; dispatched 23:25 UTC, completed ~00:51 UTC)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| own-q3-base | 20/6.0 | 1959.8s | 0.4911 | 9.84 | coherent, composition preserved, but pose shifted (faces more forward, katana angle odd); drift is structural, not just tonal |
| own-q3-stacked | 12/6.0 + EasyCache | 876.1s | 0.5291 | 10.76 | clean, darker/muted s12 look, face in shadow |
| q6-stacked-control | 12/6.0 + EasyCache | 880.8s | 0.5933 | 12.29 | clean, s12 look — SSIM byte-identical to R6/D1 controls (slow host this run) |

Within-run: Q3-stacked is 0.5% faster than the Q6 control (876.1 vs 880.8s) — no speed gain. Q3 base is much SLOWER than Q6 base (1959.8s vs 1233.5s ladder) — dequant overhead outweighs memory savings at this size (same pattern as FLUX, where Q3_K was slower than Q4_K_M). Verdict: Q3 is a dead end for speed — slower at baseline, no gain stacked, and more pose drift than Q4.

### D3 — own Q2 (run 46090176956, success, 3/3, elapsed 3447.6s; dispatched 00:52 UTC, completed ~02:01 UTC)

| Probe | Steps/CFG | Wall | SSIM | PSNR | Eye verdict |
|---|---|---|---|---|---|
| own-q2-base | 20/6.0 | 1371.1s | 0.5236 | 10.28 | coherent (NOT blank, unlike FLUX Q2), composition preserved, pose drifted (forward-facing, katana angle odd) |
| own-q2-stacked | 12/6.0 + EasyCache | 615.7s | 0.5192 | 10.00 | clean, darker/muted s12 look, same pose drift as base |
| q6-stacked-control | 12/6.0 + EasyCache | 594.6s | 0.5933 | 12.29 | clean, s12 look — SSIM byte-identical to R6/D1/D2 controls |

Within-run: Q2-stacked is 3.5% SLOWER than the Q6 control (615.7 vs 594.6s). Q2 base (1371.1s) is slower than Q6 base (1233.5s) and Q4 base (1155.6s), faster only than Q3 base (1959.8s). Pattern across D1–D3: deep quants do NOT speed up Qwen on CPU — Q4 stacked +6%, Q3 flat, Q2 slower. Dequant cost dominates below Q6; only EasyCache + fewer steps move the needle.

- Lowest stable deep quant (eye-clean at baseline): TBD
- Fastest acceptable stacked combo: TBD
- Any third-party build beating our own quants: TBD

## OPEN / obstacles

- Placement DONE 2026-11-28 ~05:25 Saigon: user explicitly confirmed proceeding despite the fresh execution-risk flag on fetched f92257f ("Cứ chạy đi, tôi chịu"); rebased dq-deepquant onto f92257f, placed staged files, committed 15294ed, pushed to main, D1 dispatched and done.
- Host variance: only within-run deltas are trustworthy (identical base
  probe: 1233.5 s ladder vs 1565.9 s R1 vs 1477.5 s R6). Each of D1–D3
  carries its own Q6 stacked control for that reason; D4–D6 compare
  against those controls cross-run (caveat applies).
- FP8 direct load is the one genuinely uncertain probe: sd.cpp's
  safetensors loader for Qwen is documented in general terms only.
