# Qwen-Image-2.1 extra-speedup explore — REPORT (2026-11-27)

Baseline for every comparison: R1–R5 speed test (results-qwen21-speed/REPORT.md) —
best quality-preserving config **Q6_K 20 steps + EasyCache: 716.7s, SSIM 0.9242**
vs Q6_K 20-step reference `base-20.png` (same swordsman prompt, seed 42, 512²).
Turbo + EasyCache was fastest (367.6s) but reframes to a close-up — excluded.

Workflow: `.github/workflows/qwen21-openvino.yml`, driver `qwen21-ov/run_explore.py`.
Budget: max 4 runs. All 4 used. R6 (different workflow) ran concurrently and is
not covered here.

## Per-path results

| Path | Run | Status | Probe | Wall | SSIM vs ref | Eye verdict |
|---|---|---|---|---|---|---|
| OV-1 viggle | 46016329942 | ✅ success (608.6s elapsed) | viggle-base (6 steps, cfg 1.0) | 167.5s | 0.4371 (PSNR 9.46) | **Degraded** — framing kept (full-body, same composition), but face blurred/featureless, detail softer, katana barely visible |
| | | | viggle-easycache (6 steps + EasyCache) | 152.6s | 0.4537 (PSNR 9.64) | **Degraded, worse** — framing kept, blank face, heavy grain/noise across the whole image, robes a muddy blob |
| | | | lightning-base (4 steps, cfg 1.0) | 187.7s | 0.3771 (PSNR 9.07) | **Mostly clean** — framing kept, face + robes readable, cleanest of the three; lowest SSIM is pose/position shift (swordsman left, sun visible), not a reframe or breakage |
| OV-2 cache | 46018091029 | ❌ cancelled, **no data** | easycache + taylorseer + cache-dit + dbcache + spectrum + ucache (20 steps, Q6_K) | — | — | **No artifact, no logs** — see "What died" |
| OV-3 res384 | 46018110199 | ✅ success (3307.9s elapsed) | q4-easycache (Q4_K_M, 20 steps + EasyCache) | 820.3s (sampling 678.3s) | 0.6782 (PSNR 13.59) | **Clean / near-ref** — closest to the reference of all OV-3 probes, but SLOWER than the 716.7s best |
| | | | res384-lanczos (384² → 512² Lanczos) | 940.6s (sampling 824.7s) | 0.2385 (PSNR 7.24) | **Clean quality, REFRAMED** — tighter framing, swordsman fills the frame; also slower than baseline |
| | | | res384-lanczos-easycache | 563.5s (sampling 465.9s) | 0.2436 (PSNR 7.40) | **Clean quality, REFRAMED** — same tighter framing; −21% vs 716.7s cross-host (see host-variance caveat) |
| OV-4 openvino | 46018129474 | ❌ failure (~30s in) | blaj INT4 via OVDiffusionPipeline, 10 steps, speed-only | — | — | **Never ran** — see "What died" |

## Which new speedup actually helped?

**None, as a same-output speedup.** Nothing beats the 716.7s EasyCache best at
comparable quality:

- **Viggle Turbo / Lightning are ~4.3–4.7× faster (153–188s) but do not reproduce
  the reference image** (SSIM 0.38–0.45 vs 0.9242). They generate a *different*
  swordsman quickly, and Viggle visibly degrades the face. Unlike official Turbo
  they do **not** reframe to a close-up — framing is kept — but the output is a
  different image, so this is not a speedup of the same render. Lightning is the
  best-looking distilled output of the three and the only one I'd call usable on
  its own terms; Viggle (both variants) is not.
- **Q4_K_M + EasyCache is slower, not faster** (820.3s vs 716.7s; sampling
  678.3s). Lower DiT quant buys no CPU speed here (dequant cost) and drops SSIM
  to 0.6782. Dead end for speed.
- **384² + Lanczos upscale reframes.** Both res384 probes render a tighter
  composition (swordsman larger in frame) — SSIM ~0.24 is the framing change,
  not breakage; the images themselves look clean. res384+EasyCache is the only
  probe nominally faster than the best (563.5s, −21%), but cross-host times are
  unreliable (see caveat) and the reframe disqualifies the lane for
  same-composition output. Within-host, EasyCache at 384² does work: sampling
  825s → 466s (−44% vs plain 384² on the same host).
- **OpenVINO: untested** (died before the driver ran). **Cache sweep: untested**
  (timed out with no artifact).

## What died and why

- **OV-2 cache — killed by the 120-min workflow timeout, 0 artifacts.**
  Dispatched 14:40:52 UTC; driver step ran until cancelled ~16:40 UTC; job
  conclusion `cancelled`. 6 probes × ~700–1000s sampling on a slow host +
  ~15 min setup exceeded 120 min. The driver's 100-min `JOB_BUDGET_S` only
  stops *starting* new probes — it cannot save a probe already running when the
  workflow timeout fires. The driver saves results.json after each probe, so
  completed probes' PNGs + results.json existed on the runner, but the upload
  step was cut off by the same timeout before anything landed (artifact count 0).
  Run logs are unrecoverable (`gh run view --log`: "log not found"; jobs API:
  BlobNotFound 404), so **how many of the 6 probes finished is unknown**.
- **OV-4 openvino — pip dependency conflict in the workflow install step.**
  `Install OpenVINO stack` failed at 14:41:38 UTC: `ERROR: Cannot install
  optimum-intel==2.2.0 and transformers==5.15.1 ... ResolutionImpossible`.
  The driver never ran, so its nightly-stack retry was never reached either —
  the retry lives in the driver, the failure is one step earlier, in the
  workflow's pinned install. No artifact. Fix: a compatible pin set in
  `qwen21-openvino.yml` (relax/align the transformers + optimum-intel pins).

## Caveats (read before quoting times)

- **Host variance (~2×) makes cross-run wall-time comparisons unreliable.**
  OV-3's plain 384² sampling took 824.7s — slower than the 512² 716.7s best —
  which is impossible on equal hardware (384² is 56% of the pixels). Only
  within-run deltas are valid (e.g. 825s → 466s for EasyCache at 384² on the
  same host). A future run should include a same-host baseline probe
  (Q6_K + EasyCache, 20 steps) for calibration.
- **SSIM vs a single reference punishes any pose/framing change.** Lightning's
  0.3771 is the lowest SSIM in OV-1 yet it is the best-looking image there;
  res384's ~0.24 is a framing change, not breakage. Eye verdicts in the table
  above are the quality call; SSIM is the similarity-to-reference call.

## OPEN / obstacles

1. **OV-2 cache sweep has no data.** Whether taylorseer / cache-dit / dbcache /
   spectrum / ucache beat EasyCache is still unanswered. A re-run needs either
   fewer probes per run (e.g. 3 + 3) or a longer workflow timeout — and a new
   run approval, since the 4-run budget is used.
2. **OV-4 OpenVINO untested.** Needs a fixed pin set in the workflow and a new
   run approval (5th run). The blaj INT4 lane (12.9s was Arc iGPU, not CPU)
   remains the only untested lane with a claimed order-of-magnitude gap.
3. **No same-output speedup found in this exploration.** The 716.7s EasyCache
   best stands. The only faster lanes either change the image (Viggle/Lightning
   distillation, res384 reframe) or are untested (OV-2/OV-4).
4. **Timeout hardening for future multi-probe runs:** the upload step needs
   `if: always()` (or per-probe artifact uploads) so a timeout cannot zero out
   completed probes the way it did OV-2; results.json is already saved per
   probe, only the upload is fragile.

## Files

- `results-qwen21-openvino/RUNS.md` — run tracker with IDs + final statuses.
- `ov1/` — OV-1 artifact: results.json, viggle-base.png,
  viggle-easycache.png, lightning-base.png + sd-cli logs.
- `ov3/` — OV-3 artifact: results.json, q4-easycache.png,
  res384-lanczos.png, res384-lanczos-easycache.png (+ raw 384² PNGs) + logs.
- OV-2: no artifact. OV-4: no artifact.
