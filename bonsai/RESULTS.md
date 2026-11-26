# DIY Bonsai-Klein: results (2026-11-26/27, runs 45858864209 + 45865137246)

## Verdict

**DIY Bonsai-Klein does NOT work as implemented: both TQ2 variants produce
pure noise, while the same-run Q4 control is a clean swordsman.** The
ternarizer is exonerated by the amax control — the failure is in the TQ2
path for Klein in sd.cpp (converter, kernel, or 2-bit being too lossy for
this architecture), not in the ternarization recipe.

**Speed: TQ2 is not faster than Q4 on x86 GitHub runners.** Within-run
per-diffusion-step: MSE run TQ2 40.8 vs Q4 42.7 s/step (tie); amax run TQ2
24.9 vs Q4 11.4 s/step (2.2x slower). Cross-run timings vary wildly by
host CPU, so only within-run comparisons count. The real speed lever is
the framework: sd.cpp Q4 (148–521s) vs ComfyUI Q4 (1200s baseline).

## Runs

| Run | Source | Ternarize | Convert | TQ2 gen | Q4 gen | TQ2 image |
|---|---|---|---|---|---|---|
| 45858864209 | mse-ternary | 355.7s, 258 tensors, 5.275B params (95.1%), mean MSE 0.00215 (amax 0.00712), zero 47.7% | 38s, 1818 MiB | wall 709s, sampling 652.9s, 16 steps, 0 skipped | wall 521s, sampling 470.2s, 5/16 skipped | pure noise (purple static) |
| 45865137246 | amax-from-bf16 | skipped | ~1 min, 1818 MiB | wall 429s, sampling 397.8s, 16 steps, 0 skipped | wall 148s, sampling 125.7s, 5/16 skipped | pure noise (brown static) |

Same prompt/seed/steps/CFG in all four generations: swordsman on misty
mountain path, 16 steps euler, CFG 1.0, seed 42, 512x512. Same Qwen3-VL
4B encoder + FLUX.2 VAE in every run. Q4 control in both runs is the same
clean swordsman (near-identical pixels).

## Why the wall clocks differ from the per-step story

EasyCache skipped 5/16 steps for Q4 in both runs and 0/16 for TQ2 in both
runs, so wall-clock TQ2-vs-Q4 comparisons are confounded. Per-step
diffusion time is the honest metric (above). The Q4 sampling time also
varied 3.7x between runs (470s vs 126s for the same 11 effective steps) —
GitHub runner host CPUs vary that much.

## Open questions / next steps

1. **Is the TQ2 file itself correct?** Inspect the converted GGUF's
   per-tensor types (gguf-dump / sd-cli info) to verify modulations and
   emb_in really stayed FP16 and the ternary tensors really are TQ2.
   The converter's `--tensor-type-rules` may have misapplied.
2. **Is TQ2 specifically broken, or is 2-bit too lossy for Klein?** Convert
   the same BF16 to Q3_K / Q2_K with the same converter and generate: if
   those work, TQ2 is the problem; if they also produce noise, sub-4-bit
   without error compensation is too lossy for this DiT.
3. **Are the ternary weights themselves at fault?** Load the
   MSE-ternarized safetensors directly in ComfyUI (original fallback):
   noise there too -> weights are the problem; degraded-but-real image ->
   the sd.cpp TQ2 path is the problem.
4. **What does a working ternary recipe look like?** Download PrismML's
   actual ternary weights (HF `prism-ml/bonsai-image-ternary-4B-gemlite-2bit`)
   and measure per-group scales / zero fraction vs this script's
   MSE-optimal scales. Their scale-fitting is undisclosed and may be
   activation-aware or error-compensated, which plain weight-only MSE is not.
5. **Do not chase TQ2 speed further on x86** until one of the above
   produces a real image: per-step TQ2 is at best tied with Q4, so even a
   working TQ2 would not beat the 1200s baseline by more than the sd.cpp
   framework switch already does.

## Files

- `bonsai/ternarize.py` — MSE-optimal ternarizer, unit-tested locally
  (synthetic weights: <=3 values per 256-block, GGML TQ2 amax round-trip
  identity OK, MSE 26% below amax on synth / 3.3x below on real weights).
- `.github/workflows/flux3-bonsai-test.yml` — workflow with
  `convert_source` input (`mse` | `amax`).
- `../results-bonsai/` — run 45858864209 artifact (MSE).
- `../results-bonsai-amax/` — run 45865137246 artifact (amax).
