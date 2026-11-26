# Bonsai-style ternarization of FLUX 3 Image Klein

DIY application of PrismML's Bonsai recipe to FLUX 3 Image Klein (5.6B DiT).
See the Bonsai Image 4B whitepaper (PrismML, May 2026) for the reference design.

## Recipe

- **Target:** matmul-heavy DiT linears only (`q/k/v_proj`, `attn_out`,
  `mlp_in`, `mlp_out` in every block) — 168 tensors, ~3.43B of 5.55B params.
  Everything else (embedders, modulation streams, time embeds, final layer,
  norms) stays FP16 — the same <5% exception tail as the whitepaper.
- **Format:** `w = s_g * t`, `t in {-1, 0, +1}`, one FP16 scale per group of
  **256** weights along the last dim (GGML's TQ2 block size `QK_K = 256`,
  not PrismML's 128 — the container dictates the grouping).
- **Scale fit:** the whitepaper does NOT disclose how `s_g` is fitted, so
  this script uses an **MSE-optimal grid search** per block
  (`s = c * mean|w|`, `c in [0.5, 2.0]`), which is strictly better MSE than
  GGML's native amax rule (`s = max|w|`). `--scale-rule amax` reproduces the
  converter-native behavior for comparison.
- **Why pre-ternarize instead of letting the converter do it:** the sd.cpp
  converter's TQ2 quantizer uses amax scaling, which is outlier-sensitive.
  Pre-ternarized values that are exact multiples of one per-block scale pass
  through that quantizer EXACTLY (verified in-script by a GGML
  quantize-round-trip identity check on sampled tensors).

## Files

- `ternarize.py` — the ternarizer (torch CPU, streams tensor-by-tensor).
  Output keeps the input safetensors layout (same names/shapes/dtypes).
- `../.github/workflows/flux3-bonsai-test.yml` — GitHub workflow:
  ternarize -> `sd-cli -M convert --type tq2_0` -> generate with sd.cpp
  (TQ2) -> same-prompt Q4_K_M comparison -> upload `results/`.

## Convert command (for reference)

```
sd-cli -M convert --diffusion-model flux3-image-klein-ternary.safetensors \
  --type tq2_0 --tensor-type-rules "modulations=f16,emb_in=f16" \
  -o flux3-klein-tq2.gguf
```

The rules keep the modulation streams and `emb_in` embedders in FP16;
`txt_in` / `time_in` / `ref_time_in` / `final_layer` / norm scales are
already skipped by the converter's built-in rules.
