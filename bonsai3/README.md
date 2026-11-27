# Bonsai round 3 — ternary DIRECTLY from base, own TQ2G128 writer

User order (2026-11-27): keep compressing Bonsai FROM THE BASE.
Compressing from Q2 is dead from the egg — double quantization
(base -> Q2 -> ternary) compounds the error. So: ternary scales are
computed from the full-precision BF16 weights only, and the GGUF is
written by our own writer, never through sd-cli's TQ2 converter.

## What rounds 1-2 proved (and what round 3 corrected)

- Round 1 fed sd-cli the BF16 checkpoint DIRECTLY (no double
  quantization) and still got pure noise; fix1's F16 ternary (no packing
  at all) was also noise. Ternary at group 256 kills this DiT outright.
- The fix2 "scale-byte" story was WRONG: the coordinator's TQ2 dequant
  read the scale at the wrong offset. The real layout (proven below)
  has the scale LAST, and the sd-cli TQ2 file is probably fine.
- Group 128 was never imaged (fix2's group-128 F16 bonus timed out
  at step 14/16). Round 3 tests group 128 first.

## The vendor format, SOLVED (not guessed)

PrismML/TokForge TQ2G128 (sd.cpp GGML type 35), proven byte-exact
against the TokForge GGUF + Prism's unpacked model (rel-L2 = 0.0 on
216 x 128-groups of qkv and on proj; scales match the unpacked amax):

- 68-byte block = 2 groups of 128 elements.
- bytes [0:32] = group-0 codes, [32:64] = group-1 codes,
  [64:66] = d0 (FP16 LE), [66:68] = d1 (FP16 LE).
- codes: GGML TQ2_0-style INTERLEAVED packing: byte j holds weights
  j, j+32, j+64, j+96 at shifts 0, 2, 4, 6; value = d * (code - 1).
- scale = per-128-group amax (FP16).
- density 0.265625 B/elem (NOT GGML TQ2_0's 0.2578125, 66 B/256).

Note: in sd.cpp's ggml, enum 35 is this 68-byte format. In current
upstream ggml, 35 is TQ2_0 (66 B/256) -- do NOT hardcode the upstream
enum. The writer below uses the format as measured, and the workflow
also probes what the runner's sd-cli `--type tq2_0` actually produces.

## Files

- `ternarize.py` -- round-1 recipe, group default 128. MSE grid search
  (s = c*mean|w|) or amax, directly from the base BF16 safetensors.
  Output: ternary BF16 safetensors (same layout).
- `pack_gguf.py` -- the own writer. Reads an F16 GGUF (sd-cli fusion
  only, no quantization), packs the 80 TokForge type-35 tensors into
  68-byte blocks, copies the rest byte-identical. Verifies every packed
  tensor: dequant(pack(w)) == w exactly (rel-L2 ~ 0) and no code == 3.
  Fails loudly otherwise. Validated locally: pack(unpacked qkv rows)
  == TokForge bytes exactly; full-layout offsets == TokForge's.
- `verify_gguf.py` -- dequantize a type-35 GGUF against a reference
  F16 GGUF (per-tensor rel-L2 + B/elem). Used for the sd-cli tq2_0
  diagnostic, NOT for our own output (the packer verifies itself).
- `tokforge-layout.json` -- the TokForge GGUF's 149 tensors
  (names/dims/types/offsets): 80 type-35 targets, 69 type-30 exceptions.

## Run 1 (workflow `flux3-bonsai3-test.yml`)

1. Base BF16 (distilled, same as rounds 1-2) -> ternarize g128 MSE + AMAX.
2. sd-cli convert -> F16 GGUFs (fusion only).
3. Own writer -> type-35 GGUFs (MSE + AMAX), verified rel-L2 = 0.
4. Diagnostic: sd-cli `--type tq2_0` on the ternary F16 + verify
   (B/elem + rel-L2 with the CORRECT 68-byte dequant).
5. Inference (same prompt/seed/steps/CFG, 512^2, 16 steps, euler):
   Q4_K_M control, type-35 MSE, type-35 AMAX, F16 MSE, F16 AMAX
   (in that priority order, each guarded by elapsed time).

Checkpoint note: "base" in the user's order means the full-precision
weights (as opposed to a quantized intermediate), not the 50-step
base-model variant. Round 3 uses the distilled BF16 safetensors, the
same checkpoint as rounds 1-2 and the same model as the Q4 control.
