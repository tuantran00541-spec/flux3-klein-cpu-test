# FastSD CPU OpenVINO INT4 test for FLUX 3 Image Klein

Tests rupeshs/fastsdcpu's ready-made OpenVINO INT4 FLUX.3 Klein
(`rupeshs/flux3-image-klein-int4-ov`, 5.6 GB, 4-step distill LoRA merged)
on a GitHub runner, against the sd.cpp baseline from run 2
(Q4 512²: 241 s @ 16 steps / 199 s @ 12 steps).

## How it works

The FastSD app stack is NOT installed (no torch, no diffusers, no Qt).
The workflow clones fastsdcpu and drives its self-contained pipeline
module `src/backend/openvino/ov_flux3_image_klein_pipeline.py` directly
via `fastsdcpu/run_fastsdcpu.py` — the module only needs openvino +
transformers (tokenizer) + huggingface_hub + numpy + Pillow + tqdm.

## Tests (same prompt + seed 42 as runs 1-2)

1. `t2i-512` — T2I 512×512, 4 steps, seed 42 (core)
2. `t2i-1024` — T2I 1024×1024, 4 steps, seed 42 (bonus)
3. `edit-sfx` — reference edit of `assets/sfx-panel.png` (bonus)

Timing is split: model load (HF download + tokenizer + OV compile)
vs per-call wall vs per-diffusion-step timings from `pipe.timings`.

## Dispatch

`gh workflow run flux3-fastsdcpu-test.yml` (120-min job cap).
Artifacts: `results-fastsdcpu/` (PNGs + `results.json` + `driver.log`).
