# flux3-klein-cpu-test

Scratch repo: prove FLUX 3 Image Klein runs on a GitHub-hosted runner (CPU-only, no GPU).

- Model: FLUX 3 Image Klein, GGUF Q4_K_M DiT (3.05 GiB) + Heretic DHQ Q4_K_M
  Qwen3-VL-4B text encoder (2.66 GB) + mmproj (0.84 GB) + FLUX2 VAE.
  Served through ComfyUI's HTTP API at http://127.0.0.1:8188.
- Client: `client.py` reads `BASE_URL` and `API_KEY` env vars
  (`API_KEY=none` — ComfyUI has no auth; the value is a placeholder only).
  It runs a speed matrix (same prompt, 512x512, 16 steps): ComfyUI torch
  CPU + Q4_K_M baseline, stable-diffusion.cpp GGML + Q4_K_M, ComfyUI + Q8_0,
  thread tuning (2 vs 4 threads), then the SFX-panel edit test and an
  optional 1024x1024 test. Every config is one entry in `results.json`.
- Workflow: `.github/workflows/flux3-cpu-test.yml` (workflow_dispatch).
- `assets/sfx-panel.png`: manga test panel used for the image-edit test.

No secrets anywhere in this repo. Public, $0 (GitHub-hosted runners).
