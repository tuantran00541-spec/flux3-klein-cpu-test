"""FastSD CPU OpenVINO INT4 FLUX 3 Image Klein test driver.

Drives rupeshs/fastsdcpu's self-contained OpenVINO pipeline
(ov_flux3_image_klein_pipeline.py, cloned to fastsdcpu-src/ by the workflow)
directly, without the full FastSD app stack (no torch, no diffusers, no Qt):
only openvino + transformers (tokenizer) + huggingface_hub + numpy + Pillow.

Tests, same prompt/seed as runs 1-2 (client.py):
  1. t2i-512   T2I 512x512, 4 steps, seed 42 (core)
  2. t2i-1024  T2I 1024x1024, 4 steps, seed 42 (bonus)
  3. edit-sfx  reference edit of assets/sfx-panel.png, EDIT_PROMPT, seed 42 (bonus)

Timing reported separately: model load (download + tokenizer + OV compile),
per-call wall, and per-diffusion-step timings from pipe.timings["steps"].
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join("fastsdcpu-src", "src", "backend", "openvino"))

from PIL import Image  # noqa: E402

MODEL_ID = "rupeshs/flux3-image-klein-int4-ov"

# Verbatim from client.py (runs 1-2).
T2I_PROMPT = (
    "A detailed anime-style illustration of a young swordsman standing on a misty "
    "mountain path at dawn, long black hair tied back, dark blue robes with silver "
    "trim, holding a sheathed katana at his side, pine trees and drifting fog in "
    "the background, soft golden morning light breaking through the mist, "
    "cinematic composition, clean linework, detailed shading."
)
EDIT_PROMPT = (
    "A clean anime manga panel of a fierce demon god character roaring, sharp teeth "
    "bared, wild spiky hair, dramatic speed lines and heavy black shading around "
    "him, the speech bubble above him completely empty and blank white with no "
    "text, no letters and no sound effects anywhere in the image, the character "
    "artwork, pose and framing unchanged."
)

SEED = 42
STEPS = 4  # flux3_ov.json default for this model (4-step distill LoRA merged)

OUT_DIR = None
LOG_F = None


def log(msg):
    line = "[%6.1fs] %s" % (time.time() - T_START, msg)
    print(line, flush=True)
    if LOG_F:
        LOG_F.write(line + "\n")
        LOG_F.flush()


T_START = time.time()


def gray_stats(path):
    im = Image.open(path).convert("L")
    px = list(im.getdata())
    return {
        "min": min(px),
        "max": max(px),
        "mean": round(sum(px) / len(px), 1),
        "size": list(Image.open(path).size),
        "bytes": os.path.getsize(path),
    }


def run_test(pipe, name, out_dir, **call_kwargs):
    rec = {"name": name, "status": "failed", "wall_seconds": None, "error": None,
           "outputs": [], "steps": STEPS, "seed": SEED}
    t0 = time.time()
    try:
        out = pipe(**call_kwargs)
        img = out.images[0]
        fname = "fastsdcpu-%s.png" % name
        fpath = os.path.join(out_dir, fname)
        img.save(fpath)
        rec["wall_seconds"] = round(time.time() - t0, 1)
        rec["step_seconds"] = [round(s, 2) for s in pipe.timings.get("steps", [])]
        rec["outputs"] = [{"file": fname, **gray_stats(fpath)}]
        rec["status"] = "success"
        log("%s SUCCESS wall=%.1fs steps=%s out=%s" % (
            name, rec["wall_seconds"], rec["step_seconds"], fname))
    except Exception as e:
        rec["wall_seconds"] = round(time.time() - t0, 1)
        rec["error"] = str(e)[:1000]
        log("%s FAILED after %.1fs: %s" % (name, rec["wall_seconds"], rec["error"]))
    return rec


def main():
    global OUT_DIR, LOG_F
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results-fastsdcpu")
    ap.add_argument("--run-edit", default="true")
    ap.add_argument("--run-1024", default="true")
    args = ap.parse_args()
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    LOG_F = open(os.path.join(OUT_DIR, "driver.log"), "w")

    import openvino as ov
    import transformers

    log("openvino %s, transformers %s" % (ov.__version__, transformers.__version__))
    log("model: %s" % MODEL_ID)

    from ov_flux3_image_klein_pipeline import OVFlux3KleinPipeline

    t0 = time.time()
    try:
        pipe = OVFlux3KleinPipeline.from_pretrained(MODEL_ID, device="CPU")
    except Exception as e:
        log("LOAD FAILED after %.1fs: %s" % (time.time() - t0, e))
        results = {"model": MODEL_ID, "load_seconds": round(time.time() - t0, 1),
                   "load_error": str(e)[:1000], "tests": []}
        json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
        LOG_F.close()
        return 1
    load_s = round(time.time() - t0, 1)
    log("LOAD OK in %.1fs (download + tokenizer + OV compile)" % load_s)
    log("config steps=%s shift=%s" % (pipe.config.get("steps"), pipe.config.get("shift")))

    results = {"model": MODEL_ID, "load_seconds": load_s,
               "baseline_sd_cpp_q4_512_12step_s": 199.1,
               "baseline_sd_cpp_q4_512_16step_s": 241.1,
               "run1_comfyui_q4_512_s": 1200.2,
               "tests": []}

    # 1. Core: T2I 512.
    results["tests"].append(run_test(
        pipe, "t2i-512", OUT_DIR, prompt=T2I_PROMPT,
        width=512, height=512, num_inference_steps=STEPS, seed=SEED))

    # 2. Bonus: T2I 1024.
    if args.run_1024 == "true":
        results["tests"].append(run_test(
            pipe, "t2i-1024", OUT_DIR, prompt=T2I_PROMPT,
            width=1024, height=1024, num_inference_steps=STEPS, seed=SEED))
    else:
        results["tests"].append({"name": "t2i-1024", "status": "skipped",
                                 "wall_seconds": None, "error": None, "outputs": []})

    # 3. Bonus: SFX-panel reference edit.
    if args.run_edit == "true":
        panel = "assets/sfx-panel.png"
        if os.path.exists(panel):
            ref = Image.open(panel)
            log("edit ref %s size=%s mode=%s" % (panel, ref.size, ref.mode))
            results["tests"].append(run_test(
                pipe, "edit-sfx", OUT_DIR, prompt=EDIT_PROMPT,
                image=ref, num_inference_steps=STEPS, seed=SEED))
        else:
            results["tests"].append({"name": "edit-sfx", "status": "skipped",
                                     "wall_seconds": None,
                                     "error": "assets/sfx-panel.png missing",
                                     "outputs": []})
    else:
        results["tests"].append({"name": "edit-sfx", "status": "skipped",
                                 "wall_seconds": None, "error": None, "outputs": []})

    json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
    log("done, total wall %.1fs" % (time.time() - T_START))
    LOG_F.close()
    core_ok = any(t["name"] == "t2i-512" and t["status"] == "success"
                  for t in results["tests"])
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
