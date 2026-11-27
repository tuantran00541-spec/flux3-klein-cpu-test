"""FastSD CPU OpenVINO INT4 FLUX 3 Image Klein — EDIT FIDELITY optimization.

The 77.5s SFX edit (run 45912610080) removes all text but re-renders the
whole panel: locally measured SSIM 0.0956 / PSNR 9.83 dB vs the original on
NON-text areas. This driver tries, on the same 172x461 SFX panel, same
EDIT_PROMPT, seed 42:

  0. introspect OVFlux3KleinPipeline.__call__ (strength / mask params?)
  1. edit-base      baseline re-run, 4 steps (reference numbers)
  2. edit-maskcomp  mask-composite: base edit inside text boxes, ORIGINAL
                    pixels outside (feathered) — pixel-identical outside
                    text by construction
  3. edit-s0.3/0.5/0.7  strength sweep, ONLY if the pipeline accepts a
                    strength-like param (else recorded as unsupported)
  4. edit-8step     8 steps at the best strength (or default if none)

Metrics per config: SSIM + PSNR vs the original on non-text pixels only
(same functions as /tmp/metrics.py, inlined here — pure numpy, no extra deps).
"""

import argparse
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.join("fastsdcpu-src", "src", "backend", "openvino"))

import numpy as np
from PIL import Image, ImageFilter  # noqa: E402

MODEL_ID = "rupeshs/flux3-image-klein-int4-ov"

# Verbatim from fastsdcpu/run_fastsdcpu.py (runs 1-2 prompt).
EDIT_PROMPT = (
    "A clean anime manga panel of a fierce demon god character roaring, sharp teeth "
    "bared, wild spiky hair, dramatic speed lines and heavy black shading around "
    "him, the speech bubble above him completely empty and blank white with no "
    "text, no letters and no sound effects anywhere in the image, the character "
    "artwork, pose and framing unchanged."
)

SEED = 42
STEPS = 4

# Text boxes as fractions of (W, H) for the 172x461 SFX panel.
TEXT_BOXES = [
    (0.09, 0.96, 0.02, 0.29),  # KWAAAAH bubble
    (0.00, 0.20, 0.28, 0.38),  # left red OH
    (0.81, 1.00, 0.40, 0.51),  # right red OH
    (0.03, 0.96, 0.72, 0.99),  # DEMON GOD bubble
]

OUT_DIR = None
LOG_F = None
T_START = time.time()


def log(msg):
    line = "[%6.1fs] %s" % (time.time() - T_START, msg)
    print(line, flush=True)
    if LOG_F:
        LOG_F.write(line + "\n")
        LOG_F.flush()


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


# ---- metrics (non-text areas only) ---------------------------------------

def _text_mask(w, h, dilate_px=5):
    m = np.zeros((h, w), dtype=bool)
    for (x0, x1, y0, y1) in TEXT_BOXES:
        xa = max(0, int(x0 * w) - dilate_px)
        xb = min(w, int(x1 * w) + dilate_px)
        ya = max(0, int(y0 * h) - dilate_px)
        yb = min(h, int(y1 * h) + dilate_px)
        m[ya:yb, xa:xb] = True
    return m


def _uniform_blur(a, k=11):
    pad = k // 2
    ap = np.pad(a, pad, mode="reflect")
    cs = np.cumsum(np.cumsum(ap, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = a.shape
    return (cs[k:k + h, k:k + w] - cs[:h, k:k + w]
            - cs[k:k + h, :w] + cs[:h, :w]) / float(k * k)


def _ssim_map(a, b):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1, mu2 = _uniform_blur(a), _uniform_blur(b)
    mu1sq, mu2sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1sq = _uniform_blur(a * a) - mu1sq
    s2sq = _uniform_blur(b * b) - mu2sq
    s12 = _uniform_blur(a * b) - mu12
    return ((2 * mu12 + c1) * (2 * s12 + c2)
            / ((mu1sq + mu2sq + c1) * (s1sq + s2sq + c2)))


def compute_metrics(orig_arr, edit_arr):
    """SSIM + PSNR of edit vs original on NON-text pixels only."""
    h, w = orig_arr.shape[:2]
    eval_mask = ~_text_mask(w, h, dilate_px=5 + 3)
    og = orig_arr.mean(axis=2)
    eg = edit_arr.mean(axis=2)
    ssim = float(_ssim_map(og, eg)[eval_mask].mean())
    sel = np.repeat(eval_mask[:, :, None], 3, axis=2)
    mse = float(((orig_arr - edit_arr) ** 2)[sel].mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return {"ssim_nontext": round(ssim, 4), "psnr_nontext_db": round(psnr, 2)}


def mask_composite(orig_img, edit_img, dilate_px=5, feather_px=3):
    """Edited pixels inside text boxes, ORIGINAL pixels outside (feathered)."""
    orig = np.asarray(orig_img.convert("RGB")).astype(np.float64)
    edit = np.asarray(edit_img.convert("RGB")).astype(np.float64)
    h, w = orig.shape[:2]
    m = np.zeros((h, w), dtype=np.float64)
    for (x0, x1, y0, y1) in TEXT_BOXES:
        xa = max(0, int(x0 * w) - dilate_px)
        xb = min(w, int(x1 * w) + dilate_px)
        ya = max(0, int(y0 * h) - dilate_px)
        yb = min(h, int(y1 * h) + dilate_px)
        m[ya:yb, xa:xb] = 1.0
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(feather_px))).astype(np.float64) / 255.0
    out = edit * m[:, :, None] + orig * (1.0 - m[:, :, None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---- test plumbing ---------------------------------------------------------

def run_edit(pipe, name, out_dir, ref, orig_arr, extra_kwargs=None, steps=STEPS):
    rec = {"name": name, "status": "failed", "wall_seconds": None, "error": None,
           "outputs": [], "steps": steps, "seed": SEED,
           "kwargs": dict(extra_kwargs or {})}
    t0 = time.time()
    try:
        kwargs = {"prompt": EDIT_PROMPT, "image": ref,
                  "num_inference_steps": steps, "seed": SEED}
        kwargs.update(extra_kwargs or {})
        out = pipe(**kwargs)
        img = out.images[0]
        fname = "fastsdcpu-%s.png" % name
        fpath = os.path.join(out_dir, fname)
        img.save(fpath)
        rec["wall_seconds"] = round(time.time() - t0, 1)
        rec["step_seconds"] = [round(s, 2) for s in pipe.timings.get("steps", [])]
        rec["outputs"] = [{"file": fname, **gray_stats(fpath)}]
        edit_arr = np.asarray(img.convert("RGB")).astype(np.float64)
        rec["metrics"] = compute_metrics(orig_arr, edit_arr)
        rec["status"] = "success"
        log("%s SUCCESS wall=%.1fs metrics=%s out=%s" % (
            name, rec["wall_seconds"], rec["metrics"], fname))
    except Exception as e:
        rec["wall_seconds"] = round(time.time() - t0, 1)
        rec["error"] = str(e)[:1000]
        log("%s FAILED after %.1fs: %s" % (name, rec["wall_seconds"], rec["error"]))
    return rec


def main():
    global OUT_DIR, LOG_F
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results-fastsdcpu-opt")
    args = ap.parse_args()
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)
    LOG_F = open(os.path.join(OUT_DIR, "driver.log"), "w")

    import openvino as ov
    import transformers

    log("openvino %s, transformers %s" % (ov.__version__, transformers.__version__))
    log("model: %s" % MODEL_ID)

    from ov_flux3_image_klein_pipeline import OVFlux3KleinPipeline

    # Step 0: introspect the pipeline call signature.
    try:
        sig = str(inspect.signature(OVFlux3KleinPipeline.__call__))
    except Exception as e:
        sig = "introspection failed: %s" % e
    log("pipeline __call__ signature: %s" % sig)
    sig_lower = sig.lower()
    strength_param = None
    for cand in ("strength", "denoising_strength", "denoise", "image_strength"):
        if cand in sig_lower:
            strength_param = cand
            break
    mask_param = None
    for cand in ("mask_image", "mask"):
        if cand in sig_lower:
            mask_param = cand
            break
    log("strength param: %s | mask param: %s" % (strength_param, mask_param))

    panel = "assets/sfx-panel.png"
    if not os.path.exists(panel):
        log("MISSING %s — cannot run edits" % panel)
        json.dump({"model": MODEL_ID, "error": "panel missing", "tests": []},
                  open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
        LOG_F.close()
        return 1
    ref = Image.open(panel)
    orig_arr = np.asarray(ref.convert("RGB")).astype(np.float64)
    log("edit ref %s size=%s mode=%s" % (panel, ref.size, ref.mode))

    t0 = time.time()
    try:
        pipe = OVFlux3KleinPipeline.from_pretrained(MODEL_ID, device="CPU")
    except Exception as e:
        log("LOAD FAILED after %.1fs: %s" % (time.time() - t0, e))
        json.dump({"model": MODEL_ID, "load_seconds": round(time.time() - t0, 1),
                   "load_error": str(e)[:1000], "tests": []},
                  open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
        LOG_F.close()
        return 1
    load_s = round(time.time() - t0, 1)
    log("LOAD OK in %.1fs" % load_s)
    log("config steps=%s shift=%s" % (pipe.config.get("steps"), pipe.config.get("shift")))

    results = {"model": MODEL_ID, "load_seconds": load_s,
               "pipeline_signature": sig,
               "strength_param": strength_param,
               "mask_param": mask_param,
               "prior_edit_ssim_nontext": 0.0956,
               "prior_edit_psnr_nontext_db": 9.83,
               "tests": []}

    # 1. Baseline re-run, 4 steps.
    base = run_edit(pipe, "edit-base", OUT_DIR, ref, orig_arr)
    results["tests"].append(base)

    # 2. Mask-composite from the baseline output (no extra diffusion).
    t0 = time.time()
    comp_rec = {"name": "edit-maskcomp", "status": "failed", "wall_seconds": None,
                "error": None, "outputs": [], "steps": STEPS, "seed": SEED,
                "kwargs": {"composite": "base-edit inside text boxes, "
                           "original pixels outside, feather 3px"}}
    if base["status"] == "success":
        try:
            base_img = Image.open(os.path.join(
                OUT_DIR, "fastsdcpu-edit-base.png"))
            comp_img = mask_composite(ref, base_img)
            fpath = os.path.join(OUT_DIR, "fastsdcpu-edit-maskcomp.png")
            comp_img.save(fpath)
            comp_rec["wall_seconds"] = round(time.time() - t0, 1)
            comp_rec["outputs"] = [{"file": "fastsdcpu-edit-maskcomp.png",
                                    **gray_stats(fpath)}]
            comp_rec["metrics"] = compute_metrics(
                orig_arr, np.asarray(comp_img.convert("RGB")).astype(np.float64))
            comp_rec["status"] = "success"
            log("edit-maskcomp SUCCESS wall=%.1fs metrics=%s" % (
                comp_rec["wall_seconds"], comp_rec["metrics"]))
        except Exception as e:
            comp_rec["wall_seconds"] = round(time.time() - t0, 1)
            comp_rec["error"] = str(e)[:1000]
            log("edit-maskcomp FAILED: %s" % comp_rec["error"])
    else:
        comp_rec["error"] = "base edit failed, nothing to composite from"
        log("edit-maskcomp SKIPPED: base edit failed")
    results["tests"].append(comp_rec)

    # 3. Strength sweep, only if the pipeline accepts a strength param.
    best_strength = None
    if strength_param:
        for s in (0.3, 0.5, 0.7):
            rec = run_edit(pipe, "edit-s%.1f" % s, OUT_DIR, ref, orig_arr,
                           extra_kwargs={strength_param: s})
            results["tests"].append(rec)
            if rec["status"] == "success" and best_strength is None:
                best_strength = s
    else:
        results["tests"].append({"name": "edit-strength-sweep", "status": "skipped",
                                 "wall_seconds": None,
                                 "error": "no strength/denoise param in "
                                 "pipeline __call__ signature",
                                 "outputs": []})
        log("strength sweep SKIPPED: no strength param")

    # 4. 8 steps at the best strength (or default if none).
    extra = {strength_param: best_strength} if (strength_param
                                                and best_strength is not None) else None
    results["tests"].append(run_edit(
        pipe, "edit-8step", OUT_DIR, ref, orig_arr,
        extra_kwargs=extra, steps=8))

    json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
    log("done, total wall %.1fs" % (time.time() - T_START))
    LOG_F.close()
    ok = any(t["status"] == "success" for t in results["tests"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
