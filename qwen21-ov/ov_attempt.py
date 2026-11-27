#!/usr/bin/env python3
"""OpenVINO INT4 CPU speed probe for Qwen-Image-2.1 (no export).

Run as a subprocess by run_explore.py --path openvino, once per stack
(released, then nightly if the first fails). Loads the pre-converted
community INT4 build via optimum-intel OVDiffusionPipeline on CPU and
generates ONE 512^2 / 10-step image with the standard prompt + seed 42.

Writes ov-int4-cpu.png + timing info into the results dir (via a small
JSON sidecar the parent merges), and prints a machine-readable tail.
Exit 0 only if an image was generated.

SPEED-ONLY probe: the build's text encoder is abliterated (differs from
stock), so SSIM vs the stock reference is NOT a fair quality comparison.
The parent flags that in results.json.
"""
import argparse
import json
import os
import sys
import time

OV_REPO = "blaj/Qwen-Image-2.1-Uncensored-OpenVINO-INT4"
PROMPT = (
    "A detailed anime-style illustration of a young swordsman standing on a misty "
    "mountain path at dawn, long black hair tied back, dark blue robes with silver "
    "trim, holding a sheathed katana at his side, pine trees and drifting fog in "
    "the background, soft golden morning light breaking through the mist, "
    "cinematic composition, clean linework, detailed shading."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    stack = os.environ.get("OV_STACK", "released")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    import torch
    from optimum.intel import OVDiffusionPipeline

    log = lambda m: print("[ov-attempt %s %.0fs] %s" % (stack, time.time() - t0, m),
                          flush=True)
    try:
        import openvino as ov
        log("openvino %s" % ov.__version__)
    except Exception:
        pass
    try:
        import optimum.intel as oi
        log("optimum-intel %s" % getattr(oi, "__version__", "?"))
    except Exception:
        pass
    try:
        import diffusers
        log("diffusers %s" % diffusers.__version__)
    except Exception:
        pass

    log("from_pretrained(%s, device=CPU, compile=True) ..." % OV_REPO)
    t_load = time.time()
    pipe = OVDiffusionPipeline.from_pretrained(
        OV_REPO, device="CPU", compile=True)
    load_s = round(time.time() - t_load, 1)
    log("pipeline loaded+compiled in %.1fs" % load_s)

    log("generating 512x512 / 10 steps / seed 42 ...")
    t_gen = time.time()
    img = pipe(
        PROMPT,
        num_inference_steps=10, height=512, width=512,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
    gen_s = round(time.time() - t_gen, 1)
    out_png = os.path.join(args.out, "ov-int4-cpu.png")
    img.save(out_png)
    log("generated in %.1fs (%.1fs/step) -> %s (%d bytes)"
        % (gen_s, gen_s / 10.0, out_png, os.path.getsize(out_png)))

    # SSIM vs ref (flagged non-comparable in the parent record).
    ssim = psnr = None
    if args.ref and os.path.exists(args.ref):
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "qwen21-ladder"))
        import run_ladder as rl
        try:
            m = rl.compute_metrics(args.ref, out_png)
            ssim, psnr = m["ssim_vs_ref"], m["psnr_vs_ref_db"]
            log("SSIM vs stock ref (NON-COMPARABLE, abliterated TE): %.4f PSNR %.2f"
                % (ssim, psnr))
        except Exception as e:
            log("metrics failed: %s" % e)

    rec = {"name": "ov-int4-cpu", "status": "success",
           "stack": stack, "steps": 10, "cfg": None, "threads": None,
           "load_seconds": load_s, "wall_seconds": gen_s,
           "seconds_per_step": round(gen_s / 10.0, 1),
           "ssim_vs_ref": ssim, "psnr_vs_ref_db": psnr,
           "quality_note": "SPEED-ONLY probe: text encoder is abliterated, "
                           "differs from stock - SSIM is NOT a fair quality "
                           "comparison",
           "dit": OV_REPO + " (INT4 IR, transformer stock, TE abliterated)",
           "outputs": [{"file": os.path.basename(out_png),
                        "bytes": os.path.getsize(out_png)}],
           "error": None}
    with open(os.path.join(args.out, "ov-attempt-%s.json" % stack), "w") as f:
        json.dump(rec, f, indent=2)
    print("OV_RESULT " + json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
