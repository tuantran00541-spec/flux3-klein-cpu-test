#!/usr/bin/env python3
"""Staged FLUX 3 Klein quant ladder: base reference -> gradual compression -> speed.

Stage 0 (base reference): generate the BASE checkpoint (BF16) at 50 steps
CFG 3.5 (the true reference) and at the ladder steps/CFG (the metric
reference). Base-at-ladder-steps is generated FIRST so the ladder reference
exists even if the 50-step generation times out.
Stage ladder: convert BF16 -> each level (one at a time), generate at fixed
steps/CFG, SSIM/PSNR vs the metric reference, delete the GGUF after scoring.
Stage speed: optimize the winning level (threads, steps, EasyCache) one
variable at a time, metrics vs the same reference.

Same prompt/seed everywhere (verbatim from client.py). No torch needed:
sd.cpp only, plus numpy/Pillow for metrics.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# --- constants (verbatim from client.py / prior runs) ---
T2I_PROMPT = (
    "A detailed anime-style illustration of a young swordsman standing on a misty "
    "mountain path at dawn, long black hair tied back, dark blue robes with silver "
    "trim, holding a sheathed katana at his side, pine trees and drifting fog in "
    "the background, soft golden morning light breaking through the mist, "
    "cinematic composition, clean linework, detailed shading."
)
BASE_REPO = "ml-resources/flux-3-image-klein-comfy"
BASE_FILE = "diffusion_models/flux3-image-klein-base-bf16.safetensors"
TE_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
TE_FILE = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
TE_VISION = "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"
VAE_FILE = "vae/flux2-vae.safetensors"
SDCPP_BIN = "sd.cpp/build/bin/sd-cli"
MODELS = "models"
RULES = "modulations=f16,emb_in=f16"
SEED = 42
JOB_BUDGET_S = 6000  # stop starting new levels after this (leave time to upload)

# Level -> convert types to try in order (first success wins; a level whose
# types all fail is recorded as skipped, never fatal). _m variants are the
# llama-style mixed quants; plain K is the uniform fallback. Both cases are
# tried because the help lists q2_K/q3_K/q4_K (capital K) while prior jobs
# used lowercase; failed parses exit in seconds, so the fallback is cheap.
LEVEL_TYPES = {
    "fp8": ["f8_e4m3"],
    "q8_0": ["q8_0"],
    "q6_k": ["q6_k", "q6_K"],
    "q5_k_m": ["q5_k_m", "q5_K_M", "q5_k", "q5_K"],
    "q4_k_m": ["q4_k_m", "q4_K_M", "q4_k", "q4_K"],
    "q3_k": ["q3_k", "q3_K"],
    "q2_k": ["q2_k", "q2_K"],
    "tq2": ["tq2_0"],
}

T_START = time.time()
SDCLI_HELP = ""


def elapsed():
    return time.time() - T_START


def log(msg):
    print("[ladder %.0fs] %s" % (elapsed(), msg), flush=True)


# ---- metrics (numpy-only SSIM, same implementation as fastsdcpu-opt) ----

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


def compute_metrics(ref_png, gen_png):
    """Whole-image SSIM + PSNR + mean abs diff of gen vs ref (both RGB PNGs)."""
    ref = np.asarray(Image.open(ref_png).convert("RGB")).astype(np.float64)
    gen = np.asarray(Image.open(gen_png).convert("RGB")).astype(np.float64)
    if ref.shape != gen.shape:
        raise ValueError("shape mismatch: ref %s vs gen %s"
                         % (ref.shape, gen.shape))
    rg, gg = ref.mean(axis=2), gen.mean(axis=2)
    ssim = float(_ssim_map(rg, gg).mean())
    mse = float(((ref - gen) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    mad = float(np.abs(ref - gen).mean())
    return {"ssim_vs_ref": round(ssim, 4), "psnr_vs_ref_db": round(psnr, 2),
            "mean_abs_diff": round(mad, 2)}


# ---- model downloads ----

def download_models():
    from huggingface_hub import hf_hub_download
    paths = {}
    paths["base"] = hf_hub_download(repo_id=BASE_REPO, filename=BASE_FILE,
                                    local_dir=MODELS)
    paths["te"] = hf_hub_download(repo_id=TE_REPO, filename=TE_FILE,
                                  local_dir=MODELS)
    paths["vision"] = hf_hub_download(repo_id=TE_REPO, filename=TE_VISION,
                                      local_dir=MODELS)
    paths["vae"] = hf_hub_download(repo_id=BASE_REPO, filename=VAE_FILE,
                                   local_dir=MODELS)
    for k, p in paths.items():
        log("model %s: %s (%d bytes)" % (k, p, os.path.getsize(p)))
    return paths


# ---- sd-cli wrappers ----

def load_help():
    global SDCLI_HELP
    try:
        with open("sd-cli-help.txt") as f:
            SDCLI_HELP = f.read()
    except Exception as e:
        log("sd-cli help file missing (%s)" % e)


def convert_level(bf16_path, level, out_dir):
    """Convert BF16 -> level GGUF. Returns (gguf_path, type_used) or (None, err)."""
    for typ in LEVEL_TYPES[level]:
        out = os.path.join(out_dir, "flux3-klein-base-%s.gguf" % level)
        cmd = [SDCPP_BIN, "-M", "convert", "--diffusion-model", bf16_path,
               "--type", typ, "--tensor-type-rules", RULES, "-o", out]
        log("convert %s as %s ..." % (level, typ))
        t0 = time.time()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=900)
            log("convert %s (%s) exit=%d in %.0fs"
                % (level, typ, p.returncode, time.time() - t0))
            if p.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000000:
                log("converted %s -> %s (%d bytes)"
                    % (level, out, os.path.getsize(out)))
                return out, typ
            log("convert %s (%s) failed: %s"
                % (level, typ, p.stdout.decode(errors="ignore")[-500:]))
        except subprocess.TimeoutExpired:
            log("convert %s (%s) timed out after 900s" % (level, typ))
        except Exception as e:
            log("convert %s (%s) error: %s" % (level, typ, e))
        if os.path.exists(out):
            os.remove(out)
    return None, "all convert types failed for %s" % level


def generate(model_path, out_png, steps, cfg, threads, timeout_s, easycache=True,
             log_path=None):
    """One sd-cli txt2img run. Returns (status, wall_seconds, error)."""
    cmd = [SDCPP_BIN,
           "--diffusion-model", model_path,
           "--llm", os.path.join(MODELS, TE_FILE),
           "--llm_vision", os.path.join(MODELS, TE_VISION),
           "--vae", os.path.join(MODELS, VAE_FILE),
           "-p", T2I_PROMPT,
           "--cfg-scale", str(cfg), "--steps", str(steps),
           "--sampling-method", "euler",
           "--width", "512", "--height", "512",
           "--threads", str(threads),
           "--output", out_png, "-v"]
    if "--offload-to-cpu" in SDCLI_HELP:
        cmd.append("--offload-to-cpu")
    if easycache and "--cache-mode" in SDCLI_HELP:
        cmd += ["--cache-mode", "easycache"]
    log("generate %s (steps=%s cfg=%s threads=%s easycache=%s timeout=%ss)"
        % (os.path.basename(out_png), steps, cfg, threads, easycache, timeout_s))
    t0 = time.time()
    try:
        lf = open(log_path, "wb") if log_path else subprocess.DEVNULL
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=timeout_s)
        wall = round(time.time() - t0, 1)
        if lf is not subprocess.DEVNULL:
            lf.close()
        if p.returncode != 0:
            return "failed", wall, "sd-cli exit %d (see %s)" % (p.returncode, log_path)
        if os.path.exists(out_png) and os.path.getsize(out_png) > 10000:
            return "success", wall, None
        return "failed", wall, "sd-cli exit 0 but no output image"
    except subprocess.TimeoutExpired:
        return "timeout", round(time.time() - t0, 1), "killed after %ss" % timeout_s
    except Exception as e:
        return "failed", round(time.time() - t0, 1), str(e)[:500]


# ---- stages ----

def stage0(args, results):
    """Base BF16 reference: ladder-steps first (metric ref), then 50 steps."""
    models = download_models()
    # 1. Metric reference: base at the ladder steps/CFG (FIRST so it exists
    #    even if the 50-step generation times out).
    rec = {"name": "base-%dstep" % args.steps, "level": "bf16",
           "quant_type_used": "bf16", "steps": args.steps, "cfg": args.cfg,
           "threads": 4, "easycache": True}
    out_png = os.path.join(args.out, "base-%d.png" % args.steps)
    st, wall, err = generate(models["base"], out_png, args.steps, args.cfg, 4,
                             1800, easycache=True,
                             log_path=os.path.join(args.out, "sdcpp-base-%d.log" % args.steps))
    rec.update({"status": st, "wall_seconds": wall, "error": err,
                "outputs": [{"file": os.path.basename(out_png),
                             "bytes": os.path.getsize(out_png)}]
                if st == "success" else []})
    results["tests"].append(rec)
    save(results, args)
    # 2. True reference: base at 50 steps CFG (BFL spec for the base model).
    rec50 = {"name": "base-%dstep" % args.stage0_steps, "level": "bf16",
             "quant_type_used": "bf16", "steps": args.stage0_steps,
             "cfg": args.cfg, "threads": 4, "easycache": True}
    out50 = os.path.join(args.out, "base-%d.png" % args.stage0_steps)
    st, wall, err = generate(models["base"], out50, args.stage0_steps, args.cfg,
                             4, 3000, easycache=True,
                             log_path=os.path.join(args.out, "sdcpp-base-%d.log" % args.stage0_steps))
    rec50.update({"status": st, "wall_seconds": wall, "error": err,
                  "outputs": [{"file": os.path.basename(out50),
                               "bytes": os.path.getsize(out50)}]
                  if st == "success" else []})
    results["tests"].append(rec50)
    # 3. Step effect: how far is base-at-ladder-steps from base-at-50?
    if rec["status"] == "success" and rec50["status"] == "success":
        try:
            m = compute_metrics(out50, out_png)
            rec50["vs_base50"] = m
            rec["vs_base50"] = {"ssim_vs_ref": m["ssim_vs_ref"],
                                "psnr_vs_ref_db": m["psnr_vs_ref_db"],
                                "mean_abs_diff": m["mean_abs_diff"]}
            log("step effect base-%d vs base-%d: SSIM %.4f PSNR %.2f dB"
                % (args.steps, args.stage0_steps, m["ssim_vs_ref"],
                   m["psnr_vs_ref_db"]))
        except Exception as e:
            log("step-effect metrics failed: %s" % e)
    save(results, args)


def stage_ladder(args, results):
    """Convert BF16 -> each level, generate at fixed steps/CFG, score vs ref."""
    models = download_models()
    ref = args.ref
    if not ref or not os.path.exists(ref):
        raise SystemExit("ERROR: reference PNG missing (%s) - dispatch stage0 first" % ref)
    log("metric reference: %s" % ref)
    for level in args.levels:
        if level not in LEVEL_TYPES:
            results["tests"].append({"name": level, "level": level,
                                     "status": "skipped",
                                     "error": "unknown level", "wall_seconds": 0})
            save(results, args)
            continue
        if elapsed() > JOB_BUDGET_S:
            results["tests"].append({"name": "base-%s" % level, "level": level,
                                     "status": "skipped",
                                     "error": "out of time (%.0fs elapsed)" % elapsed(),
                                     "wall_seconds": 0})
            save(results, args)
            continue
        rec = {"name": "base-%s" % level, "level": level, "steps": args.steps,
               "cfg": args.cfg, "threads": 4, "easycache": True}
        gguf, typ_or_err = convert_level(models["base"], level, args.out)
        if gguf is None:
            rec.update({"status": "skipped", "wall_seconds": 0,
                        "error": "convert failed: " + typ_or_err})
            results["tests"].append(rec)
            save(results, args)
            continue
        rec["quant_type_used"] = typ_or_err
        rec["gguf_bytes"] = os.path.getsize(gguf)
        out_png = os.path.join(args.out, "base-%s.png" % level)
        st, wall, err = generate(gguf, out_png, args.steps, args.cfg, 4, 1800,
                                 easycache=True,
                                 log_path=os.path.join(args.out, "sdcpp-base-%s.log" % level))
        rec.update({"status": st, "wall_seconds": wall, "error": err,
                    "outputs": [{"file": os.path.basename(out_png),
                                 "bytes": os.path.getsize(out_png)}]
                    if st == "success" else []})
        if st == "success":
            try:
                rec.update(compute_metrics(ref, out_png))
                log("level %s: SSIM %.4f PSNR %.2f dB (%.1fs)"
                    % (level, rec["ssim_vs_ref"], rec["psnr_vs_ref_db"], wall))
            except Exception as e:
                rec["error"] = "metrics failed: %s" % e
                log("metrics failed for %s: %s" % (level, e))
        results["tests"].append(rec)
        save(results, args)
        # Free disk: the GGUF is no longer needed once scored.
        try:
            os.remove(gguf)
            log("deleted %s" % gguf)
        except Exception as e:
            log("could not delete %s: %s" % (gguf, e))


def stage_speed(args, results):
    """Optimize the winning level: threads, steps, EasyCache, one at a time."""
    models = download_models()
    ref = args.ref
    if not ref or not os.path.exists(ref):
        raise SystemExit("ERROR: reference PNG missing (%s) - dispatch stage0 first" % ref)
    winner = args.winner
    if winner not in LEVEL_TYPES:
        raise SystemExit("ERROR: unknown winner level %s" % winner)
    log("speed stage on winner: %s" % winner)
    gguf, typ_or_err = convert_level(models["base"], winner, args.out)
    if gguf is None:
        results["speed"].append({"name": "speed-%s" % winner, "level": winner,
                                 "status": "skipped", "wall_seconds": 0,
                                 "error": "convert failed: " + typ_or_err})
        save(results, args)
        return
    probes = [
        ("base", args.steps, args.cfg, 4, True),
        ("t2", args.steps, args.cfg, 2, True),
        ("nocache", args.steps, args.cfg, 4, False),
    ]
    if args.steps > 12:
        probes.append(("s12", 12, args.cfg, 4, True))
    for pname, steps, cfg, threads, easycache in probes:
        if elapsed() > JOB_BUDGET_S:
            results["speed"].append({"name": "speed-%s-%s" % (winner, pname),
                                     "level": winner, "status": "skipped",
                                     "wall_seconds": 0,
                                     "error": "out of time (%.0fs elapsed)" % elapsed()})
            save(results, args)
            continue
        rec = {"name": "speed-%s-%s" % (winner, pname), "level": winner,
               "quant_type_used": typ_or_err, "steps": steps, "cfg": cfg,
               "threads": threads, "easycache": easycache}
        out_png = os.path.join(args.out, "speed-%s-%s.png" % (winner, pname))
        st, wall, err = generate(gguf, out_png, steps, cfg, threads, 1800,
                                 easycache=easycache,
                                 log_path=os.path.join(args.out, "sdcpp-speed-%s-%s.log" % (winner, pname)))
        rec.update({"status": st, "wall_seconds": wall, "error": err,
                    "outputs": [{"file": os.path.basename(out_png),
                                 "bytes": os.path.getsize(out_png)}]
                    if st == "success" else []})
        if st == "success":
            try:
                rec.update(compute_metrics(ref, out_png))
                log("speed %s: %.1fs SSIM %.4f"
                    % (pname, wall, rec["ssim_vs_ref"]))
            except Exception as e:
                rec["error"] = "metrics failed: %s" % e
        results["speed"].append(rec)
        save(results, args)
    try:
        os.remove(gguf)
        log("deleted %s" % gguf)
    except Exception as e:
        log("could not delete %s: %s" % (gguf, e))


def save(results, args):
    results["elapsed_seconds"] = round(elapsed(), 1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["stage0", "ladder", "speed"])
    ap.add_argument("--levels", default="",
                    help="comma-separated ladder levels (ladder stage)")
    ap.add_argument("--steps", type=int, default=16,
                    help="fixed steps for the ladder (and metric ref)")
    ap.add_argument("--cfg", type=float, default=3.5,
                    help="fixed CFG for the whole ladder")
    ap.add_argument("--stage0-steps", type=int, default=50,
                    help="steps for the true base reference")
    ap.add_argument("--winner", default="q4_k_m",
                    help="winning level to optimize (speed stage)")
    ap.add_argument("--run-speed", action="store_true",
                    help="run the speed stage after the ladder stage (one job)")
    ap.add_argument("--ref", default="",
                    help="metric reference PNG (base at ladder steps)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    args.levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(SDCPP_BIN):
        print("ERROR sd-cli binary missing at %s" % SDCPP_BIN)
        sys.exit(1)
    load_help()
    results = {"stage": args.stage, "levels": args.levels,
               "steps": args.steps, "cfg": args.cfg,
               "stage0_steps": args.stage0_steps, "seed": SEED,
               "prompt": T2I_PROMPT,
               "base_checkpoint": BASE_REPO + " " + BASE_FILE,
               "text_encoder": TE_REPO + " " + TE_FILE + " + " + TE_VISION,
               "vae": BASE_REPO + " " + VAE_FILE,
               "tests": [], "speed": []}
    if args.stage == "stage0":
        stage0(args, results)
    elif args.stage == "ladder":
        stage_ladder(args, results)
        if args.run_speed:
            stage_speed(args, results)
    elif args.stage == "speed":
        stage_speed(args, results)
    save(results, args)
    log("ALL DONE in %.0fs" % elapsed())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
