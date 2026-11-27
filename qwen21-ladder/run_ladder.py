#!/usr/bin/env python3
"""Staged Qwen-Image-2.1 quant ladder: base reference -> gradual compression -> speed.

Same treatment as the FLUX 3 Klein ladder (quant-ladder/run_ladder.py):
same swordsman prompt + seed 42 + 512^2 so the two models are comparable.

Differences from the FLUX ladder (documented, deliberate):
- Steps 20 / CFG 6.0 / euler: the proven sd.cpp data point for base
  Qwen-Image-2.1 (Frosty40 / DogukanUrker / AtomicChat), vs FLUX ladder's
  16 steps / CFG 3.5. Diffusers' default (50 steps / CFG 4.0) is
  unaffordable on a CPU runner.
- No EasyCache: kept OFF for the whole ladder (proven-plain path; the
  FLUX ladder used it, Qwen flow-matching support is unproven).
- No --llm_vision / mmproj: the proven sd.cpp invocations pass only
  --llm Qwen3VL-8B-Instruct-Q8_0.gguf for T2I.
- No 50-step second reference: stage 0 is a single base-20 generation
  (the metric reference). A 50-step base gen would not fit the 120-min cap.
- DiT ships as 2 BF16 shards (Qwen/Qwen-Image-2.1, transformer/); they are
  merged into one BF16 safetensors (safetensors.numpy, no torch) and the
  shards deleted before generation (disk). Shard names are resolved via
  HfApi.list_repo_files, not hardcoded, so a rename does not kill the run.
- No fp8 / tq2 rungs: ladder is Q8_0 -> Q6_K -> Q5_K_M -> Q4_K_M -> Q3_K
  -> Q2_K, converted from the merged BF16 with `sd-cli -M convert`.

Stage 0 (base reference): merged BF16 DiT + Q8_0 TE + VAE, 20 steps / CFG 6.
Stage ladder: convert BF16 -> each level (one at a time), generate at fixed
steps/CFG, SSIM/PSNR vs the metric reference, delete the GGUF after scoring.
Stage speed: optimize the winning level (threads, steps) one variable at a
time, metrics vs the same reference.

License note: Qwen-Image-2.1 is under the Qwen Research License -
research/evaluation only, commercial use needs a paid license.
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

# --- constants (same prompt/seed as the FLUX ladder, verbatim from client.py) ---
T2I_PROMPT = (
    "A detailed anime-style illustration of a young swordsman standing on a misty "
    "mountain path at dawn, long black hair tied back, dark blue robes with silver "
    "trim, holding a sheathed katana at his side, pine trees and drifting fog in "
    "the background, soft golden morning light breaking through the mist, "
    "cinematic composition, clean linework, detailed shading."
)
DIT_REPO = "Qwen/Qwen-Image-2.1"
DIT_PREFIX = "transformer/diffusion_pytorch_model-"
TE_REPO = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
TE_FILE = "Qwen3VL-8B-Instruct-Q8_0.gguf"
VAE_REPO = "Comfy-Org/Qwen-Image-2.1"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"
MERGED_BF16 = "qwen-image-2.1-bf16.safetensors"
SDCPP_BIN = "sd.cpp/build/bin/sd-cli"
MODELS = "models"
SEED = 42
JOB_BUDGET_S = 6000  # stop starting new levels after this (leave time to upload)
STAGE0_GEN_TIMEOUT_S = 3600
LADDER_GEN_TIMEOUT_S = 2700
SPEED_GEN_TIMEOUT_S = 1800

LEVEL_TYPES = {
    "q8_0": ["q8_0"],
    "q6_k": ["q6_k", "q6_K"],
    "q5_k_m": ["q5_k_m", "q5_K_M", "q5_k", "q5_K"],
    "q4_k_m": ["q4_k_m", "q4_K_M", "q4_k", "q4_K"],
    "q3_k": ["q3_k", "q3_K"],
    "q2_k": ["q2_k", "q2_K"],
}

T_START = time.time()
SDCLI_HELP = ""


def elapsed():
    return time.time() - T_START


def log(msg):
    print("[qwen21-ladder %.0fs] %s" % (elapsed(), msg), flush=True)


def disk_log(tag):
    try:
        u = shutil.disk_usage(".")
        log("disk %s: free %.1f GB / total %.1f GB"
            % (tag, u.free / 1e9, u.total / 1e9))
    except Exception as e:
        log("disk check failed: %s" % e)


# ---- metrics (numpy-only SSIM, same implementation as the FLUX ladder) ----

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


# ---- model downloads + DiT shard merge ----

def download_models():
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    # Resolve DiT shard names live (robust to renames); expect 2 shards.
    files = api.list_repo_files(DIT_REPO)
    shards = sorted(f for f in files
                    if f.startswith(DIT_PREFIX) and f.endswith(".safetensors"))
    log("DiT shards in %s: %s" % (DIT_REPO, shards))
    if not shards:
        raise SystemExit("ERROR: no DiT shards found in %s" % DIT_REPO)
    paths = {}
    paths["shards"] = [hf_hub_download(repo_id=DIT_REPO, filename=s,
                                       local_dir=MODELS) for s in shards]
    for p in paths["shards"]:
        log("shard: %s (%d bytes)" % (p, os.path.getsize(p)))
    disk_log("after-shard-download")
    paths["merged"] = merge_shards(paths["shards"])
    # Free disk: shards are no longer needed once merged.
    for p in paths["shards"]:
        try:
            os.remove(p)
            log("deleted shard %s" % p)
        except Exception as e:
            log("could not delete %s: %s" % (p, e))
    disk_log("after-merge")
    paths["te"] = hf_hub_download(repo_id=TE_REPO, filename=TE_FILE,
                                  local_dir=MODELS)
    paths["vae"] = hf_hub_download(repo_id=VAE_REPO, filename=VAE_FILE,
                                   local_dir=MODELS)
    for k in ("te", "vae"):
        log("model %s: %s (%d bytes)" % (k, paths[k], os.path.getsize(paths[k])))
    disk_log("after-all-downloads")
    return paths


def merge_shards(shard_paths):
    """Merge DiT safetensors shards into one BF16 safetensors (numpy, no torch).

    NOTE: numpy cannot parse bfloat16 on its own - `ml_dtypes` must be
    imported first so the dtype registers (run 1 failed here with
    "TypeError: data type 'bfloat16' not understood" before ml_dtypes
    was added to the workflow's pip install).
    """
    import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
    from safetensors.numpy import load_file, save_file
    merged = {}
    for p in shard_paths:
        d = load_file(p)
        log("merge: %s has %d tensors" % (os.path.basename(p), len(d)))
        dup = set(merged) & set(d)
        if dup:
            log("WARNING: %d duplicate keys across shards (later wins): %s"
                % (len(dup), sorted(dup)[:5]))
        merged.update(d)
    out = os.path.join(MODELS, MERGED_BF16)
    save_file(merged, out, metadata={"format": "pt"})
    log("merged %d tensors -> %s (%d bytes)"
        % (len(merged), out, os.path.getsize(out)))
    return out


# ---- sd-cli wrappers ----

def load_help():
    global SDCLI_HELP
    try:
        with open("sd-cli-help.txt") as f:
            SDCLI_HELP = f.read()
    except Exception as e:
        log("sd-cli help file missing (%s)" % e)


def convert_level(bf16_path, level, out_dir):
    """Convert merged BF16 -> level GGUF. Returns (gguf_path, type_used) or (None, err)."""
    for typ in LEVEL_TYPES[level]:
        out = os.path.join(out_dir, "qwen21-base-%s.gguf" % level)
        cmd = [SDCPP_BIN, "-M", "convert", "--diffusion-model", bf16_path,
               "--type", typ, "-o", out]
        log("convert %s as %s ..." % (level, typ))
        t0 = time.time()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=1200)
            log("convert %s (%s) exit=%d in %.0fs"
                % (level, typ, p.returncode, time.time() - t0))
            if p.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000000:
                log("converted %s -> %s (%d bytes)"
                    % (level, out, os.path.getsize(out)))
                disk_log("after-convert-%s" % level)
                return out, typ
            log("convert %s (%s) failed: %s"
                % (level, typ, p.stdout.decode(errors="ignore")[-500:]))
        except subprocess.TimeoutExpired:
            log("convert %s (%s) timed out after 1200s" % (level, typ))
        except Exception as e:
            log("convert %s (%s) error: %s" % (level, typ, e))
        if os.path.exists(out):
            os.remove(out)
    return None, "all convert types failed for %s" % level


def generate(model_path, out_png, steps, cfg, threads, timeout_s,
             log_path=None, extra_flags=None):
    """One sd-cli txt2img run. Returns (status, wall_seconds, error)."""
    cmd = [SDCPP_BIN,
           "--diffusion-model", model_path,
           "--llm", os.path.join(MODELS, TE_FILE),
           "--vae", os.path.join(MODELS, VAE_FILE),
           "-p", T2I_PROMPT,
           "--cfg-scale", str(cfg), "--steps", str(steps),
           "--sampling-method", "euler",
           "--width", "512", "--height", "512",
           "--threads", str(threads),
           "--output", out_png, "-v"]
    if "--seed" in SDCLI_HELP:
        cmd += ["--seed", str(SEED)]
    if "--offload-to-cpu" in SDCLI_HELP:
        cmd.append("--offload-to-cpu")
    if extra_flags:
        cmd += list(extra_flags)
    log("generate %s (steps=%s cfg=%s threads=%s timeout=%ss)"
        % (os.path.basename(out_png), steps, cfg, threads, timeout_s))
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
    """Base BF16 reference: merged BF16 at the ladder steps/CFG (single gen)."""
    models = download_models()
    rec = {"name": "base-%dstep" % args.steps, "level": "bf16",
           "quant_type_used": "bf16", "steps": args.steps, "cfg": args.cfg,
           "threads": 4, "easycache": False}
    out_png = os.path.join(args.out, "base-%d.png" % args.steps)
    st, wall, err = generate(models["merged"], out_png, args.steps, args.cfg, 4,
                             STAGE0_GEN_TIMEOUT_S,
                             log_path=os.path.join(args.out, "sdcpp-base-%d.log" % args.steps))
    rec.update({"status": st, "wall_seconds": wall, "error": err,
                "outputs": [{"file": os.path.basename(out_png),
                             "bytes": os.path.getsize(out_png)}]
                if st == "success" else []})
    results["tests"].append(rec)
    save(results, args)


def stage_ladder(args, results):
    """Convert merged BF16 -> each level, generate at fixed steps/CFG, score vs ref."""
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
               "cfg": args.cfg, "threads": 4, "easycache": False}
        gguf, typ_or_err = convert_level(models["merged"], level, args.out)
        if gguf is None:
            rec.update({"status": "skipped", "wall_seconds": 0,
                        "error": "convert failed: " + typ_or_err})
            results["tests"].append(rec)
            save(results, args)
            continue
        rec["quant_type_used"] = typ_or_err
        rec["gguf_bytes"] = os.path.getsize(gguf)
        out_png = os.path.join(args.out, "base-%s.png" % level)
        st, wall, err = generate(gguf, out_png, args.steps, args.cfg, 4,
                                 LADDER_GEN_TIMEOUT_S,
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


# Speed probes: name -> (steps or None=ladder steps, cfg or None=ladder cfg,
# threads, extra sd-cli flags). Order matters: the most important probes
# first, so a cut-short job still has the key data.
SPEED_PROBES = {
    "base": (None, None, 4, []),
    "s16": (16, None, 4, []),
    "s12": (12, None, 4, []),
    "t2": (None, None, 2, []),
    "convdirect": (None, None, 4, ["--conv-direct", "on"]),
}


def stage_speed(args, results):
    """Optimize the winning level: threads, steps, one at a time."""
    models = download_models()
    ref = args.ref
    if not ref or not os.path.exists(ref):
        raise SystemExit("ERROR: reference PNG missing (%s) - dispatch stage0 first" % ref)
    winner = args.winner
    if winner not in LEVEL_TYPES:
        raise SystemExit("ERROR: unknown winner level %s" % winner)
    log("speed stage on winner: %s probes=%s" % (winner, args.probes))
    gguf, typ_or_err = convert_level(models["merged"], winner, args.out)
    if gguf is None:
        results["speed"].append({"name": "speed-%s" % winner, "level": winner,
                                 "status": "skipped", "wall_seconds": 0,
                                 "error": "convert failed: " + typ_or_err})
        save(results, args)
        return
    for pname in args.probes:
        if pname not in SPEED_PROBES:
            results["speed"].append({"name": "speed-%s-%s" % (winner, pname),
                                     "level": winner, "status": "skipped",
                                     "wall_seconds": 0,
                                     "error": "unknown probe %s" % pname})
            save(results, args)
            continue
        psteps, pcfg, threads, extra = SPEED_PROBES[pname]
        steps = args.steps if psteps is None else psteps
        cfg = args.cfg if pcfg is None else pcfg
        if elapsed() > JOB_BUDGET_S:
            results["speed"].append({"name": "speed-%s-%s" % (winner, pname),
                                     "level": winner, "status": "skipped",
                                     "wall_seconds": 0,
                                     "error": "out of time (%.0fs elapsed)" % elapsed()})
            save(results, args)
            continue
        rec = {"name": "speed-%s-%s" % (winner, pname), "level": winner,
               "quant_type_used": typ_or_err, "steps": steps, "cfg": cfg,
               "threads": threads, "easycache": False,
               "extra_flags": extra}
        out_png = os.path.join(args.out, "speed-%s-%s.png" % (winner, pname))
        st, wall, err = generate(gguf, out_png, steps, cfg, threads,
                                 SPEED_GEN_TIMEOUT_S,
                                 log_path=os.path.join(args.out, "sdcpp-speed-%s-%s.log" % (winner, pname)),
                                 extra_flags=extra)
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
    ap.add_argument("--steps", type=int, default=20,
                    help="fixed steps for the ladder (and metric ref)")
    ap.add_argument("--cfg", type=float, default=6.0,
                    help="fixed CFG for the whole ladder")
    ap.add_argument("--winner", default="q4_k_m",
                    help="winning level to optimize (speed stage)")
    ap.add_argument("--probes", default="base,s16,s12,t2,convdirect",
                    help="comma-separated speed probes (speed stage)")
    ap.add_argument("--run-speed", action="store_true",
                    help="run the speed stage after the ladder stage (one job)")
    ap.add_argument("--ref", default="",
                    help="metric reference PNG (base at ladder steps)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    args.levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    args.probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(SDCPP_BIN):
        print("ERROR sd-cli binary missing at %s" % SDCPP_BIN)
        sys.exit(1)
    load_help()
    results = {"stage": args.stage, "levels": args.levels,
               "steps": args.steps, "cfg": args.cfg,
               "seed": SEED,
               "winner": args.winner, "probes": args.probes,
               "prompt": T2I_PROMPT,
               "base_checkpoint": DIT_REPO + " transformer/ (merged BF16)",
               "text_encoder": TE_REPO + " " + TE_FILE,
               "vae": VAE_REPO + " " + VAE_FILE,
               "license": "Qwen-Image-2.1 is under the Qwen Research License "
                          "(research/evaluation only)",
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


def run():
    """main() wrapped so an early failure (download/merge) still uploads
    a results.json with the diagnosis instead of an empty artifact."""
    try:
        main()
    except SystemExit as e:
        if e.code not in (0, None):
            _write_early_failure("SystemExit code %s" % e.code)
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        _write_early_failure(str(e)[:1000])
        sys.exit(1)


def _write_early_failure(msg):
    try:
        out = "results"
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "results.json"), "w") as f:
            json.dump({"stage": "failed-early", "elapsed_seconds": round(elapsed(), 1),
                       "error": msg, "tests": [], "speed": []}, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    run()
