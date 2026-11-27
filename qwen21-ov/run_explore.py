#!/usr/bin/env python3
"""Qwen-Image-2.1 extra-speedup exploration (OpenVINO + other lanes).

Separate from the R1-R6 sd.cpp speed test (results-qwen21-speed/): this
driver probes paths the speed test scoped but did not run, on the SAME
test (swordsman prompt, seed 42, 512^2, SSIM/PSNR vs the Q6_K reference
base-20.png) so numbers compare.

Paths (--path):
  viggle   - Viggle Turbo 6-step fused GGUF (Q6_K) + Lightning 4-step
             fused GGUF (Q6_K) in sd.cpp, cfg 1.0, per-LoRA sigmas.
             Different distillation from the official Qwen-Image-2.1-Turbo
             checkpoint (already tested: reframed), so needs its own check.
  cache    - sd.cpp --cache-mode sweep at 20 steps / Q6_K: easycache
             (within-run baseline) + taylorseer + cache-dit + dbcache +
             spectrum + ucache (only modes sd-cli --help actually lists).
  res384   - Q4_K_M DiT + EasyCache, and 384^2 generate + Lanczos upscale
             to 512^2 (plain + EasyCache). Tests the ~44% DiT-area cut.
  openvino - OpenVINO INT4 CPU speed probe, NO export: run the pre-converted
             community build (blaj, abliterated TE) via optimum-intel
             OVDiffusionPipeline on CPU, 10 steps. SPEED-ONLY probe: the
             TE differs from stock, so SSIM vs the stock reference is NOT
             a fair quality comparison (reported, flagged). Released stack
             first (optimum-intel 2.2.0 / OV 2026.4); on the known
             unsupported-pipeline failure, install the nightly stack and
             retry once. Full-pipeline export is OUT of scope (OOM-killed
             on 30 GB; see RUNS.md).

Reuses qwen21-ladder/run_ladder.py (metrics, sd-cli wrappers, model
downloads) via import. License note: Qwen-Image-2.1 is under the Qwen
Research License - research/evaluation only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "qwen21-ladder"))
import run_ladder as rl  # noqa: E402  (metrics, generate, convert, download)

T_START = time.time()
JOB_BUDGET_S = 6000  # stop starting new probes after this

VIGGLE_REPO = "Abiray/Qwen-Image-2.1-viggle-turbo-v0.3-6step-GGUF"
VIGGLE_FILE = "qwen_image_2.1_turbo_v0.3_Q6_K.gguf"
VIGGLE_SIGMAS = "1.000000,0.962557,0.923057,0.837170,0.631512,0.363570,0.0"
LIGHTNING_REPO = "darioooooo0o/Lightning-4step-Qwen-Image-2.1-GGUF"
LIGHTNING_FILE = "qwen-image-2.1-lightning-4step-Q6_K.gguf"
LIGHTNING_SIGMAS = "1.0,0.857,0.667,0.400,0.0"
OV_REPO = "blaj/Qwen-Image-2.1-Uncensored-OpenVINO-INT4"

CACHE_MODES_WANTED = ["easycache", "taylorseer", "cache-dit", "cache_dit",
                      "dbcache", "spectrum", "ucache"]


def elapsed():
    return time.time() - T_START


def log(msg):
    print("[qwen21-ov %.0fs] %s" % (elapsed(), msg), flush=True)


def disk_log(tag):
    try:
        u = shutil.disk_usage(".")
        log("disk %s: free %.1f GB / total %.1f GB"
            % (tag, u.free / 1e9, u.total / 1e9))
    except Exception as e:
        log("disk check failed: %s" % e)


def disk_free_gb():
    return shutil.disk_usage(".").free / 1e9


def download_te_vae():
    """Text encoder + VAE only (no DiT merge) - for pre-fused GGUF paths."""
    from huggingface_hub import hf_hub_download
    paths = {}
    paths["te"] = hf_hub_download(repo_id=rl.TE_REPO, filename=rl.TE_FILE,
                                  local_dir=rl.MODELS)
    paths["vae"] = hf_hub_download(repo_id=rl.VAE_REPO, filename=rl.VAE_FILE,
                                   local_dir=rl.MODELS)
    for k in ("te", "vae"):
        log("model %s: %s (%d bytes)"
            % (k, paths[k], os.path.getsize(paths[k])))
    disk_log("after-te-vae")
    return paths


def download_gguf(repo_id, filename, tag):
    """Download one pre-fused GGUF into models/. Returns path or None."""
    from huggingface_hub import hf_hub_download
    if disk_free_gb() < 10:
        log("SKIP %s: only %.1f GB free, need ~10 GB headroom" % (tag, disk_free_gb()))
        return None
    try:
        p = hf_hub_download(repo_id=repo_id, filename=filename,
                            local_dir=rl.MODELS)
        log("gguf %s: %s (%d bytes)" % (tag, p, os.path.getsize(p)))
        disk_log("after-" + tag)
        return p
    except Exception as e:
        log("download %s FAILED: %s" % (tag, e))
        return None


def gen_record(results, name, gguf, ref, out_png, steps, cfg, threads,
               timeout_s, extra_flags=None, sigmas=None, extra=None):
    """One sd-cli generate + metrics. Appends a record, returns it."""
    rec = {"name": name, "steps": steps, "cfg": cfg, "threads": threads,
           "extra_flags": extra_flags or [], "sigmas": sigmas or ""}
    if extra:
        rec.update(extra)
    st, wall, err = rl.generate(gguf, out_png, steps, cfg, threads, timeout_s,
                                log_path=os.path.join(
                                    os.path.dirname(out_png),
                                    "sdcpp-%s.log" % name),
                                extra_flags=extra_flags, sigmas=sigmas)
    rec.update({"status": st, "wall_seconds": wall, "error": err,
                "outputs": [{"file": os.path.basename(out_png),
                             "bytes": os.path.getsize(out_png)}]
                if st == "success" else []})
    if st == "success" and ref:
        try:
            rec.update(rl.compute_metrics(ref, out_png))
            log("%s: %.1fs SSIM %.4f PSNR %.2f dB"
                % (name, wall, rec["ssim_vs_ref"], rec["psnr_vs_ref_db"]))
        except Exception as e:
            rec["error"] = "metrics failed: %s" % e
    results["tests"].append(rec)
    save(results)
    return rec


def save(results):
    results["elapsed_seconds"] = round(elapsed(), 1)
    with open(os.path.join(results["_out"], "results.json"), "w") as f:
        json.dump({k: v for k, v in results.items() if not k.startswith("_")},
                  f, indent=2)


# ---- path: viggle / lightning ----

def path_viggle(args, results):
    download_te_vae()
    # Viggle Turbo 6-step fused Q6_K, cfg 1.0, Viggle sigmas.
    gguf = download_gguf(VIGGLE_REPO, VIGGLE_FILE, "viggle-q6k")
    if gguf:
        gen_record(results, "viggle-base", gguf, args.ref,
                   os.path.join(args.out, "viggle-base.png"),
                   6, 1.0, 4, 2700, sigmas=VIGGLE_SIGMAS,
                   extra={"dit": VIGGLE_REPO + " " + VIGGLE_FILE})
        if elapsed() < JOB_BUDGET_S:
            gen_record(results, "viggle-easycache", gguf, args.ref,
                       os.path.join(args.out, "viggle-easycache.png"),
                       6, 1.0, 4, 2700,
                       extra_flags=["--cache-mode", "easycache"],
                       sigmas=VIGGLE_SIGMAS,
                       extra={"dit": VIGGLE_REPO + " " + VIGGLE_FILE})
        try:
            os.remove(gguf)
            log("deleted %s" % gguf)
        except Exception as e:
            log("could not delete %s: %s" % (gguf, e))
    else:
        results["tests"].append({"name": "viggle-base", "status": "skipped",
                                 "wall_seconds": 0,
                                 "error": "viggle GGUF download failed"})
        save(results)
    # Lightning 4-step fused Q6_K, cfg 1.0, Lightning sigmas.
    if elapsed() < JOB_BUDGET_S:
        gguf2 = download_gguf(LIGHTNING_REPO, LIGHTNING_FILE, "lightning-q6k")
        if gguf2:
            gen_record(results, "lightning-base", gguf2, args.ref,
                       os.path.join(args.out, "lightning-base.png"),
                       4, 1.0, 4, 2700, sigmas=LIGHTNING_SIGMAS,
                       extra={"dit": LIGHTNING_REPO + " " + LIGHTNING_FILE})
            try:
                os.remove(gguf2)
                log("deleted %s" % gguf2)
            except Exception as e:
                log("could not delete %s: %s" % (gguf2, e))
        else:
            results["tests"].append({"name": "lightning-base",
                                     "status": "skipped", "wall_seconds": 0,
                                     "error": "lightning GGUF download failed"})
            save(results)


# ---- path: cache-mode sweep ----

def available_cache_modes():
    """Parse sd-cli --help for --cache-mode values (names vary - vs _)."""
    try:
        with open("sd-cli-help.txt") as f:
            help_text = f.read()
    except Exception:
        return []
    modes = []
    for line in help_text.splitlines():
        if "cache-mode" in line:
            # values are usually listed like: easycache|ucache|dbcache|...
            import re
            for tok in re.split(r"[\s|,()\[\]]+", line):
                t = tok.strip().lower()
                if t in ("easycache", "ucache", "dbcache", "taylorseer",
                         "cache-dit", "cache_dit", "spectrum", "cache-ec",
                         "cache_dit".replace("_", "-")):
                    modes.append(t)
    # dedupe preserving order
    seen, out = set(), []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def path_cache(args, results):
    rl.load_help()
    avail = available_cache_modes()
    log("sd-cli cache modes listed in help: %s" % avail)
    probes = {}
    if not avail:
        # Help text did not parse: try the known names blindly; a bad
        # name fails its single probe and is recorded, nothing crashes.
        log("WARNING: could not parse cache modes from help; trying known names")
        for want in CACHE_MODES_WANTED:
            probes[want] = (None, None, 4, ["--cache-mode", want])
    else:
        for want in CACHE_MODES_WANTED:
            # match ignoring - vs _
            match = next((a for a in avail if a.replace("-", "_") == want.replace("-", "_")),
                         None)
            if match:
                probes[want] = (None, None, 4, ["--cache-mode", match])
    # easycache first (within-run baseline), then the rest in listed order.
    ordered = [p for p in ["easycache"] + list(probes) if p in probes]
    rl.SPEED_PROBES.update({p: probes[p] for p in ordered})
    log("cache sweep probes: %s (missing modes skipped: %s)"
        % (ordered, [w for w in CACHE_MODES_WANTED if w not in probes]))
    ns = argparse.Namespace(dit_repo=rl.DIT_REPO, winner="q6_k",
                            probes=ordered, steps=args.steps, cfg=args.cfg,
                            ref=args.ref, out=args.out, sigmas="",
                            levels=[])
    rl.stage_speed(ns, results)
    save(results)


# ---- path: 384^2 + upscale ----

def gen_384(gguf, out_png, steps, cfg, threads, timeout_s, log_path,
            extra_flags=None):
    """sd-cli generate at 384x384 (run_ladder.generate is fixed 512)."""
    import numpy as np  # noqa: F401  (kept for parity; unused here)
    cmd = [rl.SDCPP_BIN,
           "--diffusion-model", gguf,
           "--llm", os.path.join(rl.MODELS, rl.TE_FILE),
           "--vae", os.path.join(rl.MODELS, rl.VAE_FILE),
           "-p", rl.T2I_PROMPT,
           "--cfg-scale", str(cfg), "--steps", str(steps),
           "--sampling-method", "euler",
           "--width", "384", "--height", "384",
           "--threads", str(threads),
           "--output", out_png, "-v"]
    if "--seed" in rl.SDCLI_HELP:
        cmd += ["--seed", str(rl.SEED)]
    if "--offload-to-cpu" in rl.SDCLI_HELP:
        cmd.append("--offload-to-cpu")
    if extra_flags:
        cmd += list(extra_flags)
    log("generate384 %s (steps=%s cfg=%s threads=%s)"
        % (os.path.basename(out_png), steps, cfg, threads))
    t0 = time.time()
    try:
        with open(log_path, "wb") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=timeout_s)
        wall = round(time.time() - t0, 1)
        if p.returncode != 0:
            return "failed", wall, "sd-cli exit %d (see %s)" % (p.returncode, log_path)
        if os.path.exists(out_png) and os.path.getsize(out_png) > 10000:
            return "success", wall, None
        return "failed", wall, "sd-cli exit 0 but no output image"
    except subprocess.TimeoutExpired:
        return "timeout", round(time.time() - t0, 1), "killed after %ss" % timeout_s


def upscale_lanczos(src_png, dst_png):
    from PIL import Image
    im = Image.open(src_png).convert("RGB")
    im.resize((512, 512), Image.LANCZOS).save(dst_png)
    return dst_png


def path_res384(args, results):
    rl.load_help()
    models = rl.download_models()  # merged F16 (needed for converts)
    # Q4_K_M + EasyCache at 20 steps (DiT-quant arm).
    gguf_q4, typ = rl.convert_level(models["merged"], "q4_k_m", args.out)
    if gguf_q4:
        gen_record(results, "q4-easycache", gguf_q4, args.ref,
                   os.path.join(args.out, "q4-easycache.png"),
                   args.steps, args.cfg, 4, 2700,
                   extra_flags=["--cache-mode", "easycache"],
                   extra={"level": "q4_k_m", "quant_type_used": typ})
        try:
            os.remove(gguf_q4)
        except Exception:
            pass
    else:
        results["tests"].append({"name": "q4-easycache", "status": "skipped",
                                 "wall_seconds": 0,
                                 "error": "convert failed: " + typ})
        save(results)
    # 384^2 arms need a Q6_K GGUF.
    gguf_q6, typ6 = rl.convert_level(models["merged"], "q6_k", args.out)
    if not gguf_q6:
        results["tests"].append({"name": "res384-lanczos", "status": "skipped",
                                 "wall_seconds": 0,
                                 "error": "convert failed: " + typ6})
        save(results)
        return
    for pname, extra_flags in [("res384-lanczos", []),
                               ("res384-lanczos-easycache",
                                ["--cache-mode", "easycache"])]:
        if elapsed() > JOB_BUDGET_S:
            results["tests"].append({"name": pname, "status": "skipped",
                                     "wall_seconds": 0,
                                     "error": "out of time (%.0fs elapsed)"
                                              % elapsed()})
            save(results)
            continue
        raw = os.path.join(args.out, pname + "-384.png")
        st, wall, err = gen_384(gguf_q6, raw, args.steps, args.cfg, 4, 2700,
                                os.path.join(args.out, "sdcpp-%s.log" % pname),
                                extra_flags=extra_flags)
        rec = {"name": pname, "level": "q6_k", "steps": args.steps,
               "cfg": args.cfg, "threads": 4,
               "extra_flags": extra_flags, "status": st,
               "wall_seconds": wall, "error": err,
               "gen_resolution": "384x384->512 lanczos"}
        if st == "success":
            up = os.path.join(args.out, pname + ".png")
            upscale_lanczos(raw, up)
            rec["outputs"] = [{"file": os.path.basename(up),
                               "bytes": os.path.getsize(up)}]
            try:
                rec.update(rl.compute_metrics(args.ref, up))
                log("%s: %.1fs SSIM %.4f PSNR %.2f dB"
                    % (pname, wall, rec["ssim_vs_ref"], rec["psnr_vs_ref_db"]))
            except Exception as e:
                rec["error"] = "metrics failed: %s" % e
        results["tests"].append(rec)
        save(results)
    try:
        os.remove(gguf_q6)
        log("deleted %s" % gguf_q6)
    except Exception as e:
        log("could not delete %s: %s" % (gguf_q6, e))


# ---- path: openvino (speed-only probe, no export) ----

OV_ATTEMPT = os.path.join(HERE, "ov_attempt.py")


def run_ov_attempt(stack, timeout_s=3600):
    """Run ov_attempt.py as a subprocess; returns (returncode, stdout tail)."""
    env = dict(os.environ, OV_STACK=stack)
    log("OV attempt on %s stack ..." % stack)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, OV_ATTEMPT,
                            "--ref", ARGS.ref, "--out", ARGS.out],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout_s, env=env)
        out = p.stdout.decode(errors="ignore")
        log("OV attempt (%s) exit=%d in %.0fs" % (stack, p.returncode, time.time() - t0))
        return p.returncode, out[-3000:]
    except subprocess.TimeoutExpired:
        return -1, "OV attempt timed out after %ss" % timeout_s


def pip_install_nightly():
    """Install the nightly OV stack in-place; returns True if pip exited 0."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
           "--pre", "--extra-index-url",
           "https://storage.openvinotoolkit.org/simple/wheels/nightly/",
           "openvino", "openvino-genai", "openvino-tokenizers",
           "git+https://github.com/huggingface/optimum-intel.git",
           "git+https://github.com/huggingface/diffusers.git"]
    log("pip-installing nightly OV stack (this can take several minutes) ...")
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=1800)
        log("nightly pip install exit=%d" % p.returncode)
        if p.returncode != 0:
            log("nightly pip output tail: %s"
                % p.stdout.decode(errors="ignore")[-1500:])
        return p.returncode == 0
    except Exception as e:
        log("nightly pip install error: %s" % e)
        return False


def path_openvino(args, results):
    if disk_free_gb() < 16:
        results["tests"].append({"name": "ov-int4-cpu", "status": "skipped",
                                 "wall_seconds": 0,
                                 "error": "only %.1f GB free, need ~16 GB for "
                                          "the 13 GB IR download" % disk_free_gb()})
        save(results)
        return
    rc, tail = run_ov_attempt("released")
    log("OV released attempt tail:\n%s" % tail)
    if rc == 0:
        _merge_ov_sidecar(args, results, "released")
    if rc != 0:
        log("released stack failed (exit %d) - trying nightly stack once" % rc)
        if pip_install_nightly():
            rc2, tail2 = run_ov_attempt("nightly")
            log("OV nightly attempt tail:\n%s" % tail2)
            if rc2 == 0:
                _merge_ov_sidecar(args, results, "nightly")
            else:
                results["tests"].append(
                    {"name": "ov-int4-cpu", "status": "failed",
                     "wall_seconds": 0,
                     "error": "both stacks failed; released tail: %s | "
                              "nightly tail: %s" % (tail[-800:], tail2[-800:])})
                save(results)
        else:
            results["tests"].append(
                {"name": "ov-int4-cpu", "status": "failed", "wall_seconds": 0,
                 "error": "released stack failed; nightly pip install failed; "
                          "released tail: %s" % tail[-800:]})
            save(results)
    # ov_attempt.py writes its own record into results.json on success.
    save(results)


def _merge_ov_sidecar(args, results, stack):
    """Merge ov_attempt.py's sidecar JSON into results['tests']."""
    sidecar = os.path.join(args.out, "ov-attempt-%s.json" % stack)
    try:
        with open(sidecar) as f:
            rec = json.load(f)
        results["tests"].append(rec)
        log("OV %s probe recorded: load %.1fs gen %.1fs (%.1fs/step)"
            % (stack, rec.get("load_seconds", 0), rec.get("wall_seconds", 0),
               rec.get("seconds_per_step", 0)))
    except Exception as e:
        results["tests"].append({"name": "ov-int4-cpu", "status": "failed",
                                 "wall_seconds": 0,
                                 "error": "attempt exit 0 but sidecar unreadable: %s" % e})
    save(results)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    choices=["viggle", "cache", "res384", "openvino"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--ref", default="", help="metric reference PNG")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    ARGS = args
    os.makedirs(args.out, exist_ok=True)
    results = {"path": args.path, "steps": args.steps, "cfg": args.cfg,
               "seed": rl.SEED, "prompt": rl.T2I_PROMPT,
               "reference": args.ref,
               "reference_note": "Q6_K 20-step ref (base-20.png); the 14.23 GB "
                                 "BF16/F16 cannot load on the 15 GiB runner",
               "license": "Qwen-Image-2.1 is under the Qwen Research License "
                          "(research/evaluation only)",
               "tests": [], "speed": [], "_out": args.out}
    rl.load_help()
    if args.path == "viggle":
        path_viggle(args, results)
    elif args.path == "cache":
        path_cache(args, results)
    elif args.path == "res384":
        path_res384(args, results)
    elif args.path == "openvino":
        path_openvino(args, results)
    save(results)
    log("ALL DONE in %.0fs" % elapsed())
    print(json.dumps({k: v for k, v in results.items() if not k.startswith("_")},
                     indent=2))


def run():
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
            json.dump({"path": "failed-early",
                       "elapsed_seconds": round(elapsed(), 1),
                       "error": msg, "tests": [], "speed": []}, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    run()
