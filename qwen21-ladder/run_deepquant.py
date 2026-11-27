#!/usr/bin/env python3
"""Qwen-Image-2.1 deep-quant sweep: own Q4/Q3/Q2 + third-party GGUF/FP8 builds.

Each probe runs at baseline (20 steps, CFG 6.0, 4 threads) and stacked
(EasyCache + 12 steps, the R6 best combo), scored SSIM/PSNR vs the Q6_K
reference base-20.png (same swordsman prompt, seed 42, 512^2 as the ladder
and the R1-R6 speed test).

Design notes (read before changing):
- Base BF16/F16 (14.23 GB) is PROVEN OOM on the 15 GiB runner and Q8_0
  sampling is PROVEN to time out at 3600 s (q8ref run 4) - neither is
  re-run here. Top of the sweep is Q6_K (the reference itself).
- Own quants convert once from the merged F16 (clean provenance, no
  double quantization) via run_ladder.convert_level.
- Third-party GGUFs download directly (hf_hub_download) and are used
  as-is - no re-quantization. FP8 safetensors loads directly as
  --diffusion-model (sd.cpp loads safetensors; T86 verified FP8 files
  with stable-diffusion.cpp - if the load fails the probe records the
  error and the run moves on).
- Turbo probes use the sd.cpp sigma preset "qwen-image-2.1-turbo"
  (8 steps, CFG 1.0, euler) - the same preset R5 used.
- Probes run in JSON order (most important first); the 6000 s job
  budget check skips the rest and records them as skipped.
- Results: results.json + PNGs + sd-cli logs in --out; upload as the
  qwen21-deepquant-<group> artifact. Local results dir:
  results-qwen21-deepquant/<group>/.

Third-party research (2026-11-27, HF API + READMEs, sizes by HEAD):
- unsloth/Qwen-Image-2.1-GGUF (1.4M downloads, Dynamic 2.0): Q2_K 2.47 GB,
  Q3_K_XL 3.61 GB, Q4_K_M 4.20 GB, Q6_K 6.27 GB, Q8_0 7.64 GB.
- leejet/Qwen-Image-2.1-GGUF (sd.cpp author's own, 187k): Q2_K 2.56 GB,
  Q3_K, Q4_0, Q4_K, Q5_0, Q6_K, Q8_0.
- unsloth/Qwen-Image-2.1-Turbo-GGUF (2.18M): same ladder for Turbo.
- AtomicChat/Qwen-Image-2.1-Turbo-GGUF (612k, AD importance-matrix):
  AD-Q4_K SSIM 0.850 / AD-Q3_K 0.801 / AD-Q2_K 0.671 vs BF16 (8 steps,
  vendor-measured); AD-Q4_K 4.20 GB.
- unsloth/Qwen-Image-2.1-FP8 (673k): single-file FP8 safetensors 7.12 GB.
- Dropped: IQ types (not in sd-cli convert's supported type list);
  INT4 ConvRot / Nunchaku / MNN / ONNX (not sd.cpp-loadable); blaj
  OpenVINO INT4 (OV coordinator's lane, and not sd.cpp-loadable);
  ternary (no Qwen-Image-2.1 ternary exists - Bonsai is FLUX.2 Klein);
  Abiray base Q3_K_M (secondary vendor - OPEN if budget remains).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ladder as L  # helpers: prompt, models, convert, generate, metrics

TURBO_SIGMAS = "qwen-image-2.1-turbo"


def log(msg):
    print("[deepquant %.0fs] %s" % (L.elapsed(), msg), flush=True)


def download_te_vae():
    """TE + VAE only (third-party runs need no DiT shards/merge)."""
    from huggingface_hub import hf_hub_download
    paths = {}
    paths["te"] = hf_hub_download(repo_id=L.TE_REPO, filename=L.TE_FILE,
                                  local_dir=L.MODELS)
    paths["vae"] = hf_hub_download(repo_id=L.VAE_REPO, filename=L.VAE_FILE,
                                   local_dir=L.MODELS)
    for k in ("te", "vae"):
        log("model %s: %s" % (k, paths[k]))
    L.disk_log("after-te-vae-download")
    return paths


def download_hf_file(repo, filename):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=filename, local_dir=L.MODELS)
    log("downloaded %s/%s -> %s (%d bytes)"
        % (repo, filename, p, os.path.getsize(p)))
    L.disk_log("after-download-%s" % os.path.basename(filename))
    return p


def save(results, out_dir):
    results["elapsed_seconds"] = round(L.elapsed(), 1)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def run_group(args, probes, results):
    ref = args.ref
    if not ref or not os.path.exists(ref):
        raise SystemExit("ERROR: reference PNG missing (%s) - fetch q8ref first"
                         % ref)
    log("metric reference: %s | group %s | %d probes"
        % (ref, args.group, len(probes)))
    need_own = any(p.get("source") == "own" for p in probes)
    models = L.download_models() if need_own else download_te_vae()
    converted = {}  # level -> (gguf, type) for own quants, deleted at the end

    def get_own_gguf(level):
        if level in converted:
            return converted[level]
        gguf, typ_or_err = L.convert_level(models["merged"], level, args.out)
        if gguf is None:
            return None, typ_or_err
        converted[level] = (gguf, typ_or_err)
        return gguf, typ_or_err

    for p in probes:
        rec = {"name": p["name"], "source": p.get("source", "own"),
               "repo": p.get("repo", ""), "file": p.get("file", ""),
               "level": p.get("level", ""), "steps": p["steps"],
               "cfg": p["cfg"], "threads": p.get("threads", 4),
               "easycache": "--cache-mode" in (p.get("extra") or []),
               "extra_flags": p.get("extra") or [],
               "sigmas": p.get("sigmas", ""),
               "dit_repo": p.get("dit_repo", L.DIT_REPO)}
        if L.elapsed() > L.JOB_BUDGET_S:
            rec.update({"status": "skipped", "wall_seconds": 0,
                        "error": "out of time (%.0fs elapsed)" % L.elapsed()})
            results["tests"].append(rec)
            save(results, args.out)
            continue
        # Resolve the diffusion model file for this probe.
        model_path, typ = None, ""
        src = rec["source"]
        try:
            if src == "own":
                model_path, typ = get_own_gguf(rec["level"])
                if model_path is None:
                    raise RuntimeError("convert failed: " + typ)
            elif src in ("hf-gguf", "hf-safetensors"):
                model_path = download_hf_file(rec["repo"], rec["file"])
                typ = rec["file"]
            else:
                raise RuntimeError("unknown source %s" % src)
        except Exception as e:
            rec.update({"status": "skipped", "wall_seconds": 0,
                        "error": "model resolve failed: %s" % str(e)[:300]})
            results["tests"].append(rec)
            save(results, args.out)
            continue
        rec["quant_type_used"] = typ
        try:
            rec["gguf_bytes"] = os.path.getsize(model_path)
        except Exception:
            pass
        out_png = os.path.join(args.out, "deepq-%s.png" % rec["name"])
        timeout = p.get("timeout", 2700)
        st, wall, err = L.generate(
            model_path, out_png, rec["steps"], rec["cfg"], rec["threads"],
            timeout,
            log_path=os.path.join(args.out, "sdcpp-deepq-%s.log" % rec["name"]),
            extra_flags=rec["extra_flags"] or None,
            sigmas=rec["sigmas"] or None)
        rec.update({"status": st, "wall_seconds": wall, "error": err,
                    "outputs": [{"file": os.path.basename(out_png),
                                 "bytes": os.path.getsize(out_png)}]
                    if st == "success" else []})
        if st == "success":
            try:
                rec.update(L.compute_metrics(ref, out_png))
                log("probe %s: %.1fs SSIM %.4f PSNR %.2f dB"
                    % (rec["name"], wall, rec["ssim_vs_ref"],
                       rec["psnr_vs_ref_db"]))
            except Exception as e:
                rec["error"] = "metrics failed: %s" % e
                log("metrics failed for %s: %s" % (rec["name"], e))
        results["tests"].append(rec)
        save(results, args.out)
    # Free disk: converted GGUFs are no longer needed once scored.
    for _level, (g, _t) in converted.items():
        try:
            os.remove(g)
            log("deleted %s" % g)
        except Exception as e:
            log("could not delete %s: %s" % (g, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True,
                    help="probe group from the probes file: D1..D6")
    ap.add_argument("--probes-file", default="qwen21-ladder/deepquant-probes.json")
    ap.add_argument("--ref", default="",
                    help="metric reference PNG (Q6_K base-20.png from q8ref)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(L.SDCPP_BIN):
        print("ERROR sd-cli binary missing at %s" % L.SDCPP_BIN)
        sys.exit(1)
    L.load_help()
    with open(args.probes_file) as f:
        groups = json.load(f)
    if args.group not in groups:
        print("ERROR unknown group %s (have %s)"
              % (args.group, sorted(groups)))
        sys.exit(1)
    probes = groups[args.group]
    results = {"stage": "deepquant", "group": args.group,
               "seed": L.SEED, "prompt": L.T2I_PROMPT,
               "reference": args.ref,
               "base_checkpoint": L.DIT_REPO + " transformer/ (merged, BF16->F16 "
                                  "cast; reference is a Q6_K GGUF - the 14.23 GB "
                                  "BF16/F16 cannot load on the 15 GiB runner)",
               "text_encoder": L.TE_REPO + " " + L.TE_FILE,
               "vae": L.VAE_REPO + " " + L.VAE_FILE,
               "license": "Qwen-Image-2.1 is under the Qwen Research License "
                          "(research/evaluation only)",
               "tests": []}
    try:
        run_group(args, probes, results)
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["error"] = str(e)[:1000]
    save(results, args.out)
    log("ALL DONE in %.0fs" % L.elapsed())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
