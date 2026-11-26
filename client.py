#!/usr/bin/env python3
"""Speed-matrix client for the ComfyUI HTTP API + stable-diffusion.cpp.

Env vars:
  BASE_URL - ComfyUI server URL, e.g. http://127.0.0.1:8188
  API_KEY  - placeholder only; ComfyUI has no auth. Sent as a dummy
             Authorization header so the call shape matches an API
             client. Default "none".

Config matrix (same prompt, 512x512, 16 steps unless noted), in order:
  1. comfyui-q4-512-t4   ComfyUI torch CPU + GGUF Q4_K_M (baseline)
  2. sdcpp-q4-512-t4     stable-diffusion.cpp GGML kernels + same Q4_K_M
  3. comfyui-q8-512-t4   ComfyUI + GGUF Q8_0 (quant comparison)
  4. comfyui-q8-512-t2   ComfyUI + Q8_0, 2 threads (thread tuning)
  5. edit-sfx-panel      ComfyUI image edit of assets/sfx-panel.png
  6. t2i-1024            ComfyUI 1024x1024 (only if time allows)

Each record in results.json carries: name, runtime, quant, resolution,
steps, threads, wall_seconds, status, outputs, error.
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8188").rstrip("/")
API_KEY = os.environ.get("API_KEY", "none")
HEADERS = {"Authorization": "Bearer " + API_KEY}

DIT_Q4 = "Sakura-FLUX.3-Image-Klein-Q4_K_M-3.05GiB.gguf"
DIT_Q8 = "Sakura-FLUX.3-Image-Klein-Q8_0-5.58GiB.gguf"
CLIP = "flux3-klein-te-heretic-dhq-Q4_K_M.gguf"
OFFICIAL_TE_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
OFFICIAL_TE_FILE = "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
VAE = "flux2-vae.safetensors"
MODELS = "ComfyUI/models"
SDCPP_BIN = "sd.cpp/bin/sd-cli"
COMFYUI_DIR = "ComfyUI"
COMFYUI_LOG = "comfyui.log"

# Full scene captions (no prompt upsampler is installed, so short prompts
# would give generic output).
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

T_START = time.time()
CURRENT_THREADS = {"n": 4}  # workflow starts the server with default env (~4 cores)


def elapsed():
    return time.time() - T_START


def log(msg):
    print("[client %.0fs] %s" % (elapsed(), msg), flush=True)


def build_t2i(dit, width, height, seed=42, steps=16):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": dit}},
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": CLIP, "type": "flux3"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": T2I_PROMPT, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "flux3_t2i"}},
    }


def build_edit(dit, image_filename, seed=42, steps=16):
    # ReferenceMultiLatent with "keep resolution": the output latent follows
    # the reference image size, so no EmptyFlux2LatentImage is needed.
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": dit}},
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": CLIP, "type": "flux3"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": EDIT_PROMPT, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "17": {"class_type": "ReferenceMultiLatent", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0],
            "resolution": "keep resolution", "image0": ["11", 0]}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["17", 0], "negative": ["17", 1],
            "latent_image": ["17", 2], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "flux3_edit"}},
    }


# ---- ComfyUI server management ----

def comfyui_up():
    try:
        return requests.get(BASE_URL + "/system_stats", timeout=10).status_code == 200
    except Exception:
        return False


def stop_comfyui():
    subprocess.run(["pkill", "-f", "main.py --cpu"], capture_output=True)
    for _ in range(60):
        if not comfyui_up():
            return True
        time.sleep(1)
    return False


def start_comfyui(threads):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    logf = open(COMFYUI_LOG, "ab")
    proc = subprocess.Popen(
        ["python", "main.py", "--cpu", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch"],
        cwd=COMFYUI_DIR, env=env, stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(60):
        if comfyui_up():
            log("ComfyUI up (threads=%s pid=%s)" % (threads, proc.pid))
            return True
        time.sleep(5)
    return False


def ensure_threads(n):
    if CURRENT_THREADS["n"] == n and comfyui_up():
        return True
    log("restarting ComfyUI with threads=%s" % n)
    stop_comfyui()
    time.sleep(2)
    if not start_comfyui(n):
        log("ERROR ComfyUI did not come back up (threads=%s)" % n)
        return False
    CURRENT_THREADS["n"] = n
    return True


# ---- ComfyUI API helpers ----

def queue_prompt(prompt):
    r = requests.post(BASE_URL + "/prompt", json={"prompt": prompt},
                      headers=HEADERS, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError("queue error: " + json.dumps(data["error"])[:1000])
    return data["prompt_id"]


def wait_history(prompt_id, timeout_s, poll_s=20):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(BASE_URL + "/history/" + prompt_id, headers=HEADERS, timeout=60)
        r.raise_for_status()
        entry = r.json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError("prompt error: " + json.dumps(entry)[:2000])
        time.sleep(poll_s)
    raise TimeoutError("no completion after %ss for %s" % (timeout_s, prompt_id))


def download_outputs(entry, out_dir):
    saved = []
    for node_id, node_out in (entry.get("outputs") or {}).items():
        for img in node_out.get("images", []):
            params = {"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                      "type": img.get("type", "output")}
            r = requests.get(BASE_URL + "/view", params=params, headers=HEADERS, timeout=300)
            r.raise_for_status()
            path = os.path.join(out_dir, img["filename"])
            with open(path, "wb") as f:
                f.write(r.content)
            saved.append({"file": img["filename"], "bytes": len(r.content)})
            log("saved %s (%d bytes)" % (path, len(r.content)))
    return saved


def check_nodes():
    r = requests.get(BASE_URL + "/object_info", headers=HEADERS, timeout=120)
    r.raise_for_status()
    info = r.json()
    need = ["UnetLoaderGGUF", "CLIPLoaderGGUF", "ReferenceMultiLatent",
            "EmptyFlux2LatentImage", "KSampler"]
    missing = [n for n in need if n not in info]
    log("object_info: %d node types; missing: %s" % (len(info), missing or "none"))
    return missing


def upload_image(path):
    with open(path, "rb") as f:
        r = requests.post(BASE_URL + "/upload/image",
                          files={"image": (os.path.basename(path), f, "image/png")},
                          data={"type": "input", "overwrite": "true"},
                          headers={"Authorization": "Bearer " + API_KEY}, timeout=300)
    r.raise_for_status()
    name = r.json()["name"]
    log("uploaded %s as %s" % (path, name))
    return name


def run_comfyui_test(name, prompt, timeout_s, out_dir, meta):
    log("=== test %s (timeout %ss) ===" % (name, timeout_s))
    t0 = time.time()
    rec = {"name": name, "status": "failed", "wall_seconds": 0,
           "prompt_id": None, "outputs": [], "error": None}
    rec.update(meta)
    try:
        pid = queue_prompt(prompt)
        rec["prompt_id"] = pid
        log("queued %s prompt_id=%s" % (name, pid))
        entry = wait_history(pid, timeout_s)
        rec["outputs"] = download_outputs(entry, out_dir)
        rec["status"] = "success" if rec["outputs"] else "failed"
        if not rec["outputs"]:
            rec["error"] = "completed but no output images"
    except TimeoutError as e:
        rec["status"] = "timeout"
        rec["error"] = str(e)
        log("TIMEOUT %s: %s" % (name, e))
    except Exception as e:
        rec["error"] = str(e)[:2000]
        log("FAILED %s: %s" % (name, e))
    rec["wall_seconds"] = round(time.time() - t0, 1)
    log("=== test %s: %s in %ss ===" % (name, rec["status"], rec["wall_seconds"]))
    return rec


# ---- stable-diffusion.cpp ----

def run_sdcpp_test(name, dit_file, te_file, timeout_s, out_dir, meta, log_name):
    log("=== test %s (timeout %ss) ===" % (name, timeout_s))
    t0 = time.time()
    rec = {"name": name, "status": "failed", "wall_seconds": 0,
           "prompt_id": None, "outputs": [], "error": None}
    rec.update(meta)
    out_png = os.path.join(out_dir, "sdcpp-q4-512.png")
    cmd = [SDCPP_BIN,
           "--diffusion-model", os.path.join(MODELS, "diffusion_models", dit_file),
           "--llm", os.path.join(MODELS, "text_encoders", te_file),
           "--vae", os.path.join(MODELS, "vae", VAE),
           "-p", T2I_PROMPT,
           "--cfg-scale", "1.0", "--steps", "16",
           "--sampling-method", "euler",
           "--width", "512", "--height", "512",
           "--threads", str(meta["threads"]),
           "--offload-to-cpu",
           "--output", out_png, "-v"]
    log("sdcpp cmd: %s" % " ".join(cmd[:6]) + " ...")
    try:
        with open(os.path.join(out_dir, log_name), "wb") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_s)
        if p.returncode != 0:
            rec["error"] = "sd-cli exit %d (see %s)" % (p.returncode, log_name)
            log("FAILED %s: exit %d" % (name, p.returncode))
        elif os.path.exists(out_png) and os.path.getsize(out_png) > 10000:
            rec["outputs"] = [{"file": os.path.basename(out_png),
                               "bytes": os.path.getsize(out_png)}]
            rec["status"] = "success"
        else:
            rec["error"] = "sd-cli exit 0 but no output image"
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
        rec["error"] = "sd-cli killed after %ss" % timeout_s
        log("TIMEOUT %s" % name)
    except Exception as e:
        rec["error"] = str(e)[:1000]
        log("FAILED %s: %s" % (name, e))
    rec["wall_seconds"] = round(time.time() - t0, 1)
    log("=== test %s: %s in %ss ===" % (name, rec["status"], rec["wall_seconds"]))
    return rec


def ensure_q8(out_dir):
    """Free disk (delete Q4 DiT) and download the Q8_0 DiT. Returns (ok, note)."""
    q8_path = os.path.join(MODELS, "diffusion_models", DIT_Q8)
    if os.path.exists(q8_path):
        return True, "already present"
    free_gib = shutil.disk_usage(".").free / (1024 ** 3)
    log("disk free before Q8 download: %.1f GiB" % free_gib)
    if free_gib < 6.5:
        q4_path = os.path.join(MODELS, "diffusion_models", DIT_Q4)
        if os.path.exists(q4_path):
            os.remove(q4_path)
            log("deleted Q4 DiT to free disk for Q8_0")
        free_gib = shutil.disk_usage(".").free / (1024 ** 3)
        log("disk free after deleting Q4: %.1f GiB" % free_gib)
    if free_gib < 6.5:
        return False, "not enough disk (%.1f GiB free)" % free_gib
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="webmp3/Sakura-FLUX.3-Image-Klein-Comfy-GGUF",
                        filename="diffusion_models/" + DIT_Q8, local_dir=MODELS)
    except Exception as e:
        return False, "Q8_0 download failed: " + str(e)[:500]
    if os.path.exists(q8_path):
        log("Q8_0 downloaded (%d bytes)" % os.path.getsize(q8_path))
        return True, "downloaded"
    return False, "Q8_0 download finished but file missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, choices=[384, 512, 1024])
    ap.add_argument("--run-edit", default="true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_edit = args.run_edit.lower() in ("true", "1", "yes")
    results = {"base_url": BASE_URL, "api_key_used": API_KEY,
               "model": {"dit_q4": DIT_Q4, "dit_q8": DIT_Q8, "clip": CLIP, "vae": VAE},
               "tests": []}

    def save():
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    def skipped(name, reason, meta):
        rec = {"name": name, "status": "skipped", "wall_seconds": 0,
               "prompt_id": None, "outputs": [], "error": reason}
        rec.update(meta)
        results["tests"].append(rec)
        log("skipped %s: %s" % (name, reason))
        save()
        return rec

    try:
        missing = check_nodes()
        if missing:
            log("WARNING missing nodes: %s" % missing)
    except Exception as e:
        log("object_info check failed (continuing anyway): %s" % e)

    # 1. Baseline: ComfyUI torch CPU + Q4_K_M, 512, 16 steps, default threads.
    rec_base = run_comfyui_test(
        "comfyui-q4-512-t4", build_t2i(DIT_Q4, 512, 512), 2400, args.out,
        {"runtime": "comfyui", "quant": "Q4_K_M", "resolution": "512x512",
         "steps": 16, "threads": 4})
    results["tests"].append(rec_base)
    save()

    # 2. sd.cpp GGML kernels + same Q4_K_M, same prompt, 512.
    if elapsed() < 2400:
        if os.path.exists(SDCPP_BIN):
            rec_sd = run_sdcpp_test(
                "sdcpp-q4-512-t4", DIT_Q4, CLIP, 1500, args.out,
                {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                 "steps": 16, "threads": 4}, "sdcpp-q4-512.log")
            if rec_sd["status"] != "success":
                # Retry once with the official Qwen3-VL TE if disk allows:
                # the Heretic GGUF splits the vision tower into a separate
                # mmproj file, which sd-cli may not accept.
                free_gib = shutil.disk_usage(".").free / (1024 ** 3)
                if free_gib >= 3.5:
                    try:
                        from huggingface_hub import hf_hub_download
                        hf_hub_download(repo_id=OFFICIAL_TE_REPO, filename=OFFICIAL_TE_FILE,
                                        local_dir=os.path.join(MODELS, "text_encoders"))
                        log("downloaded official TE, retrying sd.cpp")
                        rec_sd = run_sdcpp_test(
                            "sdcpp-q4-512-t4", DIT_Q4, OFFICIAL_TE_FILE, 1500, args.out,
                            {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                             "steps": 16, "threads": 4, "text_encoder": "official-qwen3vl"},
                            "sdcpp-q4-512-retry.log")
                    except Exception as e:
                        log("official TE download failed: %s" % e)
            results["tests"].append(rec_sd)
        else:
            skipped("sdcpp-q4-512-t4", "sd-cli binary missing (build failed or skipped)",
                    {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                     "steps": 16, "threads": 4})
        save()
    else:
        skipped("sdcpp-q4-512-t4", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                 "steps": 16, "threads": 4})

    # 3. Quant comparison: Q8_0 on ComfyUI, same prompt/resolution/threads.
    dit_now = DIT_Q4
    if elapsed() < 3600:
        ok, note = ensure_q8(args.out)
        if ok:
            dit_now = DIT_Q8
            if ensure_threads(4):
                rec_q8 = run_comfyui_test(
                    "comfyui-q8-512-t4", build_t2i(DIT_Q8, 512, 512), 1800, args.out,
                    {"runtime": "comfyui", "quant": "Q8_0", "resolution": "512x512",
                     "steps": 16, "threads": 4})
                results["tests"].append(rec_q8)
            else:
                skipped("comfyui-q8-512-t4", "ComfyUI restart failed",
                        {"runtime": "comfyui", "quant": "Q8_0", "resolution": "512x512",
                         "steps": 16, "threads": 4})
        else:
            skipped("comfyui-q8-512-t4", "Q8_0 unavailable: " + note,
                    {"runtime": "comfyui", "quant": "Q8_0", "resolution": "512x512",
                     "steps": 16, "threads": 4})
        save()
    else:
        skipped("comfyui-q8-512-t4", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "comfyui", "quant": "Q8_0", "resolution": "512x512",
                 "steps": 16, "threads": 4})

    # 4. Thread tuning on the winning runtime (ComfyUI), same quant as step 3.
    if elapsed() < 4500:
        if dit_now and os.path.exists(os.path.join(MODELS, "diffusion_models", dit_now)):
            if ensure_threads(2):
                rec_t2 = run_comfyui_test(
                    "comfyui-%s-512-t2" % ("q8" if dit_now == DIT_Q8 else "q4"),
                    build_t2i(dit_now, 512, 512), 1800, args.out,
                    {"runtime": "comfyui",
                     "quant": "Q8_0" if dit_now == DIT_Q8 else "Q4_K_M",
                     "resolution": "512x512", "steps": 16, "threads": 2})
                results["tests"].append(rec_t2)
                ensure_threads(4)
            else:
                skipped("comfyui-thread-t2", "ComfyUI restart failed",
                        {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                         "steps": 16, "threads": 2})
        else:
            skipped("comfyui-thread-t2", "no DiT file available",
                    {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                     "steps": 16, "threads": 2})
        save()
    else:
        skipped("comfyui-thread-t2", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                 "steps": 16, "threads": 2})

    # 5. Image edit (functional test), only if the baseline T2I succeeded.
    if rec_base["status"] == "success" and run_edit and elapsed() < 5100:
        if dit_now and os.path.exists(os.path.join(MODELS, "diffusion_models", dit_now)):
            ensure_threads(4)
            try:
                img_name = upload_image("assets/sfx-panel.png")
                rec = run_comfyui_test(
                    "edit-sfx-panel", build_edit(dit_now, img_name), 1800, args.out,
                    {"runtime": "comfyui",
                     "quant": "Q8_0" if dit_now == DIT_Q8 else "Q4_K_M",
                     "resolution": "reference-native", "steps": 16, "threads": 4})
                results["tests"].append(rec)
            except Exception as e:
                results["tests"].append(
                    {"name": "edit-sfx-panel", "status": "failed", "wall_seconds": 0,
                     "prompt_id": None, "outputs": [],
                     "error": "upload failed: " + str(e)[:1000],
                     "runtime": "comfyui", "quant": "?", "resolution": "reference-native",
                     "steps": 16, "threads": 4})
                log("FAILED edit-sfx-panel upload: %s" % e)
        else:
            skipped("edit-sfx-panel", "no DiT file available",
                    {"runtime": "comfyui", "quant": "?", "resolution": "reference-native",
                     "steps": 16, "threads": 4})
        save()
    else:
        reason = ("t2i baseline failed" if rec_base["status"] != "success" else
                  ("edit disabled" if not run_edit else
                   "out of time (%.0fs elapsed)" % elapsed()))
        skipped("edit-sfx-panel", reason,
                {"runtime": "comfyui", "quant": "?", "resolution": "reference-native",
                 "steps": 16, "threads": 4})

    # 6. 1024x1024 T2I, only if time remains.
    if rec_base["status"] == "success" and elapsed() < 5400:
        if dit_now and os.path.exists(os.path.join(MODELS, "diffusion_models", dit_now)):
            ensure_threads(4)
            rec = run_comfyui_test(
                "t2i-1024", build_t2i(dit_now, 1024, 1024), 1500, args.out,
                {"runtime": "comfyui",
                 "quant": "Q8_0" if dit_now == DIT_Q8 else "Q4_K_M",
                 "resolution": "1024x1024", "steps": 16, "threads": 4})
            results["tests"].append(rec)
        else:
            skipped("t2i-1024", "no DiT file available",
                    {"runtime": "comfyui", "quant": "?", "resolution": "1024x1024",
                     "steps": 16, "threads": 4})
        save()
    else:
        skipped("t2i-1024", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "comfyui", "quant": "?", "resolution": "1024x1024",
                 "steps": 16, "threads": 4})

    # Speed summary: fastest successful 512x512 T2I config vs baseline.
    base = next((t for t in results["tests"] if t["name"] == "comfyui-q4-512-t4"), None)
    cands = [t for t in results["tests"]
             if t["status"] == "success" and t.get("resolution") == "512x512"
             and t["name"] != "comfyui-q4-512-t4"]
    summary = {"baseline_wall_seconds": base["wall_seconds"] if base else None,
               "fastest": None, "speedup_vs_baseline": None}
    if base and base["status"] == "success" and cands:
        fastest = min(cands, key=lambda t: t["wall_seconds"])
        summary["fastest"] = {"name": fastest["name"],
                              "wall_seconds": fastest["wall_seconds"]}
        if fastest["wall_seconds"] > 0:
            summary["speedup_vs_baseline"] = round(
                base["wall_seconds"] / fastest["wall_seconds"], 2)
    results["speed_summary"] = summary
    save()
    log("ALL DONE in %.0fs" % elapsed())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
