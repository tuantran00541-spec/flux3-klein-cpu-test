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
  3. edit-sfx-panel      ComfyUI image edit of assets/sfx-panel.png
  4. sdcpp-q8-512-t4     sd.cpp + GGUF Q8_0 (quant comparison)
  5. sdcpp-q6-512-t4     sd.cpp + GGUF Q6_K (quant comparison)
  6. thread-t2           2 threads on the winning runtime (thread tuning)
  7. comfyui-ipex-*      ComfyUI with IPEX enabled (only if installed)
  8. fastest-12step      12 steps on the fastest runtime (step probe)
  9. t2i-1024            ComfyUI 1024x1024 (only if time allows)

OpenVINO: NOT attempted. OpenVINO GenAI's official supported-models page
lists Flux, FLUX.2 [klein], Qwen-Image, Qwen-Image-2.1, Z-Image for image
generation - no FLUX.3 Klein entry; Optimum Intel v2.1.0's newest FLUX
support is FLUX.2, there is no OVFlux3Pipeline. The diffusers checkpoint
is also gated (no HF token on the runner). Export would fail on the
missing architecture mapping, so no run time is spent on it.

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
DIT_Q6 = "Sakura-FLUX.3-Image-Klein-Q6_K-4.35GiB.gguf"
DIT_Q8 = "Sakura-FLUX.3-Image-Klein-Q8_0-5.58GiB.gguf"
CLIP = "flux3-klein-te-heretic-dhq-Q4_K_M.gguf"
OFFICIAL_TE_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
OFFICIAL_TE_FILE = "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
VAE = "flux2-vae.safetensors"
MODELS = "ComfyUI/models"
SDCPP_BIN = "sd.cpp/build/bin/sd-cli"
COMFYUI_DIR = "ComfyUI"
COMFYUI_LOG = "comfyui.log"
SDCLI_HELP_FILE = "sd-cli-help.txt"

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
IPEX_OK = os.path.exists("ipex_ok")
# The workflow starts ComfyUI with --disable-ipex-optimize when ipex_ok
# exists, so the baseline matrix is pure torch CPU (comparable to run 1).
CURRENT = {"threads": 4, "disable_ipex": IPEX_OK}
SERVER_META = {}
SDCLI_HELP = ""


def elapsed():
    return time.time() - T_START


def log(msg):
    print("[client %.0fs] %s" % (elapsed(), msg), flush=True)


def load_sdcli_help():
    global SDCLI_HELP
    try:
        with open(SDCLI_HELP_FILE) as f:
            SDCLI_HELP = f.read()
        log("sd-cli help loaded (%d chars)" % len(SDCLI_HELP))
    except Exception as e:
        log("sd-cli help file missing (%s); optional sd.cpp flags disabled" % e)


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
    # NOTE: the "images" input must be a nested dict in API format; a flat
    # "image0" key is rejected (run 45832838411: ReferenceMultiLatent got an
    # unexpected keyword argument 'image0').
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": dit}},
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": CLIP, "type": "flux3"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": EDIT_PROMPT, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "17": {"class_type": "ReferenceMultiLatent", "inputs": {
            "positive": ["4", 0], "negative": ["5", 0], "vae": ["3", 0],
            "resolution": "keep resolution", "images": {"image0": ["11", 0]}}},
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


def start_comfyui(threads, disable_ipex=False):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    cmd = ["python", "main.py", "--cpu", "--listen", "127.0.0.1", "--port", "8188",
           "--disable-auto-launch"]
    if disable_ipex:
        cmd.append("--disable-ipex-optimize")
    logf = open(COMFYUI_LOG, "ab")
    proc = subprocess.Popen(cmd, cwd=COMFYUI_DIR, env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(60):
        if comfyui_up():
            nlwp = ""
            try:
                nlwp = subprocess.run(["ps", "-o", "nlwp=", "-p", str(proc.pid)],
                                      capture_output=True, text=True,
                                      timeout=10).stdout.strip()
            except Exception:
                pass
            SERVER_META.update({
                "nproc": os.cpu_count(),
                "omp_num_threads": threads, "mkl_num_threads": threads,
                "comfyui_pid": proc.pid, "comfyui_nlwp": nlwp,
                "ipex_disabled": disable_ipex})
            log("ComfyUI up (threads=%s pid=%s nlwp=%s ipex_disabled=%s)"
                % (threads, proc.pid, nlwp, disable_ipex))
            return True
        time.sleep(5)
    return False


def ensure_threads(n, disable_ipex=None):
    # Default: the baseline matrix runs without IPEX (if installed) so it
    # compares cleanly to run 1; the IPEX test passes disable_ipex=False.
    if disable_ipex is None:
        disable_ipex = IPEX_OK
    if (CURRENT["threads"] == n and CURRENT["disable_ipex"] == disable_ipex
            and comfyui_up()):
        return True
    log("restarting ComfyUI with threads=%s ipex_disabled=%s" % (n, disable_ipex))
    stop_comfyui()
    time.sleep(2)
    if not start_comfyui(n, disable_ipex):
        log("ERROR ComfyUI did not come back up (threads=%s)" % n)
        return False
    CURRENT["threads"] = n
    CURRENT["disable_ipex"] = disable_ipex
    return True


def grep_ipex_log():
    try:
        with open(COMFYUI_LOG, errors="ignore") as f:
            hits = [ln.strip() for ln in f if "ipex" in ln.lower()]
        return hits[:10]
    except Exception:
        return []


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
    rec["server_meta"] = dict(SERVER_META)
    if IPEX_OK:
        rec["ipex"] = "disabled" if CURRENT["disable_ipex"] else "enabled"
    else:
        rec["ipex"] = "not-installed"
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

def run_sdcpp_test(name, dit_file, te_file, timeout_s, out_dir, meta, log_name,
                   steps=16, width=512, height=512):
    log("=== test %s (timeout %ss) ===" % (name, timeout_s))
    t0 = time.time()
    rec = {"name": name, "status": "failed", "wall_seconds": 0,
           "prompt_id": None, "outputs": [], "error": None}
    rec.update(meta)
    out_png = os.path.join(out_dir, "sdcpp-" + name + ".png")
    cmd = [SDCPP_BIN,
           "--diffusion-model", os.path.join(MODELS, "diffusion_models", dit_file),
           "--llm", os.path.join(MODELS, "text_encoders", te_file),
           "--vae", os.path.join(MODELS, "vae", VAE),
           "-p", T2I_PROMPT,
           "--cfg-scale", "1.0", "--steps", str(steps),
           "--sampling-method", "euler",
           "--width", str(width), "--height", str(height),
           "--threads", str(meta["threads"]),
           "--output", out_png, "-v"]
    # Only pass optional flags the installed sd-cli actually supports
    # (checked against the --help captured by the workflow).
    if "--offload-to-cpu" in SDCLI_HELP:
        cmd.append("--offload-to-cpu")
    if "--cache-mode" in SDCLI_HELP:
        cmd += ["--cache-mode", "easycache"]
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


def run_sdcpp_with_fallback(name, dit_file, timeout_s, out_dir, meta, log_name,
                            steps=16, width=512, height=512):
    """Try the official Qwen3-VL TE first; fall back to the Heretic TE.

    The Heretic GGUF splits the vision tower into a separate mmproj file,
    which sd-cli may not accept; the official single-file TE is the safer
    first attempt. Returns the successful record, or the last failure.
    """
    candidates = []
    if os.path.exists(os.path.join(MODELS, "text_encoders", OFFICIAL_TE_FILE)):
        candidates.append((OFFICIAL_TE_FILE, "official-qwen3vl"))
    candidates.append((CLIP, "heretic-dhq"))
    last = None
    for i, (te_file, te_label) in enumerate(candidates):
        m = dict(meta)
        m["text_encoder"] = te_label
        ln = log_name if i == 0 else log_name.replace(".log", "-retry.log")
        rec = run_sdcpp_test(name, dit_file, te_file, timeout_s, out_dir,
                             m, ln, steps=steps, width=width, height=height)
        last = rec
        if rec["status"] == "success":
            return rec
        log("sdcpp with %s failed (%s), trying next TE"
            % (te_label, (rec.get("error") or "")[:200]))
    return last


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


def ensure_q6(out_dir):
    """Download the Q6_K DiT. Returns (ok, note)."""
    q6_path = os.path.join(MODELS, "diffusion_models", DIT_Q6)
    if os.path.exists(q6_path):
        return True, "already present"
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="webmp3/Sakura-FLUX.3-Image-Klein-Comfy-GGUF",
                        filename="diffusion_models/" + DIT_Q6, local_dir=MODELS)
    except Exception as e:
        return False, "Q6_K download failed: " + str(e)[:500]
    if os.path.exists(q6_path):
        log("Q6_K downloaded (%d bytes)" % os.path.getsize(q6_path))
        return True, "downloaded"
    return False, "Q6_K download finished but file missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, choices=[384, 512, 1024])
    ap.add_argument("--run-edit", default="true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_edit = args.run_edit.lower() in ("true", "1", "yes")
    results = {"base_url": BASE_URL, "api_key_used": API_KEY,
               "model": {"dit_q4": DIT_Q4, "dit_q6": DIT_Q6, "dit_q8": DIT_Q8,
                         "clip": CLIP, "official_te": OFFICIAL_TE_FILE, "vae": VAE},
               "ipex_installed": IPEX_OK,
               "openvino": {
                   "supported": False,
                   "evidence": ("OpenVINO GenAI supported-models page (2026-11-26) lists "
                                "Flux, FLUX.2 [klein], Qwen-Image, Qwen-Image-2.1, Z-Image; "
                                "no FLUX.3 Klein. Optimum Intel v2.1.0 newest FLUX support "
                                "is FLUX.2; no OVFlux3Pipeline. Diffusers checkpoint is "
                                "gated (no HF token on runner). No export attempted.")},
               "tests": []}
    load_sdcli_help()

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

    # Pre-flight: the sd-cli binary path was wrong in run 1 (sd.cpp/bin/...).
    # Never silently skip sd.cpp again: record a loud failure if missing.
    sdcpp_available = os.path.exists(SDCPP_BIN)
    if sdcpp_available:
        log("sd-cli binary present: %s" % SDCPP_BIN)
    else:
        log("ERROR sd-cli binary missing at %s" % SDCPP_BIN)

    try:
        missing = check_nodes()
        if missing:
            log("WARNING missing nodes: %s" % missing)
    except Exception as e:
        log("object_info check failed (continuing anyway): %s" % e)

    dit_now = DIT_Q4
    sdcpp_q4_ok = False

    # 1. Baseline: ComfyUI torch CPU + Q4_K_M, 512, 16 steps, default threads.
    # (IPEX disabled here when installed, so this compares to run 1.)
    rec_base = run_comfyui_test(
        "comfyui-q4-512-t4", build_t2i(DIT_Q4, 512, 512), 2400, args.out,
        {"runtime": "comfyui", "quant": "Q4_K_M", "resolution": "512x512",
         "steps": 16, "threads": 4})
    results["tests"].append(rec_base)
    save()

    # 2. sd.cpp GGML kernels + same Q4_K_M, same prompt, 512.
    if elapsed() < 2400:
        if sdcpp_available:
            rec_sd = run_sdcpp_with_fallback(
                "sdcpp-q4-512-t4", DIT_Q4, 1500, args.out,
                {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                 "steps": 16, "threads": 4}, "sdcpp-q4-512.log")
            sdcpp_q4_ok = rec_sd["status"] == "success"
            results["tests"].append(rec_sd)
        else:
            results["tests"].append(
                {"name": "sdcpp-q4-512-t4", "status": "failed", "wall_seconds": 0,
                 "prompt_id": None, "outputs": [],
                 "error": "sd-cli binary missing at sd.cpp/build/bin/sd-cli",
                 "runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                 "steps": 16, "threads": 4})
            log("FAILED sdcpp-q4-512-t4: sd-cli binary missing")
        save()
    else:
        skipped("sdcpp-q4-512-t4", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                 "steps": 16, "threads": 4})

    # 3. Image edit (functional test), only if the baseline T2I succeeded.
    if rec_base["status"] == "success" and run_edit and elapsed() < 3600:
        if os.path.exists(os.path.join(MODELS, "diffusion_models", dit_now)):
            ensure_threads(4)
            try:
                img_name = upload_image("assets/sfx-panel.png")
                rec = run_comfyui_test(
                    "edit-sfx-panel", build_edit(dit_now, img_name), 1800, args.out,
                    {"runtime": "comfyui", "quant": "Q4_K_M",
                     "resolution": "reference-native", "steps": 16, "threads": 4})
                results["tests"].append(rec)
            except Exception as e:
                results["tests"].append(
                    {"name": "edit-sfx-panel", "status": "failed", "wall_seconds": 0,
                     "prompt_id": None, "outputs": [],
                     "error": "upload failed: " + str(e)[:1000],
                     "runtime": "comfyui", "quant": "Q4_K_M",
                     "resolution": "reference-native", "steps": 16, "threads": 4})
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

    # 4. sd.cpp quant comparison: Q8_0, same prompt/resolution/threads.
    if sdcpp_q4_ok and elapsed() < 4200:
        ok, note = ensure_q8(args.out)
        if ok:
            dit_now = DIT_Q8
            rec = run_sdcpp_with_fallback(
                "sdcpp-q8-512-t4", DIT_Q8, 1500, args.out,
                {"runtime": "sdcpp", "quant": "Q8_0", "resolution": "512x512",
                 "steps": 16, "threads": 4}, "sdcpp-q8-512.log")
            results["tests"].append(rec)
        else:
            skipped("sdcpp-q8-512-t4", "Q8_0 unavailable: " + note,
                    {"runtime": "sdcpp", "quant": "Q8_0", "resolution": "512x512",
                     "steps": 16, "threads": 4})
        save()
    else:
        skipped("sdcpp-q8-512-t4",
                "skipped (sdcpp Q4 failed or out of time, %.0fs elapsed)" % elapsed(),
                {"runtime": "sdcpp", "quant": "Q8_0", "resolution": "512x512",
                 "steps": 16, "threads": 4})

    # 5. sd.cpp quant comparison: Q6_K.
    if sdcpp_q4_ok and elapsed() < 4800:
        ok, note = ensure_q6(args.out)
        if ok:
            rec = run_sdcpp_with_fallback(
                "sdcpp-q6-512-t4", DIT_Q6, 1500, args.out,
                {"runtime": "sdcpp", "quant": "Q6_K", "resolution": "512x512",
                 "steps": 16, "threads": 4}, "sdcpp-q6-512.log")
            results["tests"].append(rec)
        else:
            skipped("sdcpp-q6-512-t4", "Q6_K unavailable: " + note,
                    {"runtime": "sdcpp", "quant": "Q6_K", "resolution": "512x512",
                     "steps": 16, "threads": 4})
        save()
    else:
        skipped("sdcpp-q6-512-t4",
                "skipped (sdcpp Q4 failed or out of time, %.0fs elapsed)" % elapsed(),
                {"runtime": "sdcpp", "quant": "Q6_K", "resolution": "512x512",
                 "steps": 16, "threads": 4})

    # 6. Thread tuning on the winning runtime (2 threads vs 4).
    if elapsed() < 5400:
        if sdcpp_q4_ok:
            rec = run_sdcpp_with_fallback(
                "sdcpp-q4-512-t2", DIT_Q4, 1500, args.out,
                {"runtime": "sdcpp", "quant": "Q4_K_M", "resolution": "512x512",
                 "steps": 16, "threads": 2}, "sdcpp-q4-512-t2.log")
            results["tests"].append(rec)
        elif os.path.exists(os.path.join(MODELS, "diffusion_models", dit_now)):
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
            skipped("thread-t2", "no DiT file available",
                    {"runtime": "?", "quant": "?", "resolution": "512x512",
                     "steps": 16, "threads": 2})
        save()
    else:
        skipped("thread-t2", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "?", "quant": "?", "resolution": "512x512",
                 "steps": 16, "threads": 2})

    # 7. IPEX: one ComfyUI 512 test with IPEX enabled (same quant as the
    # fastest successful ComfyUI config), only if IPEX installed.
    if IPEX_OK and elapsed() < 6300:
        fast_comfyui = [t for t in results["tests"]
                        if t["status"] == "success" and t.get("runtime") == "comfyui"
                        and t.get("resolution") == "512x512" and t.get("steps") == 16]
        ipex_quant = DIT_Q4
        if fast_comfyui:
            best = min(fast_comfyui, key=lambda t: t["wall_seconds"])
            ipex_quant = DIT_Q8 if best.get("quant") == "Q8_0" else DIT_Q4
        if os.path.exists(os.path.join(MODELS, "diffusion_models", ipex_quant)):
            if ensure_threads(4, disable_ipex=False):
                rec = run_comfyui_test(
                    "comfyui-ipex-%s-512-t4" % ("q8" if ipex_quant == DIT_Q8 else "q4"),
                    build_t2i(ipex_quant, 512, 512), 2400, args.out,
                    {"runtime": "comfyui",
                     "quant": "Q8_0" if ipex_quant == DIT_Q8 else "Q4_K_M",
                     "resolution": "512x512", "steps": 16, "threads": 4})
                rec["ipex_log_hits"] = grep_ipex_log()
                results["tests"].append(rec)
                ensure_threads(4)  # back to IPEX-disabled state
            else:
                skipped("comfyui-ipex", "ComfyUI restart failed",
                        {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                         "steps": 16, "threads": 4})
        else:
            skipped("comfyui-ipex", "no DiT file available",
                    {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                     "steps": 16, "threads": 4})
        save()
    elif IPEX_OK:
        skipped("comfyui-ipex", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                 "steps": 16, "threads": 4})
    else:
        skipped("comfyui-ipex", "IPEX not installed (no ipex_ok marker)",
                {"runtime": "comfyui", "quant": "?", "resolution": "512x512",
                 "steps": 16, "threads": 4})

    # 8. 12-step probe on the fastest runtime (16-step successes only).
    if elapsed() < 6600:
        cands16 = [t for t in results["tests"]
                   if t["status"] == "success" and t.get("resolution") == "512x512"
                   and t.get("steps") == 16]
        if cands16:
            fastest = min(cands16, key=lambda t: t["wall_seconds"])
            if fastest["runtime"] == "sdcpp":
                rec = run_sdcpp_with_fallback(
                    "sdcpp-%s-512-12step-t4" % fastest["quant"].lower().replace("_", ""),
                    DIT_Q4 if fastest["quant"] == "Q4_K_M"
                    else (DIT_Q6 if fastest["quant"] == "Q6_K" else DIT_Q8),
                    1500, args.out,
                    {"runtime": "sdcpp", "quant": fastest["quant"],
                     "resolution": "512x512", "steps": 12, "threads": 4},
                    "sdcpp-12step.log", steps=12)
                results["tests"].append(rec)
            else:
                dit12 = DIT_Q8 if fastest["quant"] == "Q8_0" else DIT_Q4
                ensure_threads(4)
                rec = run_comfyui_test(
                    "comfyui-%s-512-12step-t4" % ("q8" if dit12 == DIT_Q8 else "q4"),
                    build_t2i(dit12, 512, 512, steps=12), 2400, args.out,
                    {"runtime": "comfyui",
                     "quant": "Q8_0" if dit12 == DIT_Q8 else "Q4_K_M",
                     "resolution": "512x512", "steps": 12, "threads": 4})
                results["tests"].append(rec)
            save()
        else:
            skipped("fastest-12step", "no successful 16-step 512 test to base on",
                    {"runtime": "?", "quant": "?", "resolution": "512x512",
                     "steps": 12, "threads": 4})
    else:
        skipped("fastest-12step", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "?", "quant": "?", "resolution": "512x512",
                 "steps": 12, "threads": 4})

    # 9. 1024x1024 T2I with the fastest config, only if time remains.
    if rec_base["status"] == "success" and elapsed() < 6000:
        cands16 = [t for t in results["tests"]
                   if t["status"] == "success" and t.get("resolution") == "512x512"
                   and t.get("steps") == 16]
        if cands16:
            fastest = min(cands16, key=lambda t: t["wall_seconds"])
            if fastest["runtime"] == "sdcpp":
                dit1024 = (DIT_Q4 if fastest["quant"] == "Q4_K_M"
                           else (DIT_Q6 if fastest["quant"] == "Q6_K" else DIT_Q8))
                rec = run_sdcpp_with_fallback(
                    "sdcpp-%s-1024-t4" % fastest["quant"].lower().replace("_", ""),
                    dit1024, 1500, args.out,
                    {"runtime": "sdcpp", "quant": fastest["quant"],
                     "resolution": "1024x1024", "steps": 16, "threads": 4},
                    "sdcpp-1024.log", width=1024, height=1024)
                results["tests"].append(rec)
            else:
                dit1024 = DIT_Q8 if fastest["quant"] == "Q8_0" else DIT_Q4
                ensure_threads(4)
                rec = run_comfyui_test(
                    "t2i-1024", build_t2i(dit1024, 1024, 1024), 1500, args.out,
                    {"runtime": "comfyui",
                     "quant": "Q8_0" if dit1024 == DIT_Q8 else "Q4_K_M",
                     "resolution": "1024x1024", "steps": 16, "threads": 4})
                results["tests"].append(rec)
            save()
        else:
            skipped("t2i-1024", "no successful 512 test to base on",
                    {"runtime": "?", "quant": "?", "resolution": "1024x1024",
                     "steps": 16, "threads": 4})
    else:
        skipped("t2i-1024", "out of time (%.0fs elapsed)" % elapsed(),
                {"runtime": "?", "quant": "?", "resolution": "1024x1024",
                 "steps": 16, "threads": 4})

    # Speed summary: fastest successful 512x512 16-step T2I vs baseline.
    base = next((t for t in results["tests"] if t["name"] == "comfyui-q4-512-t4"), None)
    cands = [t for t in results["tests"]
             if t["status"] == "success" and t.get("resolution") == "512x512"
             and t.get("steps") == 16 and t["name"] != "comfyui-q4-512-t4"]
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
