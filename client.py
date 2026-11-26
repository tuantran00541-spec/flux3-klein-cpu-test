#!/usr/bin/env python3
"""Minimal client for the ComfyUI HTTP API.

Env vars:
  BASE_URL - ComfyUI server URL, e.g. http://127.0.0.1:8188
  API_KEY  - placeholder only; ComfyUI has no auth. Sent as a dummy
             Authorization header so the call shape matches an API
             client. Default "none".

Tests (run in order):
  1. t2i-512  - text-to-image, 512x512, 16 steps
  2. t2i-1024 - text-to-image, 1024x1024 (only if 1 succeeded and time allows)
  3. edit     - image edit of assets/sfx-panel.png (only if 1 succeeded)

Each test writes its output PNG(s) into --out and appends a record to
results.json in the same directory.
"""
import argparse
import json
import os
import sys
import time

import requests

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8188").rstrip("/")
API_KEY = os.environ.get("API_KEY", "none")
HEADERS = {"Authorization": "Bearer " + API_KEY}

DIT = "Sakura-FLUX.3-Image-Klein-Q4_K_M-3.05GiB.gguf"
CLIP = "flux3-klein-te-heretic-dhq-Q4_K_M.gguf"
VAE = "flux2-vae.safetensors"

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


def log(msg):
    print("[client] " + msg, flush=True)


def build_t2i(width, height, seed=42, steps=16):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": DIT}},
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


def build_edit(image_filename, seed=42, steps=16):
    # ReferenceMultiLatent with "keep resolution": the output latent follows
    # the reference image size, so no EmptyFlux2LatentImage is needed.
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": DIT}},
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


def run_test(name, prompt, timeout_s, out_dir):
    log("=== test %s (timeout %ss) ===" % (name, timeout_s))
    t0 = time.time()
    rec = {"name": name, "status": "failed", "wall_seconds": 0,
           "prompt_id": None, "outputs": [], "error": None}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, choices=[384, 512, 1024])
    ap.add_argument("--run-edit", default="true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_edit = args.run_edit.lower() in ("true", "1", "yes")
    t_start = time.time()
    results = {"base_url": BASE_URL, "api_key_used": API_KEY,
               "model": {"dit": DIT, "clip": CLIP, "vae": VAE}, "tests": []}

    def save():
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    try:
        missing = check_nodes()
        if missing:
            log("WARNING missing nodes: %s (tests will likely fail)" % missing)
    except Exception as e:
        log("object_info check failed (continuing anyway): %s" % e)

    # Test 1: text-to-image at requested size.
    rec1 = run_test("t2i-%d" % args.size, build_t2i(args.size, args.size),
                    timeout_s=2400, out_dir=args.out)
    results["tests"].append(rec1)
    save()
    elapsed = time.time() - t_start

    # Test 1b: 1024 if the first size was smaller, succeeded, and time allows.
    if rec1["status"] == "success" and args.size < 1024 and elapsed < 3300:
        rec = run_test("t2i-1024", build_t2i(1024, 1024), timeout_s=1500, out_dir=args.out)
        results["tests"].append(rec)
        save()
        elapsed = time.time() - t_start

    # Test 2: image edit, only if T2I succeeded and time remains.
    if rec1["status"] == "success" and run_edit and elapsed < 5400:
        try:
            img_name = upload_image("assets/sfx-panel.png")
            rec = run_test("edit-sfx-panel", build_edit(img_name),
                           timeout_s=1800, out_dir=args.out)
        except Exception as e:
            rec = {"name": "edit-sfx-panel", "status": "failed", "wall_seconds": 0,
                   "prompt_id": None, "outputs": [], "error": "upload failed: " + str(e)[:1000]}
            log("FAILED edit-sfx-panel upload: %s" % e)
        results["tests"].append(rec)
        save()
    else:
        reason = "t2i failed" if rec1["status"] != "success" else (
            "edit disabled" if not run_edit else "out of time (%.0fs elapsed)" % elapsed)
        results["tests"].append({"name": "edit-sfx-panel", "status": "skipped",
                                 "wall_seconds": 0, "prompt_id": None, "outputs": [],
                                 "error": reason})
        log("edit test skipped: " + reason)
        save()

    log("ALL DONE in %.0fs" % (time.time() - t_start))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
