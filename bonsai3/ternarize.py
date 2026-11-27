#!/usr/bin/env python3
"""DIY Bonsai-style ternarization of FLUX 3 Image Klein (BF16 -> ternary BF16).

Round 3: same recipe as round 1, but GROUP 128 (PrismML's group size;
group 256 already proven dead in F16, round 2 fix1). The ternary weights
are applied DIRECTLY to the full-precision BASE weights -- never to any
quantized intermediate (user order 2026-11-27: compressing from Q2 is
dead from the egg).

  - Target: the matmul-heavy DiT linears (q/k/v projections, attn_out, mlp_in,
    mlp_out) in every double/single block. Everything else (embedders,
    modulation streams, time embeds, final layer, norms, biases) is copied
    byte-identical in BF16 -- the same <5% exception tail as the Bonsai
    whitepaper (proj_out / x_embedder / context_embedder / time_text_embed /
    time_guidance_embed / norm_out / three modulation streams).
  - Grouping: blocks of 128 weights along the last dim, matching PrismML's
    TQ2G128 group size. The GGUF packing is done by bonsai3/pack_gguf.py
    (own writer, 68-byte blocks, type 35) -- NOT by sd-cli's tq2_0 converter.
    Pre-ternarized values that are exact multiples of one per-128-group
    scale pass through the packer EXACTLY (verified: dequant == input).
  - Scale fit (default "mse"): per 128-block, grid-search the scale s that
    minimizes ||w - s * clip(round(w/s), -1, 1)||^2 over candidates
    s = c * mean|w|, c in [0.5, 2.0]. This is strictly better MSE than the
    amax rule (s = max|w|), which is outlier-sensitive.
    "amax" rule is available for comparison (the vendor's scale rule:
    the Prism unpacked model's per-group scale IS the group amax, proven
    byte-exact against the TokForge GGUF).
  - Binary mode (--binary): t in {-1, +1}, t = sign(w), same MSE grid on s.

I/O: streams tensor-by-tensor (no full-model load; peak RAM is one tensor
chunk). Output is the same safetensors layout as the input (same names,
shapes, dtypes, offsets) -- implemented as: copy input file, then overwrite
only the ternarized tensors' data regions in place. Next step:
  sd-cli -M convert --diffusion-model <out> --type f16 -o ternary-f16.gguf
  python bonsai3/pack_gguf.py --input ternary-f16.gguf --output ternary-t35.gguf

Usage:
  python bonsai3/ternarize.py --input in.safetensors --output out.safetensors \
      --scale-rule mse --group 128 --stats-out results/ternarize-stats.json
"""
import argparse
import json
import os
import shutil
import struct
import time

import torch
from safetensors import safe_open

TARGET_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".attn_out.weight",
    ".mlp_in.weight",
    ".mlp_out.weight",
)


def is_target(name, shape):
    if len(shape) != 2:
        return False
    return any(name.endswith(s) for s in TARGET_SUFFIXES)


def ternarize_chunk(w, scale_rule, group, binary, n_candidates=12):
    """w: float32 tensor (rows, cols), cols % group == 0.
    Returns (w_ternary float32, scales (rows, nblocks), mse, mse_amax)."""
    rows, cols = w.shape
    nb = cols // group
    wb = w.reshape(rows, nb, group)
    absmean = wb.abs().mean(dim=-1)  # (rows, nb)
    if binary:
        t_sign = torch.where(wb >= 0, 1.0, -1.0)
        best_s, best_err, best_t = None, None, None
        for c in torch.linspace(0.5, 2.0, n_candidates):
            s = c * absmean
            s_safe = torch.where(s > 0, s, torch.ones_like(s))
            err = ((wb - s_safe.unsqueeze(-1) * t_sign) ** 2).mean(dim=-1)
            if best_err is None:
                best_s, best_err, best_t = s_safe, err, t_sign
            else:
                better = err < best_err
                best_s = torch.where(better, s_safe, best_s)
                best_err = torch.where(better, err, best_err)
        w_out = (best_s.unsqueeze(-1) * best_t).reshape(rows, cols)
        mse = float(best_err.mean())
    else:
        best_s, best_err, best_t = None, None, None
        for c in torch.linspace(0.5, 2.0, n_candidates):
            s = c * absmean
            s_safe = torch.where(s > 0, s, torch.ones_like(s))
            t = torch.clamp(torch.round(wb / s_safe.unsqueeze(-1)), -1.0, 1.0)
            err = ((wb - s_safe.unsqueeze(-1) * t) ** 2).mean(dim=-1)
            if best_err is None:
                best_s, best_err, best_t = s_safe, err, t
            else:
                better = err < best_err
                best_s = torch.where(better, s_safe, best_s)
                best_err = torch.where(better, err, best_err)
                best_t = torch.where(better.unsqueeze(-1), t, best_t)
        w_out = (best_s.unsqueeze(-1) * best_t).reshape(rows, cols)
        mse = float(best_err.mean())
    # amax-rule MSE for comparison (what the converter would do natively)
    amax = wb.abs().amax(dim=-1)
    amax_safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    t_amax = torch.clamp(torch.round(wb / amax_safe.unsqueeze(-1)), -1.0, 1.0)
    mse_amax = float(((wb - amax_safe.unsqueeze(-1) * t_amax) ** 2).mean())
    zero_frac = float((best_t == 0).float().mean()) if not binary else 0.0
    return w_out, best_s, mse, mse_amax, zero_frac


def ggml_tq2_identity_ok(w_bf16, group):
    """Simulate the TQ2G128 packer (amax scale per `group`, round to
    {-1,0,1}) on BF16 values and check dequant(pack(w)) == w exactly,
    per group. Ternary values that are exact multiples of one per-group
    scale survive the packer unchanged."""
    w = w_bf16.float()
    rows, cols = w.shape
    wb = w.reshape(rows, cols // group, group)
    amax = wb.abs().amax(dim=-1, keepdim=True)
    amax_safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    t = torch.clamp(torch.round(wb / amax_safe), -1.0, 1.0)
    recon = (amax_safe * t).reshape(rows, cols)
    return bool(torch.equal(recon, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale-rule", default="mse", choices=["mse", "amax"])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--row-chunk", type=int, default=1024)
    args = ap.parse_args()
    binary = args.binary
    group = args.group
    t_start = time.time()

    def log(msg):
        print("[ternarize %.0fs] %s" % (time.time() - t_start, msg), flush=True)

    # Read the input header (names, shapes, offsets).
    with open(args.input, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    names = [k for k in header if k != "__metadata__"]
    log("input %s: %d tensors" % (args.input, len(names)))

    shutil.copyfile(args.input, args.output)
    log("copied input -> output (same layout, %d bytes)" % os.path.getsize(args.output))

    stats = {"recipe": {"scale_rule": args.scale_rule, "group": group,
                         "binary": binary,
                         "targets": list(TARGET_SUFFIXES)},
             "tensors": {}, "totals": {}}
    n_tern, params_tern, params_total = 0, 0, 0
    mse_sum, mse_amax_sum, zero_sum = 0.0, 0.0, 0.0

    f_out = open(args.output, "r+b")
    with safe_open(args.input, framework="pt") as f:
        for name in names:
            meta = header[name]
            shape = meta["shape"]
            n_params = 1
            for d in shape:
                n_params *= d
            params_total += n_params
            if not is_target(name, shape):
                continue
            if shape[-1] % group != 0:
                log("SKIP %s shape %s (last dim not divisible by %d)" % (name, shape, group))
                continue
            t0 = time.time()
            w = f.get_tensor(name)  # BF16, CPU
            assert str(w.dtype) == "torch.bfloat16", (name, w.dtype)
            w32 = w.float()
            out32 = torch.empty_like(w32)
            mse_t, mse_amax_t, zero_t = 0.0, 0.0, 0.0
            n_chunks = 0
            for r0 in range(0, w32.shape[0], args.row_chunk):
                chunk = w32[r0:r0 + args.row_chunk]
                if args.scale_rule == "amax":
                    # reproduce the converter-native rule exactly
                    wb = chunk.reshape(chunk.shape[0], -1, group)
                    amax = wb.abs().amax(dim=-1, keepdim=True)
                    amax_safe = torch.where(amax > 0, amax, torch.ones_like(amax))
                    t = torch.clamp(torch.round(wb / amax_safe), -1.0, 1.0)
                    out_chunk = (amax_safe * t).reshape(chunk.shape)
                    mse_c = float(((chunk - out_chunk) ** 2).mean())
                    mse_amax_c = mse_c
                    zero_c = float((t == 0).float().mean()) if not binary else 0.0
                    scales_c = amax_safe.squeeze(-1)
                else:
                    out_chunk, scales_c, mse_c, mse_amax_c, zero_c = ternarize_chunk(
                        chunk, args.scale_rule, group, binary)
                out32[r0:r0 + args.row_chunk] = out_chunk
                mse_t += mse_c * chunk.numel()
                mse_amax_t += mse_amax_c * chunk.numel()
                zero_t += zero_c * chunk.numel()
                n_chunks += 1
            n_elem = w32.numel()
            mse_t /= n_elem
            mse_amax_t /= n_elem
            zero_t /= n_elem
            # write BF16 bytes back at the tensor's offset in the output file
            data_off = meta["data_offsets"]
            file_off = 8 + header_len + data_off[0]
            b = out32.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()
            assert len(b) == data_off[1] - data_off[0], (name, len(b), data_off)
            f_out.seek(file_off)
            f_out.write(b)
            n_tern += 1
            params_tern += n_elem
            mse_sum += mse_t * n_elem
            mse_amax_sum += mse_amax_t * n_elem
            zero_sum += zero_t * n_elem
            stats["tensors"][name] = {"params": n_elem, "shape": shape,
                                       "mse": mse_t, "mse_amax": mse_amax_t,
                                       "zero_frac": zero_t,
                                       "seconds": round(time.time() - t0, 1)}
            log("%s %s params=%d mse=%.6g mse_amax=%.6g zero=%.3f (%.0fs)"
                % (name, shape, n_elem, mse_t, mse_amax_t, zero_t, time.time() - t0))
    f_out.close()

    stats["totals"] = {
        "tensors_ternarized": n_tern,
        "params_ternarized": params_tern,
        "params_total": params_total,
        "params_ternarized_frac": params_tern / max(params_total, 1),
        "mean_mse": mse_sum / max(params_tern, 1),
        "mean_mse_amax": mse_amax_sum / max(params_tern, 1),
        "mean_zero_frac": zero_sum / max(params_tern, 1),
        "seconds": round(time.time() - t_start, 1),
    }
    log("DONE: %d tensors, %.3fB/%.3fB params ternarized (%.1f%%), "
        "mean mse=%.6g (amax %.6g), mean zero=%.3f, total %.0fs"
        % (n_tern, params_tern / 1e9, params_total / 1e9,
           100.0 * params_tern / max(params_total, 1),
           stats["totals"]["mean_mse"], stats["totals"]["mean_mse_amax"],
           stats["totals"]["mean_zero_frac"], time.time() - t_start))

    # Verification: sampled target tensors must survive a GGML TQ2
    # amax-quantize round-trip exactly (else the converter would change them).
    with safe_open(args.output, framework="pt") as f:
        targets = [n for n in names if is_target(n, header[n]["shape"])
                   and header[n]["shape"][-1] % group == 0]
        sample = targets[::max(1, len(targets) // 8)][:8]
        bad = []
        for name in sample:
            w = f.get_tensor(name)
            if not ggml_tq2_identity_ok(w, group):
                bad.append(name)
        log("verify GGML-TQ2 identity on %d sampled tensors: %s"
            % (len(sample), "OK" if not bad else "MISMATCH " + str(bad)))
        stats["verify"] = {"sampled": sample, "mismatched": bad}
        # non-target tensors must be byte-identical to the input
        with safe_open(args.input, framework="pt") as fi:
            nontarget = [n for n in names if not is_target(n, header[n]["shape"])][:5]
            ok = all(torch.equal(f.get_tensor(n), fi.get_tensor(n)) for n in nontarget)
        log("verify non-target tensors byte-identical (sample %d): %s"
            % (len(nontarget), "OK" if ok else "MISMATCH"))
        stats["verify"]["nontarget_sample_ok"] = bool(ok)

    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        with open(args.stats_out, "w") as fp:
            json.dump(stats, fp, indent=2)
        log("stats written to %s" % args.stats_out)
    print(json.dumps(stats["totals"], indent=2))


if __name__ == "__main__":
    main()
