#!/usr/bin/env python3
"""Quantize a diffusion GGUF's target linears to Q2_K (reference algorithm).

Faithful port of ggml quantize_row_q2_K_ref + make_qkx2_quants
(ggml-org/ggml, src/ggml-quants.c) into vectorized numpy, plus a minimal
GGUF reader/writer. Only the matmul-heavy DiT linears are quantized
(same target list as bonsai/ternarize.py); every other tensor is copied
byte-identical (F16/BF16 as in the source).

Q2_K block (84 bytes / 256 values): fp16 d, fp16 dmin, 16 scale bytes
(low nibble = sub-block scale 0..15, high nibble = sub-block min 0..15),
64 qs bytes. Value at position p (sub-block s = p//16):
dl = d*(scales[s]&0xF), ml = dmin*(scales[s]>>4),
code from byte (p%32)+32*(p//128), bits 2*((p//32)%4); y = dl*code - ml.

Why Q2_K and not TQ2: sd-cli's converter rejects q2_k/q3_k, and
llama-quantize cannot read sd.cpp GGUFs (a tensor name is 65 chars,
over llama.cpp's 64-char limit). So the reference quantizer is ported
here instead. Q2_K has per-16 sub-block scales -- strictly finer than
TQ2's single scale per 256-block (and PrismML's Bonsai uses group 128),
so it is the natural fix candidate for the round-1 TQ2 failure.

Usage:
  python bonsai/quantize_k.py --input models/flux3-klein-bf16-f16.gguf \\
      --output models/flux3-klein-q2k.gguf --stats-out results/quantize-q2k-stats.json
"""
import argparse
import json
import os
import shutil
import struct
import tempfile
import time

import numpy as np

TARGET_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".attn_out.weight",
    ".mlp_in.weight",
    ".mlp_out.weight",
)

GGML_TYPE_Q2_K = 10
Q2K_BLOCK = 256
Q2K_TYPE_SIZE = 84


def round_half_away(x):
    """lroundf: round half away from zero (np.rint rounds half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def fp16_to_f32(b):
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def f32_to_fp16_bytes(a):
    return np.asarray(a, dtype=np.float16).tobytes()


# ---------------------------------------------------------------- Q2_K core

def _qkx2_subblocks_inner(X):
    """One batch of make_qkx2_quants(n=16, nmax=3, weights=|x|, rmin=-0.5,
    rdelta=0.1, nstep=15, use_mad=True) over rows of X (N,16) float32.
    Returns (scale (N,), the_min (N,) >= 0, L (N,16) uint8 codes 0..3).
    L is NOT tracked through the search: it is only a fallback for
    all-zero blocks, so it is recomputed once at the end from the
    winning (scale, min) pair."""
    W = np.abs(X)
    xmin = X.min(axis=1)
    xmax = X.max(axis=1)
    sum_w = W.sum(axis=1)
    WX = W * X
    sum_x = WX.sum(axis=1)
    min0 = np.minimum(xmin, 0.0)
    Xc = X - min0[:, None]  # X - min0, reused in every pass below
    span = xmax - min0
    const = span == 0
    safe_span = np.where(const, 1.0, span)
    # initial candidate: iscale = 3/span
    iscale = 3.0 / safe_span
    L = np.clip(round_half_away(iscale[:, None] * Xc), 0, 3)
    scale = 1.0 / iscale
    best_err = (W * np.abs(scale[:, None] * L - Xc)).sum(axis=1)
    scale_cur = scale.copy()
    min_cur = min0.copy()
    for istep in range(16):  # is = 0..15
        iscale = (2.5 + 0.1 * istep) / safe_span
        Laux = np.clip(round_half_away(iscale[:, None] * Xc), 0, 3)
        T = W * Laux
        sum_l = T.sum(axis=1)
        sum_l2 = (T * Laux).sum(axis=1)
        sum_xl = (WX * Laux).sum(axis=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        valid = D > 0
        safe_D = np.where(valid, D, 1.0)
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / safe_D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / safe_D
        pos = valid & (this_min > 0)
        safe_l2 = np.where(sum_l2 > 0, sum_l2, 1.0)
        this_scale = np.where(pos, sum_xl / safe_l2, this_scale)
        this_min = np.where(pos, 0.0, this_min)
        cur_err = (W * np.abs(this_scale[:, None] * Laux
                              + (this_min - min0)[:, None] - Xc)).sum(axis=1)
        better = valid & (cur_err < best_err)
        scale_cur = np.where(better, this_scale, scale_cur)
        min_cur = np.where(better, this_min, min_cur)
        best_err = np.where(better, cur_err, best_err)
    the_min = -min_cur
    scale_cur = np.where(const, 0.0, scale_cur)
    the_min = np.where(const, -min0, the_min)
    # final codes from the winning (scale, min); 0 for constant blocks
    safe_sc = np.where(scale_cur > 0, scale_cur, 1.0)
    L = np.clip(round_half_away((X - min_cur[:, None]) / safe_sc[:, None]), 0, 3)
    L = np.where(const[:, None], 0, L)
    return scale_cur.astype(np.float32), the_min.astype(np.float32), L.astype(np.uint8)


def _qkx2_subblocks(X, batch=65536):
    """Batched wrapper: keeps working arrays cache-friendly (a 65536-row
    batch is ~4MB per array). Same result as the unbatched version."""
    outs_s, outs_m, outs_l = [], [], []
    for r0 in range(0, X.shape[0], batch):
        s, m, l = _qkx2_subblocks_inner(X[r0:r0 + batch])
        outs_s.append(s)
        outs_m.append(m)
        outs_l.append(l)
    return (np.concatenate(outs_s), np.concatenate(outs_m),
            np.concatenate(outs_l))


def quantize_q2k_rows(w):
    """w: (R, C) float32, C % 256 == 0. Returns Q2_K raw bytes.
    Faithful port of quantize_row_q2_K_ref (per 256-block)."""
    rows, cols = w.shape
    assert cols % 256 == 0
    nb = cols // 256
    X = w.reshape(rows, nb, 16, 16)
    Xr = X.reshape(-1, 16)
    scale, the_min, L = _qkx2_subblocks(Xr)
    scale = scale.reshape(rows, nb, 16)
    the_min = the_min.reshape(rows, nb, 16)
    L = L.reshape(rows, nb, 16, 16)
    max_scale = scale.max(axis=2)
    max_min = the_min.max(axis=2)
    d = np.asarray(max_scale / 15.0, dtype=np.float16)
    dmin = np.asarray(max_min / 15.0, dtype=np.float16)
    safe_ms = np.where(max_scale > 0, max_scale, 1.0)
    safe_mm = np.where(max_min > 0, max_min, 1.0)
    scale_nib = np.clip(round_half_away(15.0 * scale / safe_ms[..., None]), 0, 15)
    min_nib = np.clip(round_half_away(15.0 * the_min / safe_mm[..., None]), 0, 15)
    scale_nib = np.where(max_scale[..., None] > 0, scale_nib, 0).astype(np.uint8)
    min_nib = np.where(max_min[..., None] > 0, min_nib, 0).astype(np.uint8)
    scales_byte = (scale_nib | (min_nib << 4)).astype(np.uint8)
    # re-quantize with the QUANTIZED (nibble-rounded) d/dm, as in C
    d_f32 = d.astype(np.float32)
    dmin_f32 = dmin.astype(np.float32)
    d_sub = d_f32[..., None] * scale_nib
    dm = dmin_f32[..., None] * min_nib
    L2 = np.clip(round_half_away((X + dm[..., None]) /
                                 np.where(d_sub[..., None] > 0, d_sub[..., None], 1.0)), 0, 3)
    L2 = np.where(d_sub[..., None] > 0, L2, L).astype(np.uint8)
    # pack qs: byte b = j*32 + l holds codes at positions j+l+32*k (k=0..3)
    # as 2-bit fields k*2. codes.reshape(rows, nb, 2, 4, 32) indexes [j, k, l].
    codes = L2.reshape(rows, nb, 2, 4, 32)
    qs = (codes[..., 0, :] | (codes[..., 1, :] << 2) |
          (codes[..., 2, :] << 4) | (codes[..., 3, :] << 6)).astype(np.uint8)
    qs = qs.reshape(rows, nb, 64)
    out = np.empty((rows, nb, 84), dtype=np.uint8)
    out[..., 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    out[..., 2:4] = dmin.view(np.uint8).reshape(rows, nb, 2)
    out[..., 4:20] = scales_byte
    out[..., 20:84] = qs
    return out.tobytes()


def dequant_q2k(raw, nelem):
    """Dequantize Q2_K raw bytes -> float32. Vectorized port of
    dequantize_row_q2_K. For sub-block s, element e: code lives in byte
    32*(s//8) + 16*(s%2) + e at shift 2*((s//2)%4)."""
    nblocks = nelem // 256
    assert nelem % 256 == 0
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks, 84)
    d = arr[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = arr[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    scales = arr[:, 4:20]
    qs = arr[:, 20:84].reshape(nblocks, 2, 2, 16)  # [half, pair, byte]
    s = np.arange(16)
    j_idx, pair_idx, k = s // 8, s % 2, (s // 2) % 4
    codes = (qs[:, j_idx, pair_idx, :] >> (2 * k)[None, :, None]) & 3
    dl = (d[:, None] * (scales & 0xF)).astype(np.float32)
    ml = (dmin[:, None] * (scales >> 4)).astype(np.float32)
    return (dl[:, :, None] * codes - ml[:, :, None]).reshape(-1).astype(np.float32)

# ---------------------------------------------------------------- GGUF I/O

def _read_string(buf, off):
    (ln,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + ln].decode(), off + ln


def _read_meta_value(buf, off, vtype):
    if vtype == 0:  # UINT8
        return buf[off], off + 1
    if vtype == 1:  # INT8
        return struct.unpack_from("<b", buf, off)[0], off + 1
    if vtype == 2:  # UINT16
        return struct.unpack_from("<H", buf, off)[0], off + 2
    if vtype == 3:  # INT16
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if vtype == 4:  # UINT32
        return struct.unpack_from("<I", buf, off)[0], off + 4
    if vtype == 5:  # INT32
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if vtype == 6:  # FLOAT32
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if vtype == 7:  # BOOL
        return buf[off] != 0, off + 1
    if vtype == 8:  # STRING
        return _read_string(buf, off)
    if vtype == 9:  # ARRAY
        (etype,) = struct.unpack_from("<I", buf, off)
        off += 4
        (ln,) = struct.unpack_from("<Q", buf, off)
        off += 8
        vals = []
        for _ in range(ln):
            v, off = _read_meta_value(buf, off, etype)
            vals.append(v)
        return vals, off
    if vtype == 10:  # UINT64
        return struct.unpack_from("<Q", buf, off)[0], off + 8
    if vtype == 11:  # INT64
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if vtype == 12:  # FLOAT64
        return struct.unpack_from("<d", buf, off)[0], off + 8
    raise ValueError("unknown metadata value type %d" % vtype)


def parse_gguf(path):
    """Returns dict: version, kv_raw (bytes), kv_count, infos (ordered list
    of {name, dims, raw_type, offset}), data_start, alignment."""
    with open(path, "rb") as f:
        buf = f.read(4096 * 1024)  # header + infos only; data is streamed
    if len(buf) < 24 or buf[0:4] != b"GGUF":
        raise ValueError("not a GGUF file: %s" % path)
    version, = struct.unpack_from("<I", buf, 4)
    tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
    off = 24
    meta = {}
    for _ in range(kv_count):
        key, off = _read_string(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        val, off = _read_meta_value(buf, off, vtype)
        meta[key] = val
    kv_end = off
    infos = []
    for _ in range(tensor_count):
        name, off = _read_string(buf, off)
        (ndims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from("<%dQ" % ndims, buf, off)
        off += 8 * ndims
        (raw_type,) = struct.unpack_from("<I", buf, off)
        off += 4
        (toffset,) = struct.unpack_from("<Q", buf, off)
        off += 8
        infos.append({"name": name, "dims": dims, "raw_type": raw_type,
                      "offset": toffset})
    alignment = int(meta.get("general.alignment", 32))
    data_start = (off + alignment - 1) // alignment * alignment
    return {"version": version, "kv_count": kv_count,
            "kv_raw": buf[24:kv_end], "infos": infos,
            "data_start": data_start, "alignment": alignment,
            "tensor_count": tensor_count}


def tensor_nelem(info):
    n = 1
    for d in info["dims"]:
        n *= d
    return n


def read_tensor_raw(path, parsed, info):
    with open(path, "rb") as f:
        f.seek(parsed["data_start"] + info["offset"])
        nbytes = tensor_nbytes(info)
        return f.read(nbytes)


def tensor_nbytes(info):
    # byte size for the types we handle (F32/F16/BF16/Q2_K/Q4_K/TQ2_0)
    table = {0: (1, 4), 1: (1, 2), 30: (1, 2),
             10: (256, 84), 12: (256, 144), 26: (256, 66)}
    blk, sz = table[info["raw_type"]]
    nelem = tensor_nelem(info)
    assert nelem % blk == 0, (info["name"], info["raw_type"], nelem)
    return (nelem // blk) * sz


def raw_to_f32(raw, raw_type):
    if raw_type == 0:
        return np.frombuffer(raw, dtype=np.float32).copy()
    if raw_type == 1:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if raw_type == 30:  # BF16
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    raise ValueError("cannot convert raw type %d to float32" % raw_type)


def is_target(name, dims):
    return len(dims) == 2 and any(name.endswith(s) for s in TARGET_SUFFIXES)


def align_up(x, a):
    return (x + a - 1) // a * a


def write_gguf(out_path, parsed, out_infos, data_iter):
    """Two-pass streaming writer. out_infos: list of {name, dims, raw_type,
    nbytes} in order; data_iter yields each tensor's raw bytes in order."""
    alignment = parsed["alignment"]
    version = parsed["version"]
    kv_raw = parsed["kv_raw"]
    # pass 1: offsets
    header_len = 24 + len(kv_raw)
    infos_len = 0
    for info in out_infos:
        nb = len(info["name"].encode())
        infos_len += 8 + nb + 4 + 8 * len(info["dims"]) + 4 + 8
    data_start = align_up(header_len + infos_len, alignment)
    offsets = []
    pos = 0
    for info in out_infos:
        offsets.append(pos)
        pos += align_up(info["nbytes"], alignment)
    with open(out_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<QQ", len(out_infos), parsed["kv_count"]))
        f.write(kv_raw)
        for info, off in zip(out_infos, offsets):
            nb = info["name"].encode()
            f.write(struct.pack("<Q", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(info["dims"])))
            for d in info["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", info["raw_type"]))
            f.write(struct.pack("<Q", off))
        pad = data_start - f.tell()
        assert pad >= 0
        f.write(b"\x00" * pad)
        # pass 2: data
        for info, off in zip(out_infos, offsets):
            assert f.tell() == data_start + off, (f.tell(), data_start + off)
            raw = next(data_iter)
            assert len(raw) == info["nbytes"], (info["name"], len(raw), info["nbytes"])
            f.write(raw)
            pad = align_up(len(raw), alignment) - len(raw)
            if pad:
                f.write(b"\x00" * pad)
    return data_start


def _quantize_chunk_task(task):
    """Worker: quantize rows [r0:r1) of one tensor. task =
    (input_path, data_start, tensor_offset, dims, raw_type, r0, r1).
    Returns (tensor_name, r0, q2k_bytes). Top-level so Pool can pickle it."""
    (input_path, data_start, tensor_offset, dims, raw_type,
     name, r0, r1) = task
    cols = dims[0]
    elem_bytes = {0: 4, 1: 2, 30: 2}[raw_type]
    with open(input_path, "rb") as f:
        f.seek(data_start + tensor_offset + r0 * cols * elem_bytes)
        raw = f.read((r1 - r0) * cols * elem_bytes)
    w = raw_to_f32(raw, raw_type).reshape(r1 - r0, cols)
    return name, r0, quantize_q2k_rows(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="source GGUF (F16)")
    ap.add_argument("--output", required=True, help="output GGUF (Q2_K)")
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--row-chunk", type=int, default=512)
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel workers (0 = auto, min(4, cpu_count))")
    ap.add_argument("--tmpdir", default=None,
                    help="temp dir for quantized tensors (default: alongside output)")
    args = ap.parse_args()
    t_start = time.time()

    def log(msg):
        print("[quantize-k %.0fs] %s" % (time.time() - t_start, msg), flush=True)

    parsed = parse_gguf(args.input)
    log("input %s: %d tensors, alignment %d, data_start %d"
        % (args.input, parsed["tensor_count"], parsed["alignment"],
           parsed["data_start"]))
    out_infos = []
    stats = {"tensors": {}, "totals": {}}
    n_q, params_q, params_total = 0, 0, 0
    for info in parsed["infos"]:
        nelem = tensor_nelem(info)
        params_total += nelem
        if is_target(info["name"], info["dims"]):
            out_infos.append({"name": info["name"], "dims": info["dims"],
                              "raw_type": GGML_TYPE_Q2_K,
                              "nbytes": (nelem // Q2K_BLOCK) * Q2K_TYPE_SIZE})
            n_q += 1
            params_q += nelem
        else:
            out_infos.append({"name": info["name"], "dims": info["dims"],
                              "raw_type": info["raw_type"],
                              "nbytes": tensor_nbytes(info)})

    # ---- phase A: quantize target tensors (parallel over row-chunks) ----
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="q2k-",
                                            dir=os.path.dirname(
                                                os.path.abspath(args.output)))
    qfiles = {}
    tasks = []
    for info in parsed["infos"]:
        if not is_target(info["name"], info["dims"]):
            continue
        rows = 1
        for d in info["dims"][1:]:
            rows *= d
        safe_name = info["name"].replace("/", "_")
        qfiles[info["name"]] = os.path.join(tmpdir, safe_name + ".q2k")
        open(qfiles[info["name"]], "wb").close()
        for r0 in range(0, rows, args.row_chunk):
            tasks.append((args.input, parsed["data_start"], info["offset"],
                          info["dims"], info["raw_type"], info["name"],
                          r0, min(r0 + args.row_chunk, rows)))
    n_workers = args.workers or min(4, os.cpu_count() or 1)
    log("quantizing %d tensors (%d row-chunks, %d workers, tmp %s)"
        % (n_q, len(tasks), n_workers, tmpdir))
    done = 0
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for name, r0, blob in pool.imap(_quantize_chunk_task, tasks, chunksize=1):
                with open(qfiles[name], "ab") as f:
                    f.write(blob)
                done += 1
                if done % 500 == 0:
                    log("quantized %d/%d chunks (%.0fs)"
                        % (done, len(tasks), time.time() - t_start))
    except Exception as e:  # fall back to sequential
        log("parallel quantize failed (%r) -- falling back to sequential" % e)
        for info in parsed["infos"]:
            if not is_target(info["name"], info["dims"]):
                continue
            with open(qfiles[info["name"]], "wb") as f:
                raw = read_tensor_raw(args.input, parsed, info)
                w = raw_to_f32(raw, info["raw_type"])
                rows = 1
                for d in info["dims"][1:]:
                    rows *= d
                w = w.reshape(rows, info["dims"][0])
                for r0 in range(0, rows, args.row_chunk):
                    f.write(quantize_q2k_rows(w[r0:r0 + args.row_chunk]))
        done = len(tasks)
    log("quantize phase done: %d chunks in %.0fs"
        % (done, time.time() - t_start))

    # ---- phase B: stream the output GGUF ----
    def data_iter():
        for info in parsed["infos"]:
            if is_target(info["name"], info["dims"]):
                with open(qfiles[info["name"]], "rb") as f:
                    yield f.read()
            else:
                yield read_tensor_raw(args.input, parsed, info)

    write_gguf(args.output, parsed, out_infos, data_iter())
    log("wrote %s (%d tensors, %d quantized, %.3fB/%.3fB params, %.0fs)"
        % (args.output, len(out_infos), n_q, params_q / 1e9,
           params_total / 1e9, time.time() - t_start))

    # ---- validation: re-read output, dequantize sampled targets ----
    out_parsed = parse_gguf(args.output)
    assert out_parsed["tensor_count"] == parsed["tensor_count"]
    targets = [i for i in parsed["infos"]
               if is_target(i["name"], i["dims"])]
    step = max(1, len(targets) // args.sample)
    rels = []
    for info in targets[::step][:args.sample]:
        src_raw = read_tensor_raw(args.input, parsed, info)
        src = raw_to_f32(src_raw, info["raw_type"])
        out_info = next(o for o in out_parsed["infos"] if o["name"] == info["name"])
        assert out_info["raw_type"] == GGML_TYPE_Q2_K, out_info
        out_raw = read_tensor_raw(args.output, out_parsed, out_info)
        dq = dequant_q2k(out_raw, src.size)
        rel = float(np.linalg.norm(dq - src) / max(np.linalg.norm(src), 1e-12))
        rels.append(rel)
        stats["tensors"][info["name"]] = {
            "params": int(src.size), "rel_l2": rel,
            "src_bytes": len(src_raw), "q2k_bytes": len(out_raw)}
        log("validate %s rel_l2=%.4f" % (info["name"], rel))
    stats["totals"] = {
        "tensors_quantized": n_q,
        "params_quantized": params_q,
        "params_total": params_total,
        "params_quantized_frac": params_q / max(params_total, 1),
        "validate_rel_l2_median": float(np.median(rels)) if rels else None,
        "validate_rel_l2_max": float(max(rels)) if rels else None,
        "input_bytes": os.path.getsize(args.input),
        "output_bytes": os.path.getsize(args.output),
        "seconds": round(time.time() - t_start, 1),
    }
    log("DONE: validate rel L2 median=%.4f max=%.4f, %.0fs"
        % (stats["totals"]["validate_rel_l2_median"],
           stats["totals"]["validate_rel_l2_max"], time.time() - t_start))
    if args.stats_out:
        os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
        with open(args.stats_out, "w") as fp:
            json.dump(stats, fp, indent=2)
        log("stats written to %s" % args.stats_out)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(json.dumps(stats["totals"], indent=2))


if __name__ == "__main__":
    main()
