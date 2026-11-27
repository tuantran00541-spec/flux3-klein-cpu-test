#!/usr/bin/env python3
"""Diagnose a converted TQ2/Q3_K/Q2_K/F16 GGUF against its BF16 source.

Answers, for one converted GGUF file:
  1. INVENTORY: per-tensor GGML type (from gguf-dump text) -- did the
     FP16 exceptions (modulations, embedders, time embeds, norm_out,
     proj_out, biases, norms) really stay FP16, or did the converter's
     --tensor-type-rules misapply and quantize something sensitive?
  2. FIDELITY (TQ2 only): dequantize sampled target tensors and compare
     against the reference ternary values (computed from the BF16 source
     with the amax rule, or read from a pre-ternarized safetensors file).
     If dequant != reference, the converter mangled the values.
  3. EXCEPTIONS: F16/BF16 tensors must be value-identical to the BF16
     source (BF16 -> F16 is exact for typical weight magnitudes).

Usage:
  python bonsai/diagnose_gguf.py --gguf models/flux3-klein-tq2.gguf \
      --dump results/dump-tq2.txt \
      --bf16 models/diffusion_models/flux3-image-klein-bf16.safetensors \
      [--ternary models/flux3-image-klein-ternary.safetensors] \
      --out results/diagnose-tq2.json

The dump file is the output of `gguf-dump <gguf>` (authoritative type
names -- this script does NOT guess GGML enum numbers). Raw tensor bytes
are read with a minimal GGUF parser (header + tensor infos only).
"""
import argparse
import json
import re
import struct
import sys

import numpy as np

# block size / bytes-per-block, keyed by gguf-dump type NAME.
TYPE_TABLE = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q8_0": (32, 34),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "TQ1_0": (256, 54),
    "TQ2_0": (256, 66),
}

SENSITIVE_PATTERNS = (
    "modulat", "emb_in", "time_text_embed", "time_guidance_embed",
    "norm_out", "proj_out", "x_embedder", "context_embedder",
    "final_layer", "bias", "norm",
)

TARGET_SUFFIXES = (
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
    ".attn_out.weight", ".mlp_in.weight", ".mlp_out.weight",
)


def parse_gguf_infos(path):
    """Minimal GGUF parse: returns (data_start, {name: (raw_type, dims, offset)}, meta)."""
    with open(path, "rb") as f:
        buf = f.read()
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", buf, 0)
    assert magic == 0x46554747, "not a GGUF file"
    off = 24

    def read_string(off):
        (ln,) = struct.unpack_from("<Q", buf, off)
        off += 8
        s = buf[off:off + ln].decode("utf-8", "replace")
        return s, off + ln

    def read_value(off, vtype):
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
            return read_string(off)
        if vtype == 9:  # ARRAY
            (etype, ln) = struct.unpack_from("<IQ", buf, off)
            off += 12
            vals = []
            for _ in range(ln):
                v, off = read_value(off, etype)
                vals.append(v)
            return vals, off
        if vtype == 10:  # UINT64
            return struct.unpack_from("<Q", buf, off)[0], off + 8
        if vtype == 11:  # INT64
            return struct.unpack_from("<q", buf, off)[0], off + 8
        if vtype == 12:  # FLOAT64
            return struct.unpack_from("<d", buf, off)[0], off + 8
        raise ValueError("unknown metadata value type %d" % vtype)

    meta = {}
    for _ in range(n_kv):
        key, off = read_string(off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        val, off = read_value(off, vtype)
        meta[key] = val
    alignment = int(meta.get("general.alignment", 32))
    infos = {}
    for _ in range(n_tensors):
        name, off = read_string(off)
        (ndims,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from("<%dQ" % ndims, buf, off)
        off += 8 * ndims
        (raw_type,) = struct.unpack_from("<I", buf, off)
        off += 4
        (toffset,) = struct.unpack_from("<Q", buf, off)
        off += 8
        infos[name] = (raw_type, dims, toffset)
    data_start = (off + alignment - 1) // alignment * alignment
    return data_start, infos, meta


def types_from_gguf_py(infos):
    """Map raw GGML type ints -> names via gguf-py's enum (authoritative,
    no guessing). Returns {} if gguf is not importable."""
    try:
        from gguf import GGMLQuantizationType
    except Exception:
        return {}
    out = {}
    for name, (raw_type, dims, off) in infos.items():
        try:
            out[name] = GGMLQuantizationType(raw_type).name
        except Exception:
            pass
    return out


def parse_dump_types(dump_path):
    """Parse name -> type name from `gguf-dump` text output."""
    types = {}
    text = open(dump_path, errors="replace").read()
    for line in text.splitlines():
        m = re.search(r"([A-Za-z0-9_.]+)\s+.*?\b(F32|F16|BF16|Q8_0|Q2_K|Q3_K|Q4_K|Q5_K|Q6_K|Q8_K|TQ1_0|TQ2_0)\b", line)
        if m:
            types[m.group(1)] = m.group(2)
    return types


def ref_name(name):
    """GGUF tensors carry the 'model.diffusion_model.' prefix; the
    safetensors source does not. Strip it for reference lookups."""
    return name.removeprefix("model.diffusion_model.")


def read_tensor_bytes(path, data_start, info, nbytes):
    with open(path, "rb") as f:
        f.seek(data_start + info[2])
        return f.read(nbytes)


def dequant_tq2(raw, nelem):
    """Dequantize one TQ2_0 tensor's raw bytes -> float32 array.
    Block = 66 bytes: fp16 scale + 64 qs bytes. GGML layout
    (dequantize_row_tq2_0 in ggml-quants.c): values are emitted as
    for j in {0,32} byte-chunk, for l in 0..3, for m in 0..31:
    code = (qs[j+m] >> 2*l) & 3. So the value at linear position p
    lives in byte (p%32) + 32*(p//128), bits 2*((p//32)%4).
    (An earlier version of this function assumed sequential packing
    byte p//4 -- that is WRONG and produced rel L2 in the thousands.)"""
    nblocks = nelem // 256
    assert nelem % 256 == 0, "TQ2 tensor with nelem % 256 != 0"
    p = np.arange(256)
    byte_idx = (p % 32) + 32 * (p // 128)
    shift = 2 * ((p // 32) % 4)
    out = np.empty(nelem, dtype=np.float32)
    for b in range(nblocks):
        blk = raw[b * 66:(b + 1) * 66]
        scale = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(blk[2:66], dtype=np.uint8)
        codes = (qs[byte_idx] >> shift) & 3
        out[b * 256:(b + 1) * 256] = scale * (codes.astype(np.float32) - 1.0)
    return out


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, header_len


def read_st_tensor(path, header, header_len, name):
    meta = header[name]
    shape = meta["shape"]
    dtype = meta["dtype"]
    off0, off1 = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + off0)
        raw = f.read(off1 - off0)
    if dtype == "BF16":
        arr = bf16_to_f32(raw)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError("unsupported safetensors dtype %s" % dtype)
    return arr.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dump", default=None,
                    help="gguf-dump text output (fallback if gguf-py is missing)")
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--ternary", default=None,
                    help="pre-ternarized safetensors (MSE recipe); if given, "
                         "TQ2 dequant is compared against it exactly")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()

    report = {"gguf": args.gguf}
    data_start, infos, meta = parse_gguf_infos(args.gguf)
    # Authoritative type names: gguf-py's enum first, dump text as fallback.
    dump_types = types_from_gguf_py(infos)
    report["type_source"] = "gguf-py-enum" if dump_types else "gguf-dump-text"
    if not dump_types:
        if not args.dump:
            sys.exit("no type names: gguf-py missing and no --dump given")
        dump_types = parse_dump_types(args.dump)
    report["tensors_in_gguf"] = len(infos)
    report["dump_types_parsed"] = len(dump_types)

    # ---- 1. INVENTORY: type counts + sensitive-pattern check ----
    from collections import Counter
    type_counts = Counter(dump_types.values())
    report["type_counts"] = dict(type_counts)
    missing = [n for n in infos if n not in dump_types]
    report["tensors_missing_from_dump_parse"] = missing[:20]
    sensitive = {}
    for name, tname in dump_types.items():
        if any(p in name for p in SENSITIVE_PATTERNS):
            sensitive[name] = tname
    report["sensitive_tensors"] = sensitive
    bad_sensitive = {n: t for n, t in sensitive.items()
                     if t not in ("F16", "BF16", "F32")}
    report["sensitive_tensors_QUANTIZED"] = bad_sensitive
    print("tensors in GGUF: %d, types parsed from dump: %d"
          % (len(infos), len(dump_types)))
    print("type counts: %s" % dict(type_counts))
    print("sensitive tensors: %d, QUANTIZED (smoking gun if any): %s"
          % (len(sensitive), bad_sensitive))

    # ---- 2/3. FIDELITY + EXCEPTIONS ----
    bf16_header, bf16_hlen = load_safetensors_header(args.bf16)
    tern_header = tern_hlen = None
    if args.ternary:
        tern_header, tern_hlen = load_safetensors_header(args.ternary)

    def tensor_nbytes(name, tname):
        dims = infos[name][1]
        nelem = 1
        for d in dims:
            nelem *= d
        blk, sz = TYPE_TABLE[tname]
        assert nelem % blk == 0, (name, tname, nelem)
        return nelem, (nelem // blk) * sz

    # Exception values: F16/BF16 tensors must equal the BF16 source.
    exc_check = {}
    for name, tname in sensitive.items():
        rn = ref_name(name)
        if name not in infos or rn not in bf16_header:
            continue
        if tname in ("F16", "BF16"):
            nelem, nbytes = tensor_nbytes(name, tname)
            raw = read_tensor_bytes(args.gguf, data_start, infos[name], nbytes)
            got = np.frombuffer(raw, dtype=np.float16).astype(np.float32) \
                if tname == "F16" else bf16_to_f32(raw)
            ref = read_st_tensor(args.bf16, bf16_header, bf16_hlen, rn)
            exc_check[name] = {
                "type": tname,
                "max_abs_diff": float(np.max(np.abs(got.reshape(ref.shape) - ref))),
                "identical": bool(np.array_equal(got.reshape(ref.shape), ref)),
            }
    report["exception_value_check"] = exc_check
    bad_exc = {n: v for n, v in exc_check.items() if not v["identical"]}
    print("exception tensors value-checked: %d, NOT identical: %s"
          % (len(exc_check), list(bad_exc)))

    # TQ2 fidelity on sampled target tensors.
    tq2_names = [n for n, t in dump_types.items() if t == "TQ2_0"]
    report["tq2_tensor_count"] = len(tq2_names)
    fidelity = {}
    if tq2_names:
        targets = [n for n in tq2_names
                   if any(n.endswith(s) for s in TARGET_SUFFIXES)]
        if not targets:
            targets = tq2_names
        step = max(1, len(targets) // args.sample)
        for name in targets[::step][:args.sample]:
            tname = dump_types[name]
            nelem, nbytes = tensor_nbytes(name, tname)
            raw = read_tensor_bytes(args.gguf, data_start, infos[name], nbytes)
            rn = ref_name(name)
            if tern_header is not None and rn in tern_header:
                ref = read_st_tensor(args.ternary, tern_header, tern_hlen, rn)
                ref_kind = "ternary-file"
            else:
                w = read_st_tensor(args.bf16, bf16_header, bf16_hlen, rn)
                wb = w.reshape(-1, 256)
                amax = np.abs(wb).max(axis=-1, keepdims=True)
                amax = np.where(amax > 0, amax, 1.0)
                t = np.clip(np.round(wb / amax), -1, 1)
                ref = (amax * t).reshape(w.shape)
                ref_kind = "amax-from-bf16"
            ref = ref.reshape(-1)
            dq = dequant_tq2(raw, nelem)
            rel = float(np.linalg.norm(dq - ref) /
                        max(np.linalg.norm(ref), 1e-12))
            best = {"variant": "ggml-interleaved", "rel_l2": rel,
                    "max_abs_diff": float(np.max(np.abs(dq - ref))),
                    "zero_frac_deq": float(np.mean(dq == 0)),
                    "zero_frac_ref": float(np.mean(ref == 0))}
            fidelity[name] = dict(best, ref_kind=ref_kind,
                                  shape=list(infos[name][1]))
        report["tq2_fidelity_sample"] = fidelity
        rels = [v["rel_l2"] for v in fidelity.values()]
        print("TQ2 fidelity (n=%d): rel L2 min/med/max = %.4g / %.4g / %.4g"
              % (len(rels), min(rels), float(np.median(rels)), max(rels)))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("report written to %s" % args.out)


if __name__ == "__main__":
    main()
