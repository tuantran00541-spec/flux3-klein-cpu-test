#!/usr/bin/env python3
"""Pack a ternarized F16 GGUF into PrismML/TokForge TQ2G128 (sd.cpp GGML type 35).

The vendor format, proven byte-exact against the TokForge GGUF
(bonsai_klein_official_tq2g128.gguf) + Prism's unpacked model
(bonsai-image-ternary-4B-unpacked) -- rel-L2 = 0.0 on 216 x 128-groups
of qkv and on the proj tensor, scales matching the unpacked amax exactly:

  - 68-byte block = 2 groups of 128 elements.
  - bytes [0:32]  = group-0 2-bit codes, [32:64] = group-1 codes,
    [64:66] = d0 (FP16 LE), [66:68] = d1 (FP16 LE).
  - codes: GGML TQ2_0-style INTERLEAVED packing: byte j (0..31) holds
    weights j, j+32, j+64, j+96 at bit shifts 0, 2, 4, 6;
    value = d * (code - 1), code in {0,1,2}.
  - scale = per-128-group amax (FP16) = max |w| of the group.

This writer does NOT quantize: it packs values that are already ternary
(exact multiples of one per-128-group scale). The packer verifies that
dequant(pack(w)) == w exactly (rel-L2 ~ 0) for every packed tensor and
fails loudly if any code is 3 (input was not ternary) or the round-trip
is inexact. The converter is bypassed entirely -- the bytes are ours.

Input: an F16 GGUF (all tensors type F16=1), normally produced by
  sd-cli -M convert --diffusion-model <ternary.safetensors> --type f16
Target tensors: the 80 type-35 tensors of the TokForge file, listed in
  bonsai3/tokforge-layout.json (same tensor names/shapes as the sd-cli
  F16 conversion of the base safetensors). All other tensors are copied
  byte-identical (F16 exceptions).

GGUF layout written: same metadata bytes as the input, same tensor-info
order, types changed to 35 for targets, offsets recomputed as cumulative
sizes (all tensor sizes are multiples of 32, so no alignment padding --
matches the TokForge file, whose 148 inter-tensor gaps are all 0).
Data section starts at align32(header_end), same as the input.

Usage:
  python bonsai3/pack_gguf.py --input ternary-f16.gguf \
      --output ternary-t35.gguf --targets bonsai3/tokforge-layout.json \
      --stats-out results/pack-stats.json
"""
import argparse
import json
import struct
import sys
import time

import numpy as np

# GGML type enum (sd.cpp's ggml): bytes per element, and block structure.
# Only the types this writer handles are listed.
TYPE_INFO = {
    0: ("F32", 4.0),
    1: ("F16", 2.0),
    30: ("BF16", 2.0),
    35: ("TQ2G128", 34.0 / 128.0),
}
T35 = 35
GROUP = 128
BLOCK_BYTES = 68  # 2 groups x (32 code bytes + 2 scale bytes)


def align32(n):
    return (n + 31) // 32 * 32


def read_str(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8"), off + n


# metadata value sizes (fixed-width types); string/array handled separately
META_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def parse_gguf_header_bytes(data):
    """Parse a GGUF header from a bytes buffer holding at least the header.
    Returns (meta_raw, tensors, data_start, header_end)."""
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    (version,) = struct.unpack_from("<I", data, 4)
    (tensor_count,) = struct.unpack_from("<Q", data, 8)
    (kv_count,) = struct.unpack_from("<Q", data, 16)
    off = 24
    for _ in range(kv_count):
        _, off = read_str(data, off)
        (vt,) = struct.unpack_from("<I", data, off)
        off += 4
        if vt == 8:
            _, off = read_str(data, off)
        elif vt == 9:  # array
            (et, n) = struct.unpack_from("<IQ", data, off)
            off += 12
            for _ in range(n):
                if et == 8:
                    _, off = read_str(data, off)
                else:
                    off += META_FIXED[et]
        else:
            off += META_FIXED[vt]
    meta_raw = data[24:off]
    tensors = []
    for _ in range(tensor_count):
        name, off = read_str(data, off)
        (nd,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = list(struct.unpack_from("<%dQ" % nd, data, off))
        off += 8 * nd
        (typ,) = struct.unpack_from("<I", data, off)
        off += 4
        (tofs,) = struct.unpack_from("<Q", data, off)
        off += 8
        nelem = 1
        for d in dims:
            nelem *= d
        tensors.append({"name": name, "dims": dims, "type": typ,
                        "offset": tofs, "nelem": nelem})
    header_end = off
    data_start = align32(header_end)
    return {"version": version, "kv_count": kv_count, "meta_raw": meta_raw,
            "tensors": tensors, "data_start": data_start,
            "header_end": header_end}


def pack_ternary_68(w32):
    """w32: float32 array, flat, length % 256 == 0, already ternary.
    Returns (packed_bytes, scales_fp16, code3_count, rel_l2_vs_input).
    Vectorized over 256-blocks; each block = 2 x 128-groups."""
    n = w32.size
    assert n % 256 == 0, "nelem %d not divisible by 256" % n
    wb = w32.reshape(-1, 256)
    g0 = wb[:, :128]
    g1 = wb[:, 128:256]
    out = np.empty((wb.shape[0], BLOCK_BYTES), dtype=np.uint8)
    max_rel_l2 = 0.0
    code3 = 0
    for gi, g in enumerate((g0, g1)):
        amax = np.abs(g).max(axis=1)  # (nblocks,)
        safe = np.where(amax > 0, amax, np.ones_like(amax))
        codes = np.rint(g / safe[:, None]).astype(np.int64) + 1  # {-1,0,1}->{0,1,2}
        code3 += int(((codes < 0) | (codes > 2)).sum())
        codes = np.clip(codes, 0, 2).astype(np.uint8)
        # interleaved packing: byte j = c[j] | c[j+32]<<2 | c[j+64]<<4 | c[j+96]<<6
        c = codes.reshape(-1, 4, 32)
        packed = (c[:, 0, :] | (c[:, 1, :] << 2) | (c[:, 2, :] << 4)
                  | (c[:, 3, :] << 6)).astype(np.uint8)
        out[:, gi * 32:(gi + 1) * 32] = packed
        d16 = amax.astype(np.float16)
        out[:, 64 + gi * 2:66 + gi * 2] = d16.view(np.uint8).reshape(-1, 2)
        # verify: dequant must reproduce the input exactly
        deq = d16.astype(np.float32)[:, None] * (codes.astype(np.float32) - 1.0)
        num = float(np.linalg.norm(deq - g))
        den = float(np.linalg.norm(g))
        rel = num / den if den > 0 else (0.0 if num == 0 else float("inf"))
        max_rel_l2 = max(max_rel_l2, rel)
    return out.tobytes(), max_rel_l2, code3


def dequant_68(packed, nelem):
    """Inverse of pack_ternary_68 (for independent verification)."""
    raw = np.frombuffer(packed, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    out = np.empty(nelem, dtype=np.float32)
    wb = out.reshape(-1, 256)
    for gi in range(2):
        qs = raw[:, gi * 32:(gi + 1) * 32]
        d = raw[:, 64 + gi * 2:66 + gi * 2].copy().view(np.float16).astype(np.float32)
        codes = np.empty((qs.shape[0], 128), dtype=np.float32)
        for n in range(4):
            codes[:, n * 32:(n + 1) * 32] = \
                ((qs.astype(np.int32) >> (2 * n)) & 3) - 1
        wb[:, gi * 128:(gi + 1) * 128] = d * codes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--targets", required=True,
                    help="JSON with a 'type35' list of tensor names to pack")
    ap.add_argument("--stats-out", default=None)
    args = ap.parse_args()
    t_start = time.time()

    def log(msg):
        print("[pack %.0fs] %s" % (time.time() - t_start, msg), flush=True)

    with open(args.targets) as fp:
        layout = json.load(fp)
    targets = set(layout["type35"])
    log("target tensors (TokForge type-35 set): %d" % len(targets))

    with open(args.input, "rb") as f:
        # read the header (first 1 MB is plenty: 149 tensors)
        head_bytes = f.read(1 << 20)
        hdr = parse_gguf_header_bytes(head_bytes)
        f.seek(0)
        f.seek(hdr["data_start"])
        data_start = hdr["data_start"]
        tensors = hdr["tensors"]
        log("input %s: %d tensors, data_start=%d, header_end=%d"
            % (args.input, len(tensors), data_start, hdr["header_end"]))

        names = {t["name"] for t in tensors}
        missing = targets - names
        if missing:
            log("FATAL: %d target tensors missing from input: %s"
                % (len(missing), sorted(missing)[:5]))
            sys.exit(2)
        extra = names - set(layout.get("type30", [])) - targets
        if extra:
            log("note: %d tensors not in TokForge layout (copied as-is): %s"
                % (len(extra), sorted(extra)[:5]))

        # compute output offsets (cumulative sizes; all sizes % 32 == 0)
        out_tensors = []
        off = 0
        for t in tensors:
            is_target = t["name"] in targets
            if is_target:
                if t["type"] not in (1, 30):
                    log("FATAL: target %s has unexpected input type %d"
                        % (t["name"], t["type"]))
                    sys.exit(2)
                if t["nelem"] % 256 != 0:
                    log("FATAL: target %s nelem %d not divisible by 256"
                        % (t["name"], t["nelem"]))
                    sys.exit(2)
                out_type = T35
            else:
                out_type = t["type"]
            size = int(t["nelem"] * TYPE_INFO[out_type][1])
            out_tensors.append({"name": t["name"], "dims": t["dims"],
                                "type": out_type, "offset": off,
                                "nelem": t["nelem"], "size": size,
                                "in_type": t["type"],
                                "in_offset": t["offset"],
                                "target": is_target})
            off += size
        total_size = off
        log("output data size: %d bytes (%.3f GB)"
            % (total_size, total_size / 1e9))

        # write the output header (same metadata, same tensor order)
        with open(args.output, "wb") as fo:
            fo.write(b"GGUF")
            fo.write(struct.pack("<I", hdr["version"]))
            fo.write(struct.pack("<Q", len(out_tensors)))
            fo.write(struct.pack("<Q", hdr["kv_count"]))
            fo.write(hdr["meta_raw"])
            for t in out_tensors:
                name_b = t["name"].encode("utf-8")
                fo.write(struct.pack("<Q", len(name_b)))
                fo.write(name_b)
                fo.write(struct.pack("<I", len(t["dims"])))
                fo.write(struct.pack("<%dQ" % len(t["dims"]), *t["dims"]))
                fo.write(struct.pack("<I", t["type"]))
                fo.write(struct.pack("<Q", t["offset"]))
            header_end = fo.tell()
            assert header_end == hdr["header_end"], \
                (header_end, hdr["header_end"])
            pad = data_start - header_end
            assert pad >= 0
            fo.write(b"\x00" * pad)

            # stream tensors: pack targets, copy exceptions
            stats_tensors = {}
            worst_rel = 0.0
            total_code3 = 0
            n_packed = 0
            for t in out_tensors:
                f.seek(data_start + t["in_offset"])
                raw = f.read(int(t["nelem"] * TYPE_INFO[t["in_type"]][1]))
                if not t["target"]:
                    fo.seek(data_start + t["offset"])
                    fo.write(raw)
                    continue
                if t["in_type"] == 1:
                    w = np.frombuffer(raw, dtype="<f2").astype(np.float32)
                elif t["in_type"] == 30:
                    w = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                         << 16).view(np.float32)
                else:
                    raise ValueError("unexpected input type %d" % t["in_type"])
                packed, rel, code3 = pack_ternary_68(w)
                worst_rel = max(worst_rel, rel)
                total_code3 += code3
                # independent verification: dequant the packed bytes
                v = dequant_68(packed, t["nelem"])
                num = float(np.linalg.norm(v - w))
                den = float(np.linalg.norm(w))
                rel2 = num / den if den > 0 else (0.0 if num == 0 else float("inf"))
                worst_rel = max(worst_rel, rel2)
                fo.seek(data_start + t["offset"])
                fo.write(packed)
                n_packed += 1
                stats_tensors[t["name"]] = {
                    "params": t["nelem"], "shape": t["dims"],
                    "rel_l2_pack": rel, "rel_l2_dequant": rel2,
                    "code3": code3, "bytes": len(packed)}
                if n_packed % 20 == 0:
                    log("packed %d/%d target tensors (worst rel-L2 %.3g, code3 %d)"
                        % (n_packed, len(targets), worst_rel, total_code3))
        log("DONE: packed %d tensors, worst rel-L2 %.3g, code3 %d, "
            "output %.3f GB, total %.0fs"
            % (n_packed, worst_rel, total_code3,
               (data_start + total_size) / 1e9, time.time() - t_start))
        if total_code3 > 0 or worst_rel > 1e-6:
            log("FATAL: pack verification failed (code3=%d, worst rel-L2=%.3g)"
                % (total_code3, worst_rel))
            sys.exit(3)

    stats = {"recipe": {"format": "TQ2G128 type 35, 68-byte blocks, "
                                  "interleaved 2-bit codes, FP16 amax scale",
                         "targets": len(targets)},
             "tensors": stats_tensors,
             "totals": {"packed": n_packed, "worst_rel_l2": worst_rel,
                        "code3": total_code3,
                        "output_bytes": data_start + total_size,
                        "seconds": round(time.time() - t_start, 1)}}
    if args.stats_out:
        with open(args.stats_out, "w") as fp:
            json.dump(stats, fp, indent=2)
        log("stats written to %s" % args.stats_out)
    print(json.dumps(stats["totals"], indent=2))


if __name__ == "__main__":
    main()
