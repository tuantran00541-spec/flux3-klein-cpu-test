#!/usr/bin/env python3
"""Verify a type-35 (TQ2G128) GGUF against a reference F16 GGUF.

Used to check an sd-cli `--type tq2_0` conversion (diagnostic): does the
converter's output dequantize (with the CORRECT 68-byte layout) back to
the reference values, and is its density 0.265625 B/elem (68-byte blocks)
rather than 0.2578125 (66-byte TQ2_0)?

  python bonsai3/verify_gguf.py --gguf converted-tq2.gguf \
      --ref ternary-f16.gguf --targets bonsai3/tokforge-layout.json \
      --out results/verify-tq2.json
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pack_gguf import parse_gguf_header_bytes, dequant_68, TYPE_INFO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    layout = json.load(open(args.targets))
    targets = set(layout["type35"])

    def load_header(path):
        with open(path, "rb") as f:
            head = f.read(1 << 20)
            hdr = parse_gguf_header_bytes(head)
            return hdr

    hg = load_header(args.gguf)
    hr = load_header(args.ref)
    tg = {t["name"]: t for t in hg["tensors"]}
    tr = {t["name"]: t for t in hr["tensors"]}

    rows = []
    with open(args.gguf, "rb") as fg, open(args.ref, "rb") as fr:
        for name in sorted(targets):
            g, r = tg[name], tr[name]
            fg.seek(hg["data_start"] + g["offset"])
            raw = fg.read(int(g["nelem"] * TYPE_INFO[g["type"]][1]))
            fr.seek(hr["data_start"] + r["offset"])
            ref_raw = fr.read(int(r["nelem"] * TYPE_INFO[r["type"]][1]))
            ref = np.frombuffer(ref_raw, dtype="<f2").astype(np.float32)
            if g["type"] == 35:
                v = dequant_68(raw, g["nelem"])
                bpe = len(raw) / g["nelem"]
            else:
                v = np.frombuffer(raw, dtype="<f2").astype(np.float32) \
                    if g["type"] == 1 else None
                bpe = len(raw) / g["nelem"]
            num = float(np.linalg.norm(v - ref))
            den = float(np.linalg.norm(ref))
            rel = num / den if den > 0 else 0.0
            rows.append({"name": name, "type": g["type"], "bpe": bpe,
                         "rel_l2": rel})
    rels = [r["rel_l2"] for r in rows]
    bpes = sorted({r["bpe"] for r in rows})
    types = sorted({r["type"] for r in rows})
    summary = {"tensors": len(rows), "types": types, "bpe_values": bpes,
               "rel_l2_median": float(np.median(rels)),
               "rel_l2_max": float(np.max(rels)),
               "rel_l2_mean": float(np.mean(rels)),
               "worst": sorted(rows, key=lambda r: -r["rel_l2"])[:5]}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as fp:
            json.dump({"summary": summary, "tensors": rows}, fp, indent=2)


if __name__ == "__main__":
    main()
