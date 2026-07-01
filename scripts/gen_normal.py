#!/usr/bin/env python3
"""Generate a clustered (Normal) synthetic point set for ACS-SDH.

Points are drawn from an isotropic Gaussian of standard deviation sigma,
centered in the unit cube, and written as one whitespace-separated coordinate
row per point -- the format expected by acs via RDF_INPUT.

Example:
    python3 scripts/gen_normal.py --n 1000000 --dim 3 --sigma 0.1 \
        --out data/normal_3d_sigma0.1.xyz
    RDF_INPUT=data/normal_3d_sigma0.1.xyz ./acs 1000000 0.1 1 0.1 4 1 --dim 3

The binary rescales any input to the unit cube, so the absolute center/scale
here only affect the relative cluster tightness (controlled by sigma).
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of points")
    ap.add_argument("--dim", type=int, default=2, help="dimensionality (1..4)")
    ap.add_argument("--sigma", type=float, default=0.1,
                    help="cluster width (Gaussian std dev); smaller = tighter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output coordinate file")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pts = rng.normal(loc=0.5, scale=a.sigma, size=(a.n, a.dim))
    # write with enough precision; acs reads the first `dim` columns per line
    np.savetxt(a.out, pts, fmt="%.6f")
    print(f"wrote {a.n} {a.dim}D points (sigma={a.sigma}) -> {a.out}")


if __name__ == "__main__":
    main()
