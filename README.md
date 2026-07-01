# ACS: Autocorrelation-Based Spatial Distance Histograms

Reference implementation of the Autocorrelation-based algorithms for computing the
**spatial distance histogram (SDH / pair-distance histogram)**: the
**Basic ACS** method and its **Ring Enumeration** optimization,
together with the **Dual-Tree** spatial-index baseline they are compared against.

Instead of traversing a spatial index over point pairs, ACS observes that the number
of point pairs at each grid *displacement* is the autocorrelation of the
cell-occupancy grid — computed for all displacements at once by a single FFT — after
which each displacement is routed to histogram buckets by a geometric distance bound.
This yields an `O(N log N)` approximate histogram, orders of magnitude faster than the
dual-tree baseline. See the paper for details.

## Requirements

- A C compiler with OpenMP (e.g. `gcc`)
- [FFTW3](http://www.fftw.org/), double-precision library with OpenMP support
  (`libfftw3`, `libfftw3_omp`)
- Python 3 + NumPy (only for generating the synthetic *Normal* workloads and plots)

Debian/Ubuntu:
```bash
sudo apt-get install build-essential libfftw3-dev
```

## Build

```bash
git clone https://github.com/YichengTuLab/Autocorrelation-Based-SDH.git
cd Autocorrelation-Based-SDH
make            # produces ./acs
```
or directly:
```bash
gcc -O3 -march=native -fopenmp src/acs.c -o acs -lfftw3 -lfftw3_omp -lm
```

## Usage

```
./acs <num_points> <bin_width> [num_threads] [avg_points_per_leaf] [method] [approx] [options]
```

| Argument | Meaning | Default |
|----------|---------|---------|
| `num_points` | number of points (integer > 0) | — |
| `bin_width` | histogram bucket width `w` (float > 0) | — |
| `num_threads` | OpenMP threads | max |
| `avg_points_per_leaf` | target points per leaf cell (λ) | 10 |
| `method` | `dualtree`, `autocorr`, or `ring` | `autocorr` |
| `approx` | `0`=exact, `1`=approximate | 0 |

Options: `--dim D` (dimensionality 1–4), `--bin-scheme log:N:DMIN:DMAX`,
`--bin-edges-file PATH`, `--force-binsearch`. The paper's experiments use
`--force-binsearch`, which selects the general binary-search bin lookup (this is the
setting behind the `*_binsearch` result files).

**Input data.** With no `RDF_INPUT` set, the binary generates a Uniform i.i.d. point
set internally (fixed seed). To run on any other point set, pass a coordinate file
(one point per line, whitespace-separated; the first `dim` columns are used) via the
`RDF_INPUT` environment variable:
```bash
RDF_INPUT=data/normal_3d_sigma0.1.xyz ./acs 1000000 0.2 1 0.1 ring 1 --dim 3
```

The paper reports **single-thread** histogram-computation time (`num_threads=1`),
which equals total algorithmic work; the program prints `Histogram computation:` and
`Total time:` lines.

## Quick start

```bash
make
# Ring Enumeration, approximate, 1e6 uniform points, 2D, w=0.2:
./acs 1000000 0.2 1 0.1 ring 1 --force-binsearch

# Same input, compare the three methods (2D, 2e5 points, w=0.2):
./acs 200000 0.2 1 0.1 dualtree 1 --force-binsearch
./acs 200000 0.2 1 0.1 autocorr 1 --force-binsearch
./acs 200000 0.2 1 0.1 ring     1 --force-binsearch
```

## Reproducing the paper

- **Uniform** experiments need no data files (generated internally).
- **Normal** workloads: generate with `scripts/gen_normal.py` (σ ∈ {0.05, 0.1, 0.2}).
- **Real-world** datasets (GeoNames, TIGER, Ar-vapor MD): see [`data/README.md`](data/README.md)
  for download/conversion instructions — these files are large and are not bundled.

Sweep the `(N, w)` grid by looping `acs` over the point counts and bucket widths,
selecting `dualtree`/`autocorr`/`ring` with `approx 1`, at `avg_points_per_leaf=0.1`
(the paper's λ) and `--force-binsearch`.

## Repository layout

```
Autocorrelation-Based-SDH/
├── src/acs.c            # the implementation (all three methods, exact + approximate, d=1..4)
├── Makefile
├── scripts/
│   └── gen_normal.py        # generate clustered (Normal) synthetic inputs
├── data/README.md           # how to obtain the synthetic and real datasets
└── README.md
```

## Notes

- Exact mode reproduces the brute-force histogram; approximate mode applies the
  overlap-proportional bucket split and is the scalable operating point used in the
  experiments.

## Citation

If you use this code, please cite the paper (BibTeX to be added on publication).

## License

See [LICENSE](LICENSE).
