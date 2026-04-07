# sfcalc_gpu_collapse — GPU Structure Factors for Diffuse Scatter

GPU-accelerated calculation of crystallographic structure factors from MD supercell trajectories, for diffuse scatter analysis.

## Background

X-ray diffuse scatter carries information about correlated atomic motion that is invisible to conventional Bragg analysis. For an MD ensemble of supercell frames, the diffuse intensity is:

```
I_diffuse(H) = <|F_SG(H)|²> - |<F_SG(H)>|²
```

where `H = (H,K,L)` is a primitive-cell Miller index and the angle brackets denote averaging over MD frames. Computing this requires two quantities per frame:

- `|F_SG(H)|²` — squared amplitude in the primitive-cell ASU
- `F_SG(H)` — complex structure factor (amplitude + phase) in the primitive-cell ASU

This software calculates both efficiently using GPU FFT.

## Method

Each MD frame is stored as a P1 supercell PDB (`na × nb × nc` copies of the primitive cell). The GPU spreads atoms onto a density grid and computes the 3-D FFT, giving `F_super(h,k,l)` for the supercell in P1. These are then **collapsed** to the primitive-cell ASU using the space-group symmetry operators:

```
F_SG(H,K,L) = Σ_op  exp(2πi H·t_op)  ×  F_super(na·R_op^T·H, nb·R_op^T·K, nc·R_op^T·L)
```

where `R_op` and `t_op` are the rotation and translation of each symmetry operator (fractional coordinates). The collapse is exact for any space group in any crystal system.

Two MTZ files are written per frame:

| Output | Contents | Used for |
|--------|----------|----------|
| `outI` | Supercell P1 intensities `I = \|F_super\|²` at all h,k,l to `dmin` | Computing `<\|F\|²>` by averaging I across frames |
| `outmtz` | Primitive-cell ASU amplitudes FC + phases PHIC | Computing `\|<F>\|²` by averaging complex F across frames, then squaring |

## Requirements

- **NVIDIA GPU**, compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **CCP4 suite** — for `ccp4-python` (provides Python, numpy, and gemmi)
- **CUDA driver** 520+ (CUDA 11 runtime is bundled in `libcufft.so.11`)
- `sfcalc_gpu.so` and `libcufft.so.11` in the same directory (see Build)

## Build

Compile once on a machine with the CUDA toolkit installed (e.g. voltron):

```csh
tcsh compile_gpu.csh
```

This produces `sfcalc_gpu.so` (multi-arch: sm_70 through sm_90 + PTX fallback).

**Distribution**: ship `sfcalc_gpu.so` and `libcufft.so.11` together in the same directory. The `$ORIGIN` rpath in `sfcalc_gpu.so` means the CUDA toolkit does not need to be installed on the target machine — only an NVIDIA driver is required.

## Usage

```
ccp4-python sfcalc_gpu_collapse.py  <input.pdb>
    [dmin=1.5]          resolution limit in Angstroms (default 1.5)
    [rate=2.5]          Shannon rate (grid oversampling, default 2.5)
    [sg=P1]             space group of the PRIMITIVE cell
    [super_mult=1,1,1]  supercell multipliers na,nb,nc
    [outmtz=collapsed.mtz]   primitive-cell ASU phased MTZ
    [outI=supercell_I.mtz]   supercell P1 intensity MTZ
    [outmap=]           optional: write electron density map (CCP4 format)
    [bmax=0]            skip atoms with B > bmax (0 = keep all)
    [noise=0.01]        multi-grid noise tolerance (default 1%)
    [lib=sfcalc_gpu.so] path to the GPU shared library
```

### Example: orthorhombic 2×2×2 supercell

```csh
ccp4-python sfcalc_gpu_collapse.py  frame_001.pdb \
    sg="P 21 21 21"  super_mult=2,2,2  dmin=2.0 \
    outmtz=frame_001_collapsed.mtz  outI=frame_001_I.mtz
```

The input PDB must be a P1 supercell with `CRYST1` giving the supercell dimensions. The `sg` argument specifies the space group of the **primitive** unit cell. The script derives the primitive-cell dimensions by dividing the supercell lengths by `na`, `nb`, `nc`.

### P1 single-cell calculation

If `super_mult=1,1,1` and `sg=P1`, both outputs cover the same P1 ASU reflections and the collapse step is skipped.

## Averaging over an MD trajectory

For each frame `i`, run the script and save both MTZ files. Then:

1. **`<|F|²>`**: average the `I` column across all `outI` MTZs (standard MTZ merging).
2. **`|<F>|²`**: sum the complex structure factors (FC, PHIC) from all `outmtz` files, divide by N to get the mean complex F, then square the amplitude.
3. **`I_diffuse = <|F|²> - |<F>|²`** at each primitive-cell ASU reflection.

CCP4 programs `SCALEIT`, `CAD`, and `ADDUP` can assist with step 1. Steps 2–3 are most conveniently done in Python with gemmi or cctbx.

## Multi-grid B-factor treatment

Atoms are assigned to FFT grid levels based on their isotropic B-factor, so high-B (diffuse) atoms are spread on coarser grids and low-B (well-ordered) atoms use the full fine grid. This reduces GPU memory and time without sacrificing accuracy. The `noise` parameter controls the coarsening threshold (default 1%).

## Testing

Validate the GPU collapse against a gemmi reference for any space group:

```csh
# Single space group (by number or HM name)
ccp4-python test_one_sg.py 19
ccp4-python test_one_sg.py "P 21 21 21"  dmin=2.0

# All 230 space groups in parallel
ccp4-python test_all_sg.py
```

Pass criteria: ≥90% of gemmi reflections found, mean relative amplitude difference <0.5%, max <5%.

## Files

| File | Purpose |
|------|---------|
| `sfcalc_gpu_collapse.py` | Main Python script |
| `sfcalc_gpu.cu` | CUDA C source |
| `sfcalc_gpu.so` | Compiled GPU shared library |
| `libcufft.so.11` | cuFFT runtime (bundled for portability) |
| `compile_gpu.csh` | Build script (run on GPU machine) |
| `test_one_sg.py` | Per-space-group validation |
| `test_all_sg.py` | Full 230 SG test suite |
