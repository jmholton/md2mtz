# Claude Code Guide — fft_symmetry/claude_test

## Project Overview

GPU-accelerated crystallographic structure factor calculation for diffuse scatter analysis.

The core idea: compute F(hkl) for a **P1 supercell** via GPU FFT, then **collapse** to the primitive cell's ASU using space-group symmetry operators. This gives two outputs per MD frame:

- `outI` — supercell intensities `|F_super|²` (for computing `<|F|²>`)
- `outmtz` — primitive-cell ASU phased structure factors FC, PHIC (for computing `<F>`)

Diffuse scatter: `I_diffuse(H) = <|F_SG(H)|²> - |<F_SG(H)>|²`

## Key Files

| File | Purpose |
|------|---------|
| `sfcalc_gpu_collapse.py` | Main script: reads PDB, calls GPU FFT, collapses to ASU |
| `sfcalc_gpu.cu` | CUDA source for the GPU FFT kernel |
| `sfcalc_gpu.so` | Compiled shared library (compile on voltron only) |
| `sfcalc_gpu.py` | Older P1-only GPU sfcalc (reference/comparison) |
| `compile_gpu.csh` | Compilation script — run on voltron, not via srun |
| `test_one_sg.py` | Test one space group: GPU collapse vs gemmi sfcalc |
| `test_all_sg.py` | Run test_one_sg.py for all 230 SGs in parallel |
| `check_asu.py` | Diagnostic: print gemmi ASU reflections for a SG |
| `diag_ratio.py` | Diagnostic: print FC ratio between two MTZ files |
| `dump_mtz_hkl.py` | Diagnostic: print first 20 HKL from an MTZ |
| `compare_mtz.py` | Compare two MTZ files (used by test_one_sg.py) |

## Collapse Formula

```
F_SG(H,K,L) = Σ_op  exp(2πi H·t_op) * F_super(na*R_op^T · H, nb*..., nc*...)
```

where `t_op` is the translational part of each symmetry operator (in fractional coords) and `(na,nb,nc)` are the supercell multipliers.

## Environment

- **GPU machine**: voltron (remote, SSH)
- **Python**: `ccp4-python` (CCP4-bundled Python with gemmi, numpy, etc.)
- **Shell on voltron**: tcsh
- **CUDA**: sourced via `/programs/cuda/setup_cuda.csh`

## SSH Rule

**Always** use `cd $PWD ;` (semicolon, not `&&`) before any command on voltron:

```csh
ssh voltron "cd /home/jamesh/projects/fft_symmetry/claude_test ; ccp4-python test_one_sg.py 19"
```

- tcsh does **not** support `&&` or `||` — use `;` for sequencing
- tcsh does **not** support `2>&1` — use `>&` or omit
- Never use bash multiline `-c "..."` syntax over SSH to a tcsh host
- Keep SSH strings to one or two simple commands; write a `.csh` script for anything complex

## Build

```csh
ssh voltron "cd /home/jamesh/projects/fft_symmetry/claude_test ; tcsh compile_gpu.csh"
```

Compiles `sfcalc_gpu.cu` → `sfcalc_gpu.so` (sm_70 architecture, cufft).

## Test

Single space group:
```csh
ssh voltron "cd /home/jamesh/projects/fft_symmetry/claude_test ; ccp4-python test_one_sg.py 'P 21 21 21' dmin=2.0"
```

All 230 space groups (parallel):
```csh
ssh voltron "cd /home/jamesh/projects/fft_symmetry/claude_test ; ccp4-python test_all_sg.py"
```

Pass criteria: ≥90% common reflections, mean relative diff <0.5%, max relative diff <5%.

## ASU Enumeration (build_prim_asu)

**Use `gemmi.ReciprocalAsu(sg).is_in((H,K,L))`** — takes a plain tuple, not a `gemmi.Miller`.  
**Use `sg.operations().is_systematically_absent((H,K,L))`** — filters extinct reflections.

The old hand-coded `asu_mask_for_laue` (still in the file as dead code) was wrong for many space groups — e.g., for I23 it selected only 5/216 correct ASU reflections.

Loop H from `-H_max` to `+H_max` (triclinic P1 has valid ASU reflections with H<0).

## Known Issues / In Progress

- **P1 amplitude mismatch** (as of 2026-04-06): GPU sfcalc_gpu_collapse.py gives ~57% mean amplitude difference vs gemmi sfcalc on the same P1 structure, even with 100% reflection coverage. Root cause not yet determined. Candidates:
  - HKL sign convention mismatch between GPU output and comparison tool
  - B-factor/form-factor differences
  - Normalization issue specific to the collapse code's P1 path
  - Note: the older `sfcalc_gpu.py` (non-collapse) passed P1 tests — compare normalization paths
- **test_all_sg.py** not yet run end-to-end (blocked by the amplitude issue above)

## gemmi API Notes

```python
sg  = gemmi.find_spacegroup_by_name('I 2 3')
asu = gemmi.ReciprocalAsu(sg)
asu.is_in((1, 0, 0))          # True/False — plain tuple, NOT gemmi.Miller
sg.operations().is_systematically_absent((1, 0, 0))  # True/False
sg.laue_str()                  # e.g. 'm-3'
sg.crystal_system_str()        # e.g. 'cubic'
```
