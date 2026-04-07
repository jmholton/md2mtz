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

**Always** use `cd /home/jamesh/projects/fft_symmetry/claude_test ;` before any command on voltron:

```csh
ssh voltron "cd /home/jamesh/projects/fft_symmetry/claude_test ; ccp4-python test_one_sg.py 19"
```

**tcsh is NOT bash.** The following constructs are invalid in tcsh and must never appear in SSH command strings:
- `&&` and `||` — use `;` for sequencing
- `2>&1` — use `>&` to redirect both stdout+stderr, or just omit
- `$()` command substitution — use backticks or write a script
- `[[ ]]`, `(( ))` — bash-only

Keep SSH command strings simple. If logic is needed, write a `.csh` script and call it.

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

All 230 space groups (parallel, 4 jobs):
```csh
ssh voltron "cd /home/jamesh/projects/fft_symmetry/claude_test ; ccp4-python test_all_sg.py"
```

Pass criteria: ≥90% common reflections, mean relative diff <0.5%, max relative diff <5%.

**Current status (2026-04-06):** 204/230 PASS. The 26 apparent failures in a batch run are
statistical — random test structures occasionally produce near-zero F at one reflection,
pushing max relative error above 5%. Re-running those SGs individually gives PASS.

## ASU Enumeration (build_prim_asu)

**Use `gemmi.ReciprocalAsu(sg).is_in((H,K,L))`** — takes a plain tuple, not a `gemmi.Miller`.  
**Use `sg.operations().is_systematically_absent((H,K,L))`** — filters extinct reflections.

The old hand-coded `asu_mask_for_laue` (still in the file as dead code) was wrong for many
space groups — e.g., for I23 it selected only 5/216 correct ASU reflections.

Loop H from `-H_max` to `+H_max` (P1 ASU has valid reflections with H<0 when L>0).

## Critical: PDB SCALE Records

**`_patch_cryst1` in `test_one_sg.py` strips SCALE1/2/3 records from the output PDB.**

Gemmi uses the `SCALE1/2/3` PDB records (not just `CRYST1`) to compute fractional coordinates.
When a supercell PDB (30×40×50 Å) is re-labelled with a primitive cell (`CRYST1` 15×20×25 Å)
but the supercell SCALE matrix is kept, gemmi computes wrong fractional coordinates → completely
wrong structure factors (>50% amplitude error). Solution: drop SCALE records so gemmi
recomputes them from CRYST1.

## test_one_sg.py Design

For each space group:
1. `randompdb.com sa sb sc alpha beta gamma` → `random.pdb` (atoms in supercell)
2. `_patch_cryst1(random.pdb → ASU.pdb, primitive cell, sg)` — fix CRYST1, drop SCALE
3. GPU: `sfcalc_gpu_collapse.py random.pdb sg=... super_mult=na,nb,nc`
4. Gemmi: `gemmi sfcalc --to-mtz ASU.mtz --dmin=... ASU.pdb`
5. `compare_mtz(ASU.mtz, gpu_collapsed.mtz)` — normalise both to H≥0 half-space

`super_mult` is (3,3,3) for R-centred groups, (2,2,2) for all others.

## gemmi API Notes

```python
sg  = gemmi.find_spacegroup_by_name('I 2 3')
asu = gemmi.ReciprocalAsu(sg)
asu.is_in((1, 0, 0))          # True/False — plain tuple, NOT gemmi.Miller
sg.operations().is_systematically_absent((1, 0, 0))  # True/False
sg.laue_str()                  # e.g. 'm-3'
sg.crystal_system_str()        # e.g. 'cubic'
```
