#!/usr/bin/env ccp4-python
"""
sfcalc_gpu.py
=============
GPU-accelerated structure factor calculator.

Matches: gemmi sfcalc --dmin=DMIN --rate=RATE --write-map=MAP --to-mtz=MTZ PDB

Usage:
  sfcalc_gpu.py  input.pdb  [dmin=1.5]  [rate=2.5]  [outmtz=out.mtz]
                 [outmap=out.map]  [bmax=0]  [lib=sfcalc_gpu.so]

  dmin   : resolution cutoff in Angstroms  (default 1.5)
  rate   : grid oversampling rate          (default 2.5)
  outmtz : output MTZ filename             (default sfcalc_gpu.mtz)
  outmap : output CCP4 map filename        (default sfcalc_gpu.map, empty = skip)
  bmax   : skip atoms with B > bmax        (default 0 = keep all)
  lib    : path to compiled .so            (default ./sfcalc_gpu.so)

Multi-grid algorithm
--------------------
Atoms are binned by B-factor using the relation B = 9*d^2 (the B where the
FWHM of the B-blurring kernel equals the FWHM of the resolution kernel at
d-spacing d).  Atoms at level L are spread on a grid with d_L = dmin*2^L
and added to the fine-grid F array in reciprocal space.

  Level 0  (d = dmin   ):  B < 9*(2*dmin)^2
  Level 1  (d = 2*dmin ):  9*(2*dmin)^2 <= B < 9*(4*dmin)^2
  Level k  (d = 2^k*dmin): 9*(2^k*dmin)^2 <= B < 9*(2^(k+1)*dmin)^2

Each level has ~8x fewer voxels than the previous one.
All accumulation is done in float32 to avoid large complex128 temporaries.
"""

import sys
import os
import ctypes
import math
import numpy as np
import gemmi

ELEM_IDX   = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'P': 4, 'S': 5}
DEFAULT_ELEM = 0


def parse_args(argv):
    args = {
        'pdb':    None,
        'dmin':   1.5,
        'rate':   2.5,
        'outmtz': 'sfcalc_gpu.mtz',
        'outmap': 'sfcalc_gpu.map',
        'bmax':   0.0,
        'lib':    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sfcalc_gpu.so'),
    }
    for arg in argv[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower().strip()
            if   key == 'dmin':                args['dmin']   = float(val)
            elif key == 'rate':                args['rate']   = float(val)
            elif key in ('outmtz', 'mtz'):     args['outmtz'] = val
            elif key in ('outmap', 'map'):     args['outmap'] = val
            elif key in ('bmax','bmax_skip'):  args['bmax']   = float(val)
            elif key == 'lib':                 args['lib']    = val
        elif arg.endswith('.pdb') or arg.endswith('.cif'):
            args['pdb'] = arg
    return args


def good_fft_size(n):
    best = n * 10
    i2 = 1
    while i2 <= n * 2:
        i3 = i2
        while i3 <= n * 2:
            i5 = i3
            while i5 <= n * 2:
                if i5 >= n:
                    best = min(best, i5)
                i5 *= 5
            i3 *= 3
        i2 *= 2
    return best


def compute_levels(dmin, rate, ax, ay, az, min_pts=4):
    """Return list of (d_level, nx, ny, nz) for each level."""
    levels = []
    d = dmin
    while True:
        s = d / (2.0 * rate)
        nx = good_fft_size(max(min_pts, math.ceil(ax / s)))
        ny = good_fft_size(max(min_pts, math.ceil(ay / s)))
        nz = good_fft_size(max(min_pts, math.ceil(az / s)))
        levels.append((d, nx, ny, nz))
        if min(nx, ny, nz) <= min_pts:
            break
        d *= 2.0
    return levels


def assign_levels(B_arr, dmin, n_levels):
    """Assign each atom to its coarsest adequate grid level (vectorised).
    Level L is adequate when d_L = dmin*2^L <= sqrt(B/9).
    => L = floor(0.5 * log2(B / (9*dmin^2))), clamped to [0, n_levels-1].
    """
    ratio = B_arr / (9.0 * dmin * dmin)
    log2r = np.where(ratio > 1.0, np.log2(ratio.clip(min=1.0)), 0.0)
    lev   = np.floor(0.5 * log2r).astype(np.int32)
    return np.clip(lev, 0, n_levels - 1)


def run_gpu_raw(lib, x, y, z, B, el, nx, ny, nz, ax, ay, az, do_map=False):
    """Call the GPU spreading+FFT library.
    Returns (F_real_flat, F_imag_flat, map_buf_or_None, nkept).
    F_real/F_imag are float32, shape (nz*(ny*(nx//2+1)),), NOT yet normalised.
    """
    nx2   = nx // 2 + 1
    fft_n = nx2 * ny * nz

    # Use bytearray backing instead of np.zeros.  On Linux, np.zeros for large
    # arrays uses mmap with lazy physical page allocation (calloc semantics),
    # so the first write (cudaMemcpy) triggers ~1 µs/page faults (~1.6 s for
    # 340 MB).  bytearray() pre-faults pages by actually writing zeros in the
    # constructor, so cudaMemcpy runs at full PCIe bandwidth.
    _rbuf  = bytearray(fft_n * 4)
    _ibuf  = bytearray(fft_n * 4)
    F_real = np.frombuffer(_rbuf, dtype=np.float32)
    F_imag = np.frombuffer(_ibuf, dtype=np.float32)

    if do_map:
        _mbuf   = bytearray(nx * ny * nz * 4)
        map_buf = np.frombuffer(_mbuf, dtype=np.float32)
    else:
        map_buf = None

    def fptr(arr):
        if arr is None:
            return ctypes.cast(None, ctypes.POINTER(ctypes.c_float))
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    nkept = lib.spread_and_fft(
        len(x),
        fptr(x), fptr(y), fptr(z), fptr(B),
        el.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nx, ny, nz,
        ctypes.c_float(ax), ctypes.c_float(ay), ctypes.c_float(az),
        ctypes.c_float(0.0),          # bmax_skip disabled; Python pre-filters
        fptr(map_buf), fptr(F_real), fptr(F_imag),
    )
    if nkept < 0:
        sys.exit(f"ERROR: spread_and_fft returned {nkept}")
    return F_real, F_imag, map_buf, nkept


def add_to_fine(acc, nx, ny, nz, coarse, nx_c, ny_c, nz_c):
    """Add coarse float32 array (nz_c, ny_c, nx_c2) into fine (nz, ny, nx2)
    using physical Miller-index mapping."""
    nx_c2 = nx_c // 2 + 1

    K_c   = np.arange(ny_c, dtype=np.int32)
    K_val = np.where(K_c <= ny_c // 2, K_c, K_c - ny_c)
    iy_f  = np.where(K_val >= 0, K_val, K_val + ny)

    L_c   = np.arange(nz_c, dtype=np.int32)
    L_val = np.where(L_c <= nz_c // 2, L_c, L_c - nz_c)
    iz_f  = np.where(L_val >= 0, L_val, L_val + nz)

    acc[iz_f[:, None, None],
        iy_f[None, :, None],
        np.arange(nx_c2)[None, None, :]] += coarse


def main():
    args = parse_args(sys.argv)
    if args['pdb'] is None:
        print(__doc__)
        sys.exit("ERROR: no PDB file specified")

    dmin = args['dmin']
    rate = args['rate']
    bmax = args['bmax']

    # ------------------------------------------------------------------
    # 1. Load PDB
    # ------------------------------------------------------------------
    print(f"Reading {args['pdb']} ...")
    st   = gemmi.read_structure(args['pdb'])
    cell = st.cell
    sg   = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    print(f"  Space group: {sg.hm}")

    ax, ay, az = cell.a, cell.b, cell.c
    print(f"  Cell: {ax:.3f} x {ay:.3f} x {az:.3f}  "
          f"angles: {cell.alpha:.2f} {cell.beta:.2f} {cell.gamma:.2f}")
    if (abs(cell.alpha - 90) > 0.01 or abs(cell.beta  - 90) > 0.01
                                     or abs(cell.gamma - 90) > 0.01):
        sys.exit("ERROR: GPU kernel currently only supports orthogonal cells")

    xs, ys, zs, Bs, els = [], [], [], [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    xs.append(p.x); ys.append(p.y); zs.append(p.z)
                    Bs.append(atom.b_iso)
                    els.append(ELEM_IDX.get(atom.element.name.upper().strip(), DEFAULT_ELEM))

    x_arr  = np.array(xs,  dtype=np.float32)
    y_arr  = np.array(ys,  dtype=np.float32)
    z_arr  = np.array(zs,  dtype=np.float32)
    B_arr  = np.maximum(np.array(Bs, dtype=np.float32), 0.0)
    el_arr = np.array(els, dtype=np.int32)
    natoms = len(x_arr)
    print(f"  Atoms: {natoms}  B: {B_arr.min():.2f} .. {B_arr.max():.2f}")

    if bmax > 0:
        keep  = B_arr <= bmax
        nskip = int((~keep).sum())
        if nskip:
            print(f"  Skipping {nskip} atoms with B > {bmax}")
        x_arr, y_arr, z_arr, B_arr, el_arr = (
            a[keep] for a in (x_arr, y_arr, z_arr, B_arr, el_arr))
        natoms = int(keep.sum())

    # ------------------------------------------------------------------
    # 2. Build multi-grid level table and assign atoms
    # ------------------------------------------------------------------
    levels   = compute_levels(dmin, rate, ax, ay, az)
    n_levels = len(levels)
    atom_lev = assign_levels(B_arr, dmin, n_levels)

    d0, nx, ny, nz = levels[0]
    nx2 = nx // 2 + 1
    V_cell = ax * ay * az

    print(f"  Multi-grid levels (B = 9*d^2 criterion):")
    for L, (d_L, nx_L, ny_L, nz_L) in enumerate(levels):
        n_L  = int((atom_lev == L).sum())
        B_lo = 9.0 * (dmin * 2**L)**2       if L > 0             else 0.0
        B_hi = 9.0 * (dmin * 2**(L+1))**2   if L < n_levels - 1  else float('inf')
        vfrac = (nx_L * ny_L * nz_L) / (nx * ny * nz) * 100
        print(f"    L{L}: d={d_L:.1f}A  grid {nx_L}x{ny_L}x{nz_L} "
              f"({vfrac:.0f}% voxels)  B=[{B_lo:.0f},{B_hi:.0f})  {n_L} atoms")

    # ------------------------------------------------------------------
    # 3. Load GPU library
    # ------------------------------------------------------------------
    lib_path = args['lib']
    if not os.path.exists(lib_path):
        sys.exit(f"ERROR: shared library not found: {lib_path}")

    lib = ctypes.CDLL(lib_path)
    lib.spread_and_fft.restype  = ctypes.c_int
    lib.spread_and_fft.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    # ------------------------------------------------------------------
    # 4. Spread each level; accumulate into fine-grid float32 R/I arrays.
    #    Conjugate and normalise in-place before accumulating to avoid
    #    large complex128 temporaries.
    # ------------------------------------------------------------------
    do_map = bool(args['outmap'])

    # Fine-grid accumulators (float32, shape (nz, ny, nx2))
    acc_real = np.zeros((nz, ny, nx2), dtype=np.float32)
    acc_imag = np.zeros((nz, ny, nx2), dtype=np.float32)
    map_buf  = None
    nkept_total = 0

    import time
    t_start = time.perf_counter()

    for L, (d_L, nx_L, ny_L, nz_L) in enumerate(levels):
        mask_L = (atom_lev == L)
        n_L    = int(mask_L.sum())
        if n_L == 0:
            continue

        t0 = time.perf_counter()
        Fr, Fi, mb, nk = run_gpu_raw(
            lib,
            x_arr[mask_L], y_arr[mask_L], z_arr[mask_L],
            B_arr[mask_L], el_arr[mask_L],
            nx_L, ny_L, nz_L, ax, ay, az,
            do_map=(do_map and L == 0),
        )
        t1 = time.perf_counter()

        # Normalise: V_cell/(nx*ny*nz); negate imag for cuFFT->crystallographic conjugate
        norm_L = np.float32(V_cell / (nx_L * ny_L * nz_L))
        Fr *= norm_L
        Fi *= -norm_L

        nx_L2 = nx_L // 2 + 1
        Fr3 = Fr.reshape(nz_L, ny_L, nx_L2)
        Fi3 = Fi.reshape(nz_L, ny_L, nx_L2)

        if L == 0:
            acc_real += Fr3
            acc_imag += Fi3
            if mb is not None:
                map_buf = mb
        else:
            add_to_fine(acc_real, nx, ny, nz, Fr3, nx_L, ny_L, nz_L)
            add_to_fine(acc_imag, nx, ny, nz, Fi3, nx_L, ny_L, nz_L)

        t2 = time.perf_counter()
        print(f"    L{L}: {n_L} atoms  GPU {(t1-t0)*1000:.0f} ms  combine {(t2-t1)*1000:.0f} ms")
        nkept_total += nk

    t_end = time.perf_counter()
    print(f"  GPU total time: {(t_end-t_start)*1000:.0f} ms  ({nkept_total} atoms spread)")

    # ------------------------------------------------------------------
    # 5. Extract ASU reflections -> MTZ
    #    Work entirely in float32; no large complex128 array created.
    # ------------------------------------------------------------------
    print("Extracting structure factors from FFT output ...")

    H_1d = np.arange(nx2, dtype=np.int32)
    K_1d = np.arange(ny,  dtype=np.int32)
    L_1d = np.arange(nz,  dtype=np.int32)
    K_1d = np.where(K_1d > ny // 2, K_1d - ny, K_1d)
    L_1d = np.where(L_1d > nz // 2, L_1d - nz, L_1d)

    inv_d2    = ((H_1d[None, None, :] / ax) ** 2 +
                 (K_1d[None, :,  None] / ay) ** 2 +
                 (L_1d[:,  None, None] / az) ** 2)
    inv_dmin2 = 1.0 / (dmin * dmin)

    H3 = H_1d[None, None, :]
    K3 = K_1d[None, :,  None]
    L3 = L_1d[:,  None, None]

    laue = sg.laue_str()
    if laue in ('-1',):
        asu_mask = (L3 > 0) | ((L3 == 0) & (K3 > 0)) | ((L3 == 0) & (K3 == 0) & (H3 > 0))
    elif laue in ('2/m',):
        asu_mask = (H3 >= 0) & (L3 >= 0) & ((H3 > 0) | (K3 >= 0))
    elif laue in ('mmm',):
        asu_mask = (H3 >= 0) & (K3 >= 0) & (L3 >= 0) & ~((H3 == 0) & (K3 == 0) & (L3 == 0))
    elif laue in ('4/m', '4/mmm'):
        asu_mask = (H3 >= K3) & (K3 >= 0) & (L3 >= 0) & ~((H3 == 0) & (K3 == 0) & (L3 == 0))
    elif laue in ('-3', '-3m', '6/m', '6/mmm'):
        asu_mask = ((H3 >= 0) & (K3 >= 0) & (L3 >= 0) & (H3 >= K3) &
                    ~((H3 == 0) & (K3 == 0) & (L3 == 0)))
    elif laue in ('m-3', 'm-3m'):
        asu_mask = (H3 >= K3) & (K3 >= L3) & (L3 >= 0) & ~((H3 == 0) & (K3 == 0) & (L3 == 0))
    else:
        sys.exit(f"ERROR: unsupported Laue class '{laue}' for space group {sg.hm}")

    mask = (inv_d2 <= inv_dmin2) & asu_mask

    # Extract directly from float32 accumulators — no complex128 array needed
    re_sel = acc_real[mask]
    im_sel = acc_imag[mask]
    amp    = np.hypot(re_sel, im_sel).astype(np.float32)
    phi    = np.degrees(np.arctan2(im_sel, re_sel)).astype(np.float32)

    H_sel = np.broadcast_to(H3, (nz, ny, nx2))[mask].astype(np.float32)
    K_sel = np.broadcast_to(K3, (nz, ny, nx2))[mask].astype(np.float32)
    L_sel = np.broadcast_to(L3, (nz, ny, nx2))[mask].astype(np.float32)

    out_data = np.column_stack([H_sel, K_sel, L_sel, amp, phi])
    print(f"  ASU reflections: {len(out_data)}")

    out_mtz            = gemmi.Mtz(with_base=False)
    out_mtz.spacegroup = sg
    out_mtz.cell       = cell
    base_ds = out_mtz.add_dataset("HKL_base");   base_ds.wavelength = 0.0
    data_ds = out_mtz.add_dataset("SFCALC_GPU"); data_ds.wavelength = 1.0
    ds_id   = data_ds.id

    for label, ctype in [('H','H'),('K','H'),('L','H'),('FC','F'),('PHIC','P')]:
        col = out_mtz.add_column(label, ctype)
        col.dataset_id = 0 if ctype == 'H' else ds_id

    if len(out_data):
        out_mtz.set_data(out_data)
    out_mtz.write_to_file(args['outmtz'])
    print(f"  Written: {args['outmtz']}")

    # ------------------------------------------------------------------
    # 6. Write CCP4 map (level-0 atoms only when multi-grid is active)
    # ------------------------------------------------------------------
    if do_map and map_buf is not None:
        n_coarse = int((atom_lev > 0).sum())
        if n_coarse:
            print(f"  Note: map contains level-0 atoms only "
                  f"({natoms - n_coarse} of {natoms})")
        rho_3d = map_buf.reshape(nz, ny, nx)
        ccp4   = gemmi.Ccp4Map()
        ccp4.grid = gemmi.FloatGrid(rho_3d.astype(np.float32))
        ccp4.grid.unit_cell  = cell
        ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(args['outmap'])
        print(f"  Written: {args['outmap']}")

    print("Done.")


if __name__ == '__main__':
    main()
