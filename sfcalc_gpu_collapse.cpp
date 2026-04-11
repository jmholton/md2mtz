/*
 * sfcalc_gpu_collapse.cpp
 * =======================
 * C++ port of sfcalc_gpu_collapse.py — same algorithm, no Python overhead.
 *
 * Usage: same command-line interface as the Python script.
 *   ./sfcalc_gpu_collapse  input.pdb  [dmin=1.5]  [rate=2.5]  [sg=P1]
 *       [super_mult=1,1,1]  [outmtz=collapsed.mtz]  [outI=supercell_I.mtz]
 *       [outmap=]  [bmax=0]  [noise=0.01]  [lib=sfcalc_gpu.so]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <dlfcn.h>

#include <gemmi/model.hpp>
#include <gemmi/mmread.hpp>
#include <gemmi/symmetry.hpp>
#include <gemmi/mtz.hpp>

// ---------------------------------------------------------------------------
// GPU library function (extern "C" in sfcalc_gpu.cu)
// ---------------------------------------------------------------------------
typedef int (*SpreadAndFft_fn)(
    int natoms,
    float *x, float *y, float *z, float *B, int *elem,
    int nx, int ny, int nz,
    float ax, float ay, float az,
    float alpha, float beta, float gamma,
    float Bmax_skip,
    float *map_out, float *F_real, float *F_imag
);

// ---------------------------------------------------------------------------
// Element index (must match order in sfcalc_gpu.cu)
// ---------------------------------------------------------------------------
static int elem_idx(const std::string& name) {
    if (name == "C") return 0;
    if (name == "H") return 1;
    if (name == "N") return 2;
    if (name == "O") return 3;
    if (name == "P") return 4;
    if (name == "S") return 5;
    return 0;  // default C
}

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------
static const double LEVEL_FACTOR = 1.41421356237; // sqrt(2)

static int good_fft_size(int n) {
    int best = n * 10;
    for (int i2 = 1; i2 <= n*2; i2 *= 2)
        for (int i3 = i2; i3 <= n*2; i3 *= 3)
            for (int i5 = i3; i5 <= n*2; i5 *= 5)
                if (i5 >= n) best = std::min(best, i5);
    return best;
}

struct Level { double d; int nx, ny, nz; };

static std::vector<Level> compute_levels(double dmin, double rate,
                                          double ax, double ay, double az,
                                          int min_pts = 4) {
    std::vector<Level> levels;
    double d = dmin;
    while (true) {
        double s  = d / (2.0 * rate);
        int nx = good_fft_size(std::max(min_pts, (int)ceil(ax / s)));
        int ny = good_fft_size(std::max(min_pts, (int)ceil(ay / s)));
        int nz = good_fft_size(std::max(min_pts, (int)ceil(az / s)));
        levels.push_back({d, nx, ny, nz});
        if (std::min({nx, ny, nz}) <= min_pts) break;
        d *= LEVEL_FACTOR;
    }
    return levels;
}

static double noise_wpx(double noise_frac, double nb2d = 4.0, double nb3d = 6.0) {
    double ln2      = log(2.0);
    double noise_2d = noise_frac * (nb2d / nb3d);
    return (4.0/9.0) * sqrt(-log(log(noise_2d + 1.0)) / ln2);
}

static void assign_levels(const std::vector<float>& B, double pixel_fine,
                           double noise_frac, int n_levels,
                           std::vector<int>& out_lev) {
    double ln2   = log(2.0);
    double log_f = log(LEVEL_FACTOR);
    double w_px  = noise_wpx(noise_frac);
    out_lev.resize(B.size());
    for (size_t i = 0; i < B.size(); i++) {
        double fwhm      = sqrt(ln2 * B[i]) / (2.0 * M_PI);
        double pixel_req = fwhm / w_px;
        double ratio     = pixel_req / pixel_fine;
        int lev = 0;
        if (ratio >= 1.0)
            lev = (int)floor(log(ratio) / log_f);
        lev = std::max(0, std::min(lev, n_levels - 1));
        out_lev[i] = lev;
    }
}

static double b_threshold_for_level(int L, double pixel_fine, double noise_frac) {
    if (L == 0) return 0.0;
    double ln2    = log(2.0);
    double w_px   = noise_wpx(noise_frac);
    double pix_L  = pixel_fine * pow(LEVEL_FACTOR, L);
    double fwhm_min = pix_L * w_px;
    return fwhm_min * fwhm_min * (4.0 * M_PI * M_PI) / ln2;
}

// ---------------------------------------------------------------------------
// Accumulate coarse FFT output into fine-grid arrays (frequency-domain insert)
// Matches the Python add_to_fine logic exactly.
// acc is [nz][ny][nx2], coarse is [nz_c][ny_c][nx_c2], both row-major.
// ---------------------------------------------------------------------------
static void add_to_fine(float* acc_re, float* acc_im,
                         int nx, int ny, int nz,
                         const float* c_re, const float* c_im,
                         int nx_c, int ny_c, int nz_c) {
    int nx_c2 = nx_c/2 + 1;
    int nx2   = nx/2 + 1;
    int Ln = nz_c/2 + 1, Lh = nz_c - Ln;
    int Kn = ny_c/2 + 1, Kh = ny_c - Kn;

    for (int iz = 0; iz < Ln; iz++) {
        int fz = iz;
        for (int iy = 0; iy < Kn; iy++) {
            size_t fi = ((size_t)fz*ny + iy) * nx2;
            size_t ci = ((size_t)iz*ny_c + iy) * nx_c2;
            for (int ix = 0; ix < nx_c2; ix++) {
                acc_re[fi+ix] += c_re[ci+ix];
                acc_im[fi+ix] += c_im[ci+ix];
            }
        }
        if (Kh) {
            for (int iy = 0; iy < Kh; iy++) {
                size_t fi = ((size_t)fz*ny + (ny-Kh+iy)) * nx2;
                size_t ci = ((size_t)iz*ny_c + (Kn+iy)) * nx_c2;
                for (int ix = 0; ix < nx_c2; ix++) {
                    acc_re[fi+ix] += c_re[ci+ix];
                    acc_im[fi+ix] += c_im[ci+ix];
                }
            }
        }
    }
    if (Lh) {
        for (int iz = 0; iz < Lh; iz++) {
            int fz = nz - Lh + iz;
            for (int iy = 0; iy < Kn; iy++) {
                size_t fi = ((size_t)fz*ny + iy) * nx2;
                size_t ci = ((size_t)(Ln+iz)*ny_c + iy) * nx_c2;
                for (int ix = 0; ix < nx_c2; ix++) {
                    acc_re[fi+ix] += c_re[ci+ix];
                    acc_im[fi+ix] += c_im[ci+ix];
                }
            }
            if (Kh) {
                for (int iy = 0; iy < Kh; iy++) {
                    size_t fi = ((size_t)fz*ny + (ny-Kh+iy)) * nx2;
                    size_t ci = ((size_t)(Ln+iz)*ny_c + (Kn+iy)) * nx_c2;
                    for (int ix = 0; ix < nx_c2; ix++) {
                        acc_re[fi+ix] += c_re[ci+ix];
                        acc_im[fi+ix] += c_im[ci+ix];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Blur correction: multiply F(h,k,l) by exp(+b_add * stol^2)
// ---------------------------------------------------------------------------
static void apply_blur(float* acc_re, float* acc_im,
                        int nx, int ny, int nz,
                        float b_add,
                        float gs11, float gs22, float gs33,
                        float gs12, float gs13, float gs23) {
    int nx2 = nx/2 + 1;
    for (int iz = 0; iz < nz; iz++) {
        float L = (iz <= nz/2) ? (float)iz : (float)(iz - nz);
        for (int iy = 0; iy < ny; iy++) {
            float K = (iy <= ny/2) ? (float)iy : (float)(iy - ny);
            size_t row = ((size_t)iz * ny + iy) * nx2;
            for (int ix = 0; ix < nx2; ix++) {
                float H = (float)ix;
                float stol2 = 0.25f * (H*H*gs11 + K*K*gs22 + L*L*gs33
                              + 2.f*(H*K*gs12 + H*L*gs13 + K*L*gs23));
                float corr = expf(b_add * stol2);
                acc_re[row+ix] *= corr;
                acc_im[row+ix] *= corr;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build primitive-cell ASU reflection list
// ---------------------------------------------------------------------------
struct HKL { int h, k, l; };

static std::vector<HKL> build_prim_asu(const gemmi::SpaceGroup* sg,
                                        const gemmi::UnitCell& cell,
                                        double dmin) {
    gemmi::UnitCell rc = cell.reciprocal();
    int H_max = (int)ceil(1.0 / (dmin * rc.a));
    int K_max = (int)ceil(1.0 / (dmin * rc.b));
    int L_max = (int)ceil(1.0 / (dmin * rc.c));

    // Reciprocal metric tensor for general cell
    double rca = cos(rc.alpha * M_PI / 180.0);
    double rcb = cos(rc.beta  * M_PI / 180.0);
    double rcg = cos(rc.gamma * M_PI / 180.0);
    double gs11 = rc.a*rc.a, gs22 = rc.b*rc.b, gs33 = rc.c*rc.c;
    double gs12 = rc.a*rc.b*rcg, gs13 = rc.a*rc.c*rcb, gs23 = rc.b*rc.c*rca;
    double inv_dmin2 = 1.0 / (dmin * dmin);

    gemmi::ReciprocalAsu asu(sg);
    gemmi::GroupOps ops = sg->operations();

    std::vector<HKL> result;
    for (int H = -H_max; H <= H_max; H++) {
        for (int K = -K_max; K <= K_max; K++) {
            for (int L = -L_max; L <= L_max; L++) {
                if (!asu.is_in({H, K, L})) continue;
                double inv_d2 = H*H*gs11 + K*K*gs22 + L*L*gs33
                              + 2.0*(H*K*gs12 + H*L*gs13 + K*L*gs23);
                if (inv_d2 <= 0 || inv_d2 > inv_dmin2) continue;
                if (ops.is_systematically_absent({H, K, L})) continue;
                result.push_back({H, K, L});
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Collapse supercell to primitive-cell ASU
// ---------------------------------------------------------------------------
static void collapse_to_prim_asu(const float* acc_re, const float* acc_im,
                                   int nx, int ny, int nz,
                                   int na, int nb, int nc,
                                   const gemmi::SpaceGroup* sg,
                                   const std::vector<HKL>& asu,
                                   std::vector<float>& F_re,
                                   std::vector<float>& F_im) {
    int nx2  = nx/2 + 1;
    int n    = (int)asu.size();
    F_re.assign(n, 0.f);
    F_im.assign(n, 0.f);

    gemmi::GroupOps ops = sg->operations();
    int den = gemmi::Op::DEN;

    for (const gemmi::Op& op : ops) {
        auto& rot  = op.rot;
        auto& tran = op.tran;

        for (int i = 0; i < n; i++) {
            int H = asu[i].h, K = asu[i].k, L = asu[i].l;

            // R^T * (H,K,L) — column-major (transposed) indexing
            int Hr = (rot[0][0]*H + rot[1][0]*K + rot[2][0]*L) / den;
            int Kr = (rot[0][1]*H + rot[1][1]*K + rot[2][1]*L) / den;
            int Lr = (rot[0][2]*H + rot[1][2]*K + rot[2][2]*L) / den;

            int SH = na * Hr, SK = nb * Kr, SL = nc * Lr;

            // Friedel mate if in H<0 half-space
            bool friedel = (SH < 0) || (SH == 0 && SK < 0)
                         || (SH == 0 && SK == 0 && SL < 0);
            if (friedel) { SH = -SH; SK = -SK; SL = -SL; }

            int ix = SH;
            int iy = (SK >= 0) ? SK : SK + ny;
            int iz = (SL >= 0) ? SL : SL + nz;

            if (ix >= nx2 || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;

            size_t idx = (size_t)iz * ny * nx2 + iy * nx2 + ix;
            float re   =  acc_re[idx];
            float im   = friedel ? -acc_im[idx] : acc_im[idx];

            // Phase factor: exp(2πi H·t)
            float phase = (float)(2.0 * M_PI *
                          (H * tran[0] + K * tran[1] + L * tran[2]) / (double)den);
            float cp = cosf(phase), sp = sinf(phase);

            F_re[i] += re * cp - im * sp;
            F_im[i] += re * sp + im * cp;
        }
    }
}

// ---------------------------------------------------------------------------
// Write MTZ (FC + PHIC)
// ---------------------------------------------------------------------------
static void write_mtz(const std::vector<HKL>& hkl,
                       const std::vector<float>& amp,
                       const std::vector<float>& phi,
                       const gemmi::UnitCell& cell,
                       const gemmi::SpaceGroup* sg,
                       const std::string& path) {
    int n = (int)hkl.size();
    gemmi::Mtz mtz(false);
    mtz.spacegroup = sg;
    mtz.cell = cell;
    auto& hkl_ds  = mtz.add_dataset("HKL_base");   hkl_ds.wavelength  = 0.0;
    auto& data_ds = mtz.add_dataset("SFCALC_GPU");  data_ds.wavelength = 1.0;
    int ds_id = data_ds.id;
    mtz.add_column("H",    'H', 0,     -1, false);
    mtz.add_column("K",    'H', 0,     -1, false);
    mtz.add_column("L",    'H', 0,     -1, false);
    mtz.add_column("FC",   'F', ds_id, -1, false);
    mtz.add_column("PHIC", 'P', ds_id, -1, false);

    std::vector<float> data(5 * n);
    for (int i = 0; i < n; i++) {
        data[5*i+0] = (float)hkl[i].h;
        data[5*i+1] = (float)hkl[i].k;
        data[5*i+2] = (float)hkl[i].l;
        data[5*i+3] = amp[i];
        data[5*i+4] = phi[i];
    }
    mtz.set_data(data.data(), 5 * n);
    mtz.write_to_file(path);
    fprintf(stderr, "  Written: %s  (%d reflections)\n", path.c_str(), n);
}

// ---------------------------------------------------------------------------
// Write intensity MTZ (I = |F|^2, P1 space)
// ---------------------------------------------------------------------------
static void write_mtz_I(const std::vector<HKL>& hkl,
                         const std::vector<float>& intensity,
                         const gemmi::UnitCell& cell,
                         const std::string& path) {
    int n = (int)hkl.size();
    const gemmi::SpaceGroup* sg_p1 = gemmi::find_spacegroup_by_name("P 1");
    gemmi::Mtz mtz(false);
    mtz.spacegroup = sg_p1;
    mtz.cell = cell;
    auto& hkl_ds  = mtz.add_dataset("HKL_base");   hkl_ds.wavelength  = 0.0;
    auto& data_ds = mtz.add_dataset("SFCALC_GPU");  data_ds.wavelength = 1.0;
    int ds_id = data_ds.id;
    mtz.add_column("H", 'H', 0,     -1, false);
    mtz.add_column("K", 'H', 0,     -1, false);
    mtz.add_column("L", 'H', 0,     -1, false);
    mtz.add_column("I", 'J', ds_id, -1, false);

    std::vector<float> data(4 * n);
    for (int i = 0; i < n; i++) {
        data[4*i+0] = (float)hkl[i].h;
        data[4*i+1] = (float)hkl[i].k;
        data[4*i+2] = (float)hkl[i].l;
        data[4*i+3] = intensity[i];
    }
    mtz.set_data(data.data(), 4 * n);
    mtz.write_to_file(path);
    fprintf(stderr, "  Written: %s  (%d reflections)\n", path.c_str(), n);
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
struct Args {
    std::string pdb, outmtz, outI, outmap, lib, sg_name;
    double dmin, rate, bmax, noise;
    int na, nb, nc;
};

static Args parse_args(int argc, char** argv) {
    // Compute default lib path: same directory as this executable
    std::string exe(argv[0]);
    std::string dir = exe.substr(0, exe.rfind('/') + 1);

    Args a;
    a.outmtz   = "collapsed.mtz";
    a.outI     = "supercell_I.mtz";
    a.outmap   = "";
    a.lib      = dir + "sfcalc_gpu.so";
    a.sg_name  = "P 1";
    a.dmin     = 1.5;
    a.rate     = 2.5;
    a.bmax     = 0.0;
    a.noise    = 0.01;
    a.na = a.nb = a.nc = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) {
            // positional: PDB file
            if (arg.size() > 4 &&
                (arg.substr(arg.size()-4) == ".pdb" ||
                 arg.substr(arg.size()-4) == ".cif"))
                a.pdb = arg;
            continue;
        }
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq+1);
        // lowercase key
        for (auto& c : key) c = tolower(c);

        if (key == "dmin")            a.dmin   = atof(val.c_str());
        else if (key == "rate")       a.rate   = atof(val.c_str());
        else if (key == "outmtz" || key == "mtz")  a.outmtz = val;
        else if (key == "outi" || key == "outsq" || key == "outisq") a.outI = val;
        else if (key == "outmap" || key == "map")  a.outmap = val;
        else if (key == "bmax")       a.bmax   = atof(val.c_str());
        else if (key == "noise")      a.noise  = atof(val.c_str());
        else if (key == "lib")        a.lib    = val;
        else if (key == "sg" || key == "spacegroup" || key == "space_group")
            a.sg_name = val;
        else if (key == "super_mult" || key == "mult" || key == "multipliers") {
            // parse na,nb,nc  (comma or x separated)
            std::string v = val;
            for (auto& c : v) if (c == 'x') c = ',';
            if (sscanf(v.c_str(), "%d,%d,%d", &a.na, &a.nb, &a.nc) != 3) {
                fprintf(stderr, "ERROR: super_mult must be na,nb,nc\n");
                exit(1);
            }
        }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.pdb.empty()) {
        fprintf(stderr, "Usage: sfcalc_gpu_collapse <input.pdb> [key=val ...]\n");
        return 1;
    }

    double dmin  = args.dmin;
    double rate  = args.rate;
    double bmax  = args.bmax;
    double noise = args.noise;
    int na = args.na, nb = args.nb, nc = args.nc;

    // Resolve space group
    const gemmi::SpaceGroup* sg = gemmi::find_spacegroup_by_name(args.sg_name);
    if (!sg) { fprintf(stderr, "ERROR: unknown space group '%s'\n", args.sg_name.c_str()); return 1; }

    // -----------------------------------------------------------------------
    // 1. Load PDB
    // -----------------------------------------------------------------------
    fprintf(stderr, "Reading %s ...\n", args.pdb.c_str());
    gemmi::Structure st = gemmi::read_structure_file(args.pdb);
    gemmi::UnitCell& cell = st.cell;
    double ax = cell.a, ay = cell.b, az = cell.c;
    fprintf(stderr, "  Supercell: %.3f x %.3f x %.3f  angles: %.2f %.2f %.2f\n",
            ax, ay, az, cell.alpha, cell.beta, cell.gamma);

    // Primitive cell
    gemmi::UnitCell prim_cell(ax/na, ay/nb, az/nc, cell.alpha, cell.beta, cell.gamma);
    fprintf(stderr, "  super_mult: %d x %d x %d\n", na, nb, nc);
    fprintf(stderr, "  Primitive cell: %.3f x %.3f x %.3f\n",
            prim_cell.a, prim_cell.b, prim_cell.c);
    fprintf(stderr, "  Space group: %s\n", sg->xhm());

    // Extract atoms
    std::vector<float> xs, ys, zs, Bs;
    std::vector<int>   els;
    for (auto& model : st.models)
        for (auto& chain : model.chains)
            for (auto& res : chain.residues)
                for (auto& atom : res.atoms) {
                    xs.push_back((float)atom.pos.x);
                    ys.push_back((float)atom.pos.y);
                    zs.push_back((float)atom.pos.z);
                    Bs.push_back(std::max(0.f, atom.b_iso));
                    els.push_back(elem_idx(atom.element.uname()));
                }

    int natoms = (int)xs.size();
    float B_min = *std::min_element(Bs.begin(), Bs.end());
    float B_max_val = *std::max_element(Bs.begin(), Bs.end());
    fprintf(stderr, "  Atoms: %d  B: %.2f .. %.2f\n", natoms, B_min, B_max_val);

    // bmax filter
    if (bmax > 0) {
        std::vector<float> xs2, ys2, zs2, Bs2;
        std::vector<int>   els2;
        int nskip = 0;
        for (int i = 0; i < natoms; i++) {
            if (Bs[i] > (float)bmax) { nskip++; continue; }
            xs2.push_back(xs[i]); ys2.push_back(ys[i]); zs2.push_back(zs[i]);
            Bs2.push_back(Bs[i]); els2.push_back(els[i]);
        }
        if (nskip) fprintf(stderr, "  Skipping %d atoms with B > %.1f\n", nskip, bmax);
        xs = xs2; ys = ys2; zs = zs2; Bs = Bs2; els = els2;
        natoms = (int)xs.size();
    }

    // Auto-blur
    float b_add      = (float)((dmin * rate) * (dmin * rate) / (M_PI * M_PI));
    float sigma_min  = sqrtf(b_add / (4.f * (float)(M_PI*M_PI)));
    float pixel_fine_v = (float)(dmin / (2.0 * rate));
    fprintf(stderr, "  Auto-blur: b_add = %.4f A^2  (sigma_min = %.3f A, pixel = %.3f A)\n",
            b_add, sigma_min, pixel_fine_v);
    for (int i = 0; i < natoms; i++) Bs[i] += b_add;

    // -----------------------------------------------------------------------
    // 2. Multi-grid levels
    // -----------------------------------------------------------------------
    auto levels    = compute_levels(dmin, rate, ax, ay, az);
    int n_levels   = (int)levels.size();
    double pix_fine = dmin / (2.0 * rate);
    std::vector<int> atom_lev;
    assign_levels(Bs, pix_fine, noise, n_levels, atom_lev);

    int nx = levels[0].nx, ny = levels[0].ny, nz = levels[0].nz;
    int nx2 = nx/2 + 1;
    double V_cell = cell.volume;
    double w_px   = noise_wpx(noise);

    int last_occ = 0;
    for (int L = 0; L < n_levels; L++)
        for (int i = 0; i < natoms; i++)
            if (atom_lev[i] == L) { last_occ = L; break; }

    fprintf(stderr, "  Multi-grid levels (noise<=%.1f%%, w_px=%.3f, step=sqrt(2)):\n",
            noise*100, w_px);
    for (int L = 0; L <= last_occ; L++) {
        int n_L = 0;
        for (int i = 0; i < natoms; i++) if (atom_lev[i] == L) n_L++;
        double B_lo = b_threshold_for_level(L,   pix_fine, noise);
        double B_hi = (L < n_levels-1) ? b_threshold_for_level(L+1, pix_fine, noise)
                                       : 1e30;
        double vfrac = (double)(levels[L].nx * levels[L].ny * levels[L].nz)
                     / (double)(nx * ny * nz) * 100.0;
        fprintf(stderr, "    L%d: pixel=%.3fA  grid %dx%dx%d (%.1f%%)  "
                "B=[%.1f,%.1f)  %d atoms\n",
                L, levels[L].d/rate/2,
                levels[L].nx, levels[L].ny, levels[L].nz,
                vfrac, B_lo, B_hi, n_L);
    }

    // -----------------------------------------------------------------------
    // 3. Load GPU library
    // -----------------------------------------------------------------------
    void* gpu_lib = dlopen(args.lib.c_str(), RTLD_NOW);
    if (!gpu_lib) {
        fprintf(stderr, "ERROR: cannot load %s: %s\n", args.lib.c_str(), dlerror());
        return 1;
    }
    SpreadAndFft_fn spread_and_fft =
        (SpreadAndFft_fn)dlsym(gpu_lib, "spread_and_fft");
    if (!spread_and_fft) {
        fprintf(stderr, "ERROR: spread_and_fft not found in %s\n", args.lib.c_str());
        return 1;
    }

    // -----------------------------------------------------------------------
    // 4. Spread + FFT each level → accumulate into fine-grid float32 arrays
    // -----------------------------------------------------------------------
    size_t acc_size = (size_t)nz * ny * nx2;
    std::vector<float> acc_re(acc_size, 0.f);
    std::vector<float> acc_im(acc_size, 0.f);

    for (int L = 0; L <= last_occ; L++) {
        // collect atoms for this level
        std::vector<float> lx, ly, lz, lB;
        std::vector<int>   lel;
        for (int i = 0; i < natoms; i++) {
            if (atom_lev[i] != L) continue;
            lx.push_back(xs[i]); ly.push_back(ys[i]); lz.push_back(zs[i]);
            lB.push_back(Bs[i]); lel.push_back(els[i]);
        }
        int n_L = (int)lx.size();
        if (n_L == 0) continue;

        int NX = levels[L].nx, NY = levels[L].ny, NZ = levels[L].nz;
        int NX2 = NX/2 + 1;
        size_t fft_n = (size_t)NX2 * NY * NZ;

        std::vector<float> Fr(fft_n), Fi(fft_n);
        std::vector<float> map_buf;
        float* map_ptr = nullptr;
        if (!args.outmap.empty() && L == 0) {
            map_buf.resize((size_t)NX * NY * NZ);
            map_ptr = map_buf.data();
        }

        int nkept = spread_and_fft(
            n_L,
            lx.data(), ly.data(), lz.data(), lB.data(), lel.data(),
            NX, NY, NZ,
            (float)ax, (float)ay, (float)az,
            (float)cell.alpha, (float)cell.beta, (float)cell.gamma,
            0.f,
            map_ptr, Fr.data(), Fi.data()
        );
        if (nkept < 0) {
            fprintf(stderr, "ERROR: spread_and_fft returned %d\n", nkept);
            return 1;
        }

        float norm_L = (float)(V_cell / ((double)NX * NY * NZ));
        for (size_t j = 0; j < fft_n; j++) {
            Fr[j] *=  norm_L;
            Fi[j] *= -norm_L;
        }

        if (L == 0) {
            for (size_t j = 0; j < acc_size; j++) {
                acc_re[j] += Fr[j];
                acc_im[j] += Fi[j];
            }
            if (map_ptr) {
                // store map_buf for later (write CCP4 map) — not yet implemented
                // TODO: write CCP4 map via gemmi
            }
        } else {
            add_to_fine(acc_re.data(), acc_im.data(), nx, ny, nz,
                        Fr.data(), Fi.data(), NX, NY, NZ);
        }
    }

    // -----------------------------------------------------------------------
    // 5. Apply blur correction: F(H) *= exp(+b_add * stol^2)
    // -----------------------------------------------------------------------
    {
        gemmi::UnitCell rc = cell.reciprocal();
        float cg = (float)cos(rc.gamma * M_PI / 180.0);
        float cb = (float)cos(rc.beta  * M_PI / 180.0);
        float ca = (float)cos(rc.alpha * M_PI / 180.0);
        float gs11 = (float)(rc.a*rc.a), gs22 = (float)(rc.b*rc.b), gs33 = (float)(rc.c*rc.c);
        float gs12 = (float)(rc.a*rc.b)*cg;
        float gs13 = (float)(rc.a*rc.c)*cb;
        float gs23 = (float)(rc.b*rc.c)*ca;
        apply_blur(acc_re.data(), acc_im.data(), nx, ny, nz,
                   b_add, gs11, gs22, gs33, gs12, gs13, gs23);
    }

    // -----------------------------------------------------------------------
    // 6. Extract supercell P1 ASU → write intensity MTZ
    // -----------------------------------------------------------------------
    fprintf(stderr, "Extracting supercell structure factors ...\n");

    // Reciprocal metric for supercell (for resolution filter)
    double inv_dmin2 = 1.0 / (dmin * dmin);
    // For orthogonal-like: inv_d2 = (H/ax)^2 + (K/ay)^2 + (L/az)^2
    // Use exact reciprocal metric
    gemmi::UnitCell rc_sc = cell.reciprocal();
    double rca = cos(rc_sc.alpha * M_PI / 180.0);
    double rcb = cos(rc_sc.beta  * M_PI / 180.0);
    double rcg = cos(rc_sc.gamma * M_PI / 180.0);
    double m11 = rc_sc.a*rc_sc.a, m22 = rc_sc.b*rc_sc.b, m33 = rc_sc.c*rc_sc.c;
    double m12 = rc_sc.a*rc_sc.b*rcg, m13 = rc_sc.a*rc_sc.c*rcb, m23 = rc_sc.b*rc_sc.c*rca;

    std::vector<HKL>  sc_hkl;
    std::vector<float> sc_I;

    for (int iz = 0; iz < nz; iz++) {
        int L = (iz <= nz/2) ? iz : iz - nz;
        for (int iy = 0; iy < ny; iy++) {
            int K = (iy <= ny/2) ? iy : iy - ny;
            for (int ix = 0; ix < nx2; ix++) {
                int H = ix;
                // P1 ASU: L>0, or L==0 K>0, or L==0 K==0 H>0
                bool in_asu = (L > 0) || (L == 0 && K > 0) || (L == 0 && K == 0 && H > 0);
                if (!in_asu) continue;
                double inv_d2 = H*H*m11 + K*K*m22 + L*L*m33
                              + 2.0*(H*K*m12 + H*L*m13 + K*L*m23);
                if (inv_d2 <= 0 || inv_d2 > inv_dmin2) continue;

                size_t idx = (size_t)iz * ny * nx2 + iy * nx2 + ix;
                float re = acc_re[idx], im = acc_im[idx];
                sc_hkl.push_back({H, K, L});
                sc_I.push_back(re*re + im*im);
            }
        }
    }

    if (!args.outI.empty())
        write_mtz_I(sc_hkl, sc_I, cell, args.outI);

    // -----------------------------------------------------------------------
    // 7. Collapse to primitive-cell ASU → write phased MTZ
    // -----------------------------------------------------------------------
    bool simple_p1 = (na == 1 && nb == 1 && nc == 1 && std::string(sg->hm) == "P 1");

    if (simple_p1) {
        int n = (int)sc_hkl.size();
        std::vector<float> amp(n), phi(n);
        for (int i = 0; i < n; i++) {
            size_t idx = (size_t)
                ((sc_hkl[i].l < 0 ? sc_hkl[i].l + nz : sc_hkl[i].l)) * ny * nx2
                + ((sc_hkl[i].k < 0 ? sc_hkl[i].k + ny : sc_hkl[i].k)) * nx2
                + sc_hkl[i].h;
            float re = acc_re[idx], im = acc_im[idx];
            amp[i] = sqrtf(re*re + im*im);
            phi[i] = atan2f(im, re) * (float)(180.0 / M_PI);
        }
        write_mtz(sc_hkl, amp, phi, cell, sg, args.outmtz);
    } else {
        fprintf(stderr, "Collapsing supercell to primitive-cell ASU (%s, %d operators) ...\n",
                sg->xhm(), (int)sg->operations().order());
        auto asu_refl = build_prim_asu(sg, prim_cell, dmin);
        fprintf(stderr, "  Primitive-cell ASU reflections: %d\n", (int)asu_refl.size());

        std::vector<float> F_re, F_im;
        collapse_to_prim_asu(acc_re.data(), acc_im.data(), nx, ny, nz,
                              na, nb, nc, sg, asu_refl, F_re, F_im);

        int n = (int)asu_refl.size();
        std::vector<float> amp(n), phi(n);
        for (int i = 0; i < n; i++) {
            amp[i] = sqrtf(F_re[i]*F_re[i] + F_im[i]*F_im[i]);
            phi[i] = atan2f(F_im[i], F_re[i]) * (float)(180.0 / M_PI);
        }
        write_mtz(asu_refl, amp, phi, prim_cell, sg, args.outmtz);
    }

    dlclose(gpu_lib);
    fprintf(stderr, "Done.\n");
    return 0;
}
