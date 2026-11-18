#include "fv/Switches.hpp"
#include "fv/CellFlux.hpp"
#include "fv/Profile.hpp"
#include "fv/ViscousBound.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#include "SchemesSimd.hpp"
#include "fv/MemoryBytes.hpp"
#include "fv/ParallelScan.hpp"

namespace fv {

double CellFluxEngine::fluxes(const CellMesh& m, const std::vector<int>& faces, const double* u_target) {
    const int nlist = static_cast<int>(faces.size());
    const EOS& eos = eos_;
    const Field& U = *us_;
    const Limiter lim = muscl_ ? muscl_->limiter() : Limiter::Minmod;
    if (F_.size() != m.faces.size()) F_.resize(m.faces.size());
    double smax_all = 0.0;
    auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };
    // non-reflecting ghost of leaf c across `side`: central differences along the other axis from the neighbour codes
    auto nrbc = [&](int c, Face side, const State& Ui) {
        BoundaryManager::NrbcTangent tg;
        const int axis = face_axis(side), l = m.level[c];
        const int nbase[3] = {m.nx, m.ny, m.nz};
        for (int t = 0; t < dim_; ++t) {   // the tangential axes, in the order of the uniform solver
            if (t == axis) continue;
            if (dim_ == 3 && l == 0 && nbase[t] == 1) continue;   // a single cell across: no tangential difference (as the uniform solver)
            const int cl = nbr_[static_cast<size_t>(ns_) * c + 2 * t], ch = nbr_[static_cast<size_t>(ns_) * c + 2 * t + 1];
            double span = 2.0;
            if (cl < 0) span -= 1.0;
            if (ch < 0) span -= 1.0;
            if (span <= 0.0) continue;
            tg.use[t] = true;
            tg.Wlo[t] = cl < 0 ? eos.cons_to_prim(Ui) : eos.cons_to_prim(cell_state(cl));
            tg.Whi[t] = ch < 0 ? eos.cons_to_prim(Ui) : eos.cons_to_prim(cell_state(ch));
            tg.span[t] = span;
        }
        const double hs[3] = {m.dx(l), m.dy(l), dim_ == 3 ? m.dz(l) : 0.0};
        return bc_.nonreflecting_ghost(side, Ui, tg, hs);
    };
    // boundary_order = 2: the faces on a physical side and the interior faces next to it are reconstructed through ghost layers (the
    // boundary condition's own extension of the leaf, as BoundaryManager::fill_ghost_slab builds them). A periodic or non-reflecting
    // side keeps the first-order treatment.
    const bool bo2 = muscl_ && bc_.params().boundary_order == 2;
    // WENO5-Z: a wide face (wide_) takes the six-cell stencil, every other face the four-cell one with the WENO3-Z formula
    const bool weno5 = muscl_ && muscl_->scheme() == Reconstruction::Weno5;
    const bool weno3 = muscl_ && (muscl_->scheme() == Reconstruction::Weno3 || weno5);
    const auto side_ok = [&](Face side) { return bc_.type(side) != BCType::Periodic && bc_.type(side) != BCType::NonReflecting; };
    const auto ghost_w = [&](Face side, const State& Ui) {   // primitive ghost state of a cell next to `side`
        return eos.cons_to_prim(bc_.ghost_state(side, Ui, u_target[static_cast<int>(side)]));
    };
    // the two face states from four consecutive primitive cells along the normal (the formulas of the interior faces)
    const auto recon4 = [&](const State& a, const State& b, const State& c, const State& d, State& wl, State& wr) {
        for (int q = 0; q < kNVar; ++q) {
            const double c0 = a[q], c1 = b[q], c2 = c[q], c3 = d[q];
            if (weno3) {
                wl[q] = muscl_->weno3_value(c0, c1, c2);
                wr[q] = muscl_->weno3_value(c3, c2, c1);
            } else {
                const double bb = c2 - c1;
                wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, bb);
                wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, bb);
            }
        }
    };
    // the two face states from six consecutive primitive cells along the normal (WENO5-Z, the loop of the uniform reconstruction)
    const auto recon6 = [&](const State& a, const State& b, const State& c, const State& d, const State& e, const State& g, State& wl,
                            State& wr) {
        for (int q = 0; q < kNVar; ++q) {
            wl[q] = muscl_->weno5_value(a[q], b[q], c[q], d[q], e[q]);
            wr[q] = muscl_->weno5_value(g[q], e[q], d[q], c[q], b[q]);
        }
    };
    const bool fb_on = fallback_on && static_cast<int>(fallback_flags.size()) == nleaf_;
    const auto fb_leaf = [&](int code) { return code >= 0 && code < nleaf_ && fallback_flags[static_cast<size_t>(code)] != 0; };
    const auto fluid_leaf_of = [&](int code, int level) {   // a leaf of `level` that is not in the body
        return code >= 0 && code < nleaf_ && m.level[static_cast<size_t>(code)] == level &&
               (solid_.empty() || !solid_[static_cast<size_t>(code)]);
    };
#pragma omp parallel if (nlist >= kOmpMinWork)
    {
        double local = 0.0;
        // The Riemann problems of the faces go to the solver in chunks (four at a time with AVX2); the results are the same as
        // one call per face.
        constexpr int kChunk = 256;
        State bUL[kChunk] = {}, bUR[kChunk] = {}, bF[kChunk];   // zeroed: the compiler cannot see that a flush reads only what was filled
        int bax[kChunk] = {}, bface[kChunk];
        double bsm[kChunk];
        int nb = 0;
        const auto flush = [&]() {
            riemann_.face_flux_batch(bUL, bUR, bax, nb, bF, bsm);
            for (int k = 0; k < nb; ++k) {
                F_[bface[k]] = bF[k];
                if (bsm[k] > local) local = bsm[k];
            }
            nb = 0;
        };
        // faces with an ordinary four-point MUSCL stencil: reconstruction and Riemann solver together
        int mst[4 * kChunk], max_[kChunk], mface[kChunk];
        State mF[kChunk];
        double msm[kChunk];
        int nm = 0;
        // faces with an ordinary WENO stencil (four cells with WENO3-Z, six with WENO5-Z on a wide face, none of them a ghost layer):
        // the face states by the vector kernels, then the Riemann batch above
        int wst4[4 * kChunk], wst6[6 * kChunk], wface4[kChunk], wface6[kChunk], wax4[kChunk], wax6[kChunk];
        int nw4 = 0, nw6 = 0;
        const auto flush_weno = [&](bool five) {
            int& nw = five ? nw6 : nw4;
            if (nw == 0) return;   // also: without a reconstruction muscl_ is null
            const int* wst = five ? wst6 : wst4;
            const int* wface = five ? wface6 : wface4;
            const int* wax = five ? wax6 : wax4;
            // the face states go straight into the Riemann batch (nw <= kChunk, so after a flush they fit)
            if (nb + nw > kChunk) flush();
            muscl_->weno_face_states(five, W_.data(), wst, nw, bUL + nb, bUR + nb);
            for (int k = 0; k < nw; ++k) {
                bax[nb + k] = wax[k];
                bface[nb + k] = wface[k];
            }
            nb += nw;
            if (nb == kChunk) flush();
            nw = 0;
        };
        const auto flush_muscl = [&]() {
            riemann_.muscl_face_flux_batch(lim, W_.data(), mst, max_, nm, mF, msm, dim_ == 2);
            for (int k = 0; k < nm; ++k) {
                F_[mface[k]] = mF[k];
                if (msm[k] > local) local = msm[k];
            }
            nm = 0;
        };
#pragma omp for schedule(static) nowait
        for (int fi = 0; fi < nlist; ++fi) {
            const int f = faces[static_cast<size_t>(fi)];
            const CellFace& face = m.faces[f];
            State UL, UR;
            // positivity_fallback: a face next to a flagged leaf is first order (the two leaf values), a boundary face too
            const bool fb_face = fb_on && (fb_leaf(face.left) || fb_leaf(face.right));
            if (face.kind == CellFaceKind::Boundary) {
                const bool ghost_right = face.left >= 0;
                const int c = ghost_right ? face.left : face.right;
                const State Ui = U[c];
                int n1 = -1;   // the leaf (or virtual cell) behind the boundary leaf: the second ghost layer mirrors it
                if (bo2 && side_ok(face.side) && !fb_face) {
                    n1 = nbr_[static_cast<size_t>(ns_) * c + (static_cast<int>(face.side) ^ 1)];
                    if (n1 == kOutside) throw std::logic_error("CellFluxEngine: a boundary face reads a leaf that is not in this rank's partial mesh");
                }
                if (n1 >= 0) {   // second-order boundary face: cells (n1, c, g0, g1) towards the boundary
                    const State Un = cell_state(n1);
                    const State Wc = eos.cons_to_prim(Ui), Wn = eos.cons_to_prim(Un);
                    const State g0 = ghost_w(face.side, Ui), g1 = ghost_w(face.side, Un);
                    State wl, wr;
                    // WENO5-Z: three ghost layers when the two cells behind the boundary leaf are fluid leaves of its level
                    // (cells n2, n1, c, g0, g1, g2 towards the boundary); else the four cells, as the faces that are not wide
                    int n2 = -1;
                    if (weno5 && fluid_leaf_of(c, face.level) && fluid_leaf_of(n1, face.level)) {
                        n2 = nbr_[static_cast<size_t>(ns_) * n1 + (static_cast<int>(face.side) ^ 1)];
                        if (n2 == kOutside) throw std::logic_error("CellFluxEngine: a boundary face reads a leaf that is not in this rank's partial mesh");
                        if (!fluid_leaf_of(n2, face.level)) n2 = -1;
                    }
                    if (n2 >= 0) {
                        const State Un2 = cell_state(n2);
                        const State Wn2 = eos.cons_to_prim(Un2), g2 = ghost_w(face.side, Un2);
                        if (ghost_right)
                            recon6(Wn2, Wn, Wc, g0, g1, g2, wl, wr);
                        else
                            recon6(g2, g1, g0, Wc, Wn, Wn2, wl, wr);
                    } else if (ghost_right) {
                        recon4(Wn, Wc, g0, g1, wl, wr);
                    } else {
                        recon4(g1, g0, Wc, Wn, wl, wr);
                    }
                    UL = eos.prim_to_cons(wl);
                    UR = eos.prim_to_cons(wr);
                } else {
                    const State Ug = bc_.type(face.side) == BCType::NonReflecting
                                         ? nrbc(c, face.side, Ui)
                                         : bc_.ghost_state(face.side, Ui, u_target[static_cast<int>(face.side)]);
                    UL = ghost_right ? Ui : Ug;
                    UR = ghost_right ? Ug : Ui;
                    if (muscl_) {
                        roundtrip(UL);
                        roundtrip(UR);
                    }
                }
            } else if (!muscl_ || fb_face) {
                UL = U[face.left];
                UR = U[face.right];
            } else {
                const std::array<int, 4>& s = stencil_[f];
                // a stencil entry that replicates a leaf at a physical side is replaced by the ghost layer with boundary_order = 2
                State gtmp0, gtmp3;
                const State *p0 = &W_[s[0]], *p3 = &W_[s[3]];
                bool wall = false;
                if (bo2) {
                    const Face back = static_cast<Face>(2 * face.axis), fwd = static_cast<Face>(2 * face.axis + 1);
                    if (s[0] == s[1] && nbr_[static_cast<size_t>(ns_) * s[1] + 2 * face.axis] == -1 && side_ok(back)) {
                        gtmp0 = ghost_w(back, U[s[1]]);
                        p0 = &gtmp0;
                        wall = true;
                    }
                    if (s[3] == s[2] && nbr_[static_cast<size_t>(ns_) * s[2] + 2 * face.axis + 1] == -1 && side_ok(fwd)) {
                        gtmp3 = ghost_w(fwd, U[s[2]]);
                        p3 = &gtmp3;
                        wall = true;
                    }
                }
                const State &w0 = *p0, &w1 = W_[s[1]], &w2 = W_[s[2]], &w3 = *p3;
                State wl, wr;
                if (weno5 && wide_[static_cast<size_t>(f)]) {
                    // six cells (far[0], s0, s1, s2, s3, far[1]); a replicated outer cell at a physical side is the next ghost
                    // layer with boundary_order = 2: the mirror of the second leaf from the side when s0 (s3) is the first layer,
                    // of the first leaf when the stencil ends on it
                    const std::array<int, 2>& fr = far_[static_cast<size_t>(f)];
                    State gf0, gf1;
                    const State *pf0 = &W_[fr[0]], *pf1 = &W_[fr[1]];
                    if (bo2) {
                        const Face back = static_cast<Face>(2 * face.axis), fwd = static_cast<Face>(2 * face.axis + 1);
                        if (p0 == &gtmp0) {
                            gf0 = ghost_w(back, U[s[2]]);
                            pf0 = &gf0;
                        } else if (fr[0] == s[0] && nbr_[static_cast<size_t>(ns_) * s[0] + 2 * face.axis] == -1 && side_ok(back)) {
                            gf0 = ghost_w(back, U[s[0]]);
                            pf0 = &gf0;
                        }
                        if (p3 == &gtmp3) {
                            gf1 = ghost_w(fwd, U[s[1]]);
                            pf1 = &gf1;
                        } else if (fr[1] == s[3] && nbr_[static_cast<size_t>(ns_) * s[3] + 2 * face.axis + 1] == -1 && side_ok(fwd)) {
                            gf1 = ghost_w(fwd, U[s[3]]);
                            pf1 = &gf1;
                        }
                    }
                    if (pf0 == &W_[fr[0]] && pf1 == &W_[fr[1]] && p0 == &W_[s[0]] && p3 == &W_[s[3]]) {
                        const int cells[6] = {fr[0], s[0], s[1], s[2], s[3], fr[1]};
                        for (int k = 0; k < 6; ++k) wst6[6 * nw6 + k] = cells[k];
                        wax6[nw6] = face.axis;
                        wface6[nw6] = f;
                        if (++nw6 == kChunk) flush_weno(true);
                        continue;
                    }
                    recon6(*pf0, w0, w1, w2, w3, *pf1, wl, wr);
                } else if (face.kind == CellFaceKind::CoarseFine) {
                    // the fine side is reconstructed against the (virtual) cell across the face, the coarse side is its cell state
                    const bool fine_left = m.level[face.left] > m.level[face.right];
                    State gcb, gcf;   // the coarse leaf's ghost layers where it touches a physical side
                    bool cgb = false, cgf = false;
                    if (bo2 && coarse_side_second_order) {
                        const int C = fine_left ? s[3] : s[0], ax = face.axis;
                        const Face back = static_cast<Face>(2 * ax), fwd = static_cast<Face>(2 * ax + 1);
                        if (nbr_raw_[static_cast<size_t>(ns_) * C + 2 * ax] == -1 && side_ok(back)) {
                            gcb = ghost_w(back, U[C]);
                            cgb = true;
                        }
                        if (nbr_raw_[static_cast<size_t>(ns_) * C + 2 * ax + 1] == -1 && side_ok(fwd)) {
                            gcf = ghost_w(fwd, U[C]);
                            cgf = true;
                        }
                    }
                    for (int q = 0; q < kNVar; ++q) {
                        // coarse leaf C of the face: its state and its neighbours across the two sides along the normal
                        const int C = fine_left ? s[3] : s[0];
                        const int ax = face.axis;
                        const int code_b = nbr_raw_[static_cast<size_t>(ns_) * C + 2 * ax];
                        const int code_f = nbr_raw_[static_cast<size_t>(ns_) * C + 2 * ax + 1];
                        const double wc = coarse_side_second_order ? W_[C][q] : 0.0;
                        const double wb = !coarse_side_second_order ? 0.0 : (code_b < 0 ? (cgb ? gcb[q] : wc) : W_[code_b][q]);
                        const double wf = !coarse_side_second_order ? 0.0 : (code_f < 0 ? (cgf ? gcf[q] : wc) : W_[code_f][q]);
                        if (fine_left) {
                            const double c0 = w0[q], c1 = w1[q], c2 = w2[q];
                            wl[q] = weno3 ? muscl_->weno3_value(c0, c1, c2) : c1 + 0.5 * apply_limiter(lim, c1 - c0, c2 - c1);
                            wr[q] = !coarse_side_second_order ? w3[q]
                                    : weno3 ? muscl_->weno3_value(wf, wc, wb) : wc - 0.5 * apply_limiter(lim, wf - wc, wc - wb);
                        } else {
                            const double c1 = w1[q], c2 = w2[q], c3 = w3[q];
                            wl[q] = !coarse_side_second_order ? w0[q]
                                    : weno3 ? muscl_->weno3_value(wb, wc, wf) : wc + 0.5 * apply_limiter(lim, wc - wb, wf - wc);
                            wr[q] = weno3 ? muscl_->weno3_value(c3, c2, c1) : c2 - 0.5 * apply_limiter(lim, c3 - c2, c2 - c1);
                        }
                    }
                } else if (weno3 && !wall) {   // WENO3-Z on four leaves: the vector kernel
                    for (int k = 0; k < 4; ++k) wst4[4 * nw4 + k] = s[k];
                    wax4[nw4] = face.axis;
                    wface4[nw4] = f;
                    if (++nw4 == kChunk) flush_weno(false);
                    continue;
                } else if (wall) {   // a face with a ghost layer: the same four cells, the scalar reconstruction
                    recon4(w0, w1, w2, w3, wl, wr);
                } else {
                    for (int k = 0; k < 4; ++k) mst[4 * nm + k] = s[k];
                    max_[nm] = face.axis;
                    mface[nm] = f;
                    if (++nm == kChunk) flush_muscl();
                    continue;
                }
                UL = eos.prim_to_cons(wl);
                UR = eos.prim_to_cons(wr);
            }
            bUL[nb] = UL;
            bUR[nb] = UR;
            bax[nb] = face.axis;
            bface[nb] = f;
            if (++nb == kChunk) flush();
        }
        flush_weno(false);
        flush_weno(true);
        flush();
        flush_muscl();
#pragma omp critical(fv_cellflux_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

void viscous_cell_quantities(const EOS& eos, const State& Ui, const State& n0, const State& n1, const State& n2, const State& n3,
                             double dx, double dy, const ViscousCellArrays& out, int c) {
    const State W = eos.cons_to_prim(Ui);
    double gx[3], gy[3];
    for (int q = 0; q < 3; ++q) {
        gx[q] = (n1[q] - n0[q]) / (2.0 * dx);
        gy[q] = (n3[q] - n2[q]) / (2.0 * dy);
    }
    const double rho = W[0], u = W[1], v = W[2];
    out.u[c] = u;
    out.v[c] = v;
    out.T[c] = W[3] / rho;
    out.dudx[c] = (gx[1] - u * gx[0]) / rho;
    out.dudy[c] = (gy[1] - u * gy[0]) / rho;
    out.dvdx[c] = (gx[2] - v * gx[0]) / rho;
    out.dvdy[c] = (gy[2] - v * gy[0]) / rho;
}

void viscous_cell_quantities_run(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dx_of_level,
                                 const double* dy_of_level, int c0, int n, const ViscousCellArrays& out) {
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        done = n & ~3;
        if (done > 0) {
            double* const o[7] = {out.u, out.v, out.T, out.dudx, out.dudy, out.dvdx, out.dvdy};
            detail::viscous_cells_simd(eos, U, nbr, level, dx_of_level, dy_of_level, c0, done, o);
        }
    }
#endif
    for (int c = c0 + done; c < c0 + n; ++c) {
        const int* nb = nbr + static_cast<size_t>(4) * c;
        viscous_cell_quantities(eos, U[c], U[nb[0]], U[nb[1]], U[nb[2]], U[nb[3]], dx_of_level[level[c]], dy_of_level[level[c]], out,
                                c);
    }
}

double viscous_cell_speed(const EOS& eos, double mu, double k_cond, const State& U, double dx, double dy) {
    const double gamma = eos.gamma;
    const double L = step_length_2d(dx, dy), inv_h2 = inv_h2_2d(dx, dy);
    const State W = eos.cons_to_prim(U);
    const double rho = W[0];
    const double nu_eff = viscous_nu_eff(mu, k_cond, gamma, rho);
    return viscous_speed_bound(std::max(std::abs(W[1]), std::abs(W[2])), eos.sound_speed(W[3], rho), L, inv_h2, nu_eff);
}

double viscous_speed_range(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dx_of_level,
                           const double* dy_of_level, int cb, int ce) {
    double s_max = 0.0;
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        done = (ce - cb) & ~3;
        if (done > 0) s_max = detail::viscous_speed_simd(eos, mu, k_cond, U, level, dx_of_level, dy_of_level, cb, done);
    }
#endif
    for (int c = cb + done; c < ce; ++c)
        s_max = std::max(s_max, viscous_cell_speed(eos, mu, k_cond, U[c], dx_of_level[level[c]], dy_of_level[level[c]]));
    return s_max;
}

namespace {

// Cell sizes per level (CellMesh::dx / dy) for the viscous kernels
struct LevelSizes {
    std::vector<double> dx, dy;
    explicit LevelSizes(const CellMesh& m) : dx(static_cast<size_t>(m.max_level) + 1), dy(dx.size()) {
        for (int l = 0; l <= m.max_level; ++l) {
            dx[static_cast<size_t>(l)] = m.dx(l);
            dy[static_cast<size_t>(l)] = m.dy(l);
        }
    }
};

#ifdef FV_SIMD
constexpr int kViscBlock = 256;   // list entries per block of the viscous loops (the unit of the OpenMP schedule)
#endif

}   // namespace

#ifdef FV_SIMD
void CellFluxEngine::cell_quantities(const CellMesh& m, const std::vector<int>& cells) {
    if (dim_ == 3) {
        cell_quantities_3d(m, cells);
        return;
    }
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(cells.size());
    const LevelSizes hs(m);
    const ViscousCellArrays out{su_.data(), sv_.data(), sT_.data(), dudx_.data(), dudy_.data(), dvdx_.data(), dvdy_.data()};
    const int nblock = (n + kViscBlock - 1) / kViscBlock;
    // Runs of consecutive leaves whose four neighbours are leaves go to the range kernel (four at a time with AVX2); a cell with a
    // virtual neighbour or a physical edge is done on its own.
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int b = 0; b < nblock; ++b) {
        const int kb = b * kViscBlock, ke = std::min(n, kb + kViscBlock);
        int run_c = 0, run_n = 0;
        for (int k = kb; k < ke; ++k) {
            const int c = cells[k];
            const int* code = &nbr_[static_cast<size_t>(4) * c];
            const auto leaf = [&](int s) { return static_cast<unsigned>(code[s]) < static_cast<unsigned>(nleaf_); };
            const bool leaves = leaf(0) && leaf(1) && leaf(2) && leaf(3);
            if (leaves && run_n > 0 && c == run_c + run_n) {
                ++run_n;
                continue;
            }
            if (run_n > 0) viscous_cell_quantities_run(eos, U.data(), nbr_.data(), m.level.data(), hs.dx.data(), hs.dy.data(), run_c,
                                                       run_n, out);
            run_n = 0;
            if (leaves) {
                run_c = c;
                run_n = 1;
                continue;
            }
            const int l = m.level[c];
            const State& Ui = U[c];
            State nb[4];   // left, right, bottom, top values (the diffusive ghost at a physical edge)
            for (int s = 0; s < 4; ++s) nb[s] = code[s] >= 0 ? cell_state(code[s]) : bc_.viscous_ghost(static_cast<Face>(s), Ui);
            viscous_cell_quantities(eos, Ui, nb[0], nb[1], nb[2], nb[3], m.dx(l), m.dy(l), out, c);
        }
        if (run_n > 0)
            viscous_cell_quantities_run(eos, U.data(), nbr_.data(), m.level.data(), hs.dx.data(), hs.dy.data(), run_c, run_n, out);
    }
}
#else
// Without the vector kernels (non-x86, FV_NO_SIMD): the original loop
void CellFluxEngine::cell_quantities(const CellMesh& m, const std::vector<int>& cells) {
    if (dim_ == 3) {
        cell_quantities_3d(m, cells);
        return;
    }
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(cells.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int k = 0; k < n; ++k) {
        const int c = cells[k];
        const int l = m.level[c];
        const State& Ui = U[c];
        const State W = eos.cons_to_prim(Ui);
        State nb[4];   // left, right, bottom, top values (the diffusive ghost at a physical edge)
        for (int s = 0; s < 4; ++s) {
            const int code = nbr_[static_cast<size_t>(4) * c + s];
            nb[s] = code >= 0 ? cell_state(code) : bc_.viscous_ghost(static_cast<Face>(s), Ui);
        }
        const double dx = m.dx(l), dy = m.dy(l);
        double gx[3], gy[3];
        for (int q = 0; q < 3; ++q) {
            gx[q] = (nb[1][q] - nb[0][q]) / (2.0 * dx);
            gy[q] = (nb[3][q] - nb[2][q]) / (2.0 * dy);
        }
        const double rho = W[0], u = W[1], v = W[2];
        su_[c] = u;
        sv_[c] = v;
        sT_[c] = W[3] / rho;
        dudx_[c] = (gx[1] - u * gx[0]) / rho;
        dudy_[c] = (gy[1] - u * gy[0]) / rho;
        dvdx_[c] = (gx[2] - v * gx[0]) / rho;
        dvdy_[c] = (gy[2] - v * gy[0]) / rho;
    }
}
#endif

void CellFluxEngine::viscous_fluxes(const CellMesh& m, const std::vector<int>& faces) {
    if (dim_ == 3) {
        viscous_fluxes_3d(m, faces);
        return;
    }
    const int nlist = static_cast<int>(faces.size());
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu();
    const double k_cond = eos_.transport_law().thermal_conductivity(mu, ns.prandtl());
    const Field& U = *us_;
    const bool bo2 = bc_.params().boundary_order == 2;
    if (Fv_.size() != m.faces.size()) Fv_.resize(m.faces.size());
#pragma omp parallel for schedule(static) if (nlist >= kOmpMinWork)
    for (int fi = 0; fi < nlist; ++fi) {
        const int f = faces[static_cast<size_t>(fi)];
        const CellFace& face = m.faces[f];
        const int ax = face.axis, lv = face.level;
        double uL, vL, TL, uR, vR, TR, h;
        double ca, cb;   // cross derivatives of u and v along the other axis
        if (face.kind == CellFaceKind::Boundary) {
            const bool ghost_left = face.left < 0;
            const int c = ghost_left ? face.right : face.left;
            const State g = eos_.cons_to_prim(bc_.viscous_ghost(face.side, U[c]));
            const double gu = g[1], gv = g[2], gT = g[3] / g[0];
            if (ghost_left) {
                uL = gu;
                vL = gv;
                TL = gT;
                uR = su_[c];
                vR = sv_[c];
                TR = sT_[c];
            } else {
                uL = su_[c];
                vL = sv_[c];
                TL = sT_[c];
                uR = gu;
                vR = gv;
                TR = gT;
            }
            h = ax == 0 ? m.dx(lv) : m.dy(lv);
            if (bo2) {
                // second-order boundary face: the cross derivatives are the tangential difference of the face values (ghost and
                // cell average) of the two neighbours along the wall (BoundaryManager::fix_frame_corners for the uniform mesh). The
                // neighbour across a physical edge of the tangential side (a corner) is the ghost of the leaf on that side, and its
                // ghost is the corner entry of the padded frame.
                const int tl = ax == 0 ? 2 : 0;   // tangential sides: bottom / top for an x-face, left / right for a y-face
                const auto tface = [&](int side_index, double& fu, double& fv) {
                    const int code = nbr_[static_cast<size_t>(4) * c + side_index];
                    if (code == kOutside) throw std::logic_error("CellFluxEngine: a boundary face reads a leaf that is not in this rank's partial mesh");
                    State in, gh;
                    if (code >= 0) {
                        in = cell_state(code);
                        gh = bc_.viscous_ghost(face.side, in);
                    } else {
                        // the corner ghost is always the left / right rule applied to the bottom / top ghost (not symmetric for a
                        // moving wall): an x-face has face.side = left / right, a y-face has it the other way round
                        const Face ts = static_cast<Face>(side_index);
                        in = bc_.viscous_ghost(ts, U[c]);
                        gh = ax == 0 ? bc_.viscous_ghost(face.side, in) : bc_.viscous_ghost(ts, bc_.viscous_ghost(face.side, U[c]));
                    }
                    const State Wg = eos_.cons_to_prim(gh), Wi = eos_.cons_to_prim(in);
                    fu = 0.5 * (Wg[1] + Wi[1]);
                    fv = 0.5 * (Wg[2] + Wi[2]);
                };
                double ulo, vlo, uhi, vhi;
                tface(tl, ulo, vlo);
                tface(tl + 1, uhi, vhi);
                const double h2 = 2.0 * (ax == 0 ? m.dy(lv) : m.dx(lv));
                ca = (uhi - ulo) / h2;
                cb = (vhi - vlo) / h2;
            } else {
                const double a = ax == 0 ? dudy_[c] : dudx_[c], b = ax == 0 ? dvdy_[c] : dvdx_[c];
                ca = 0.5 * (a + a);
                cb = 0.5 * (b + b);
            }
        } else {
            const int L = face.left, R = face.right;
            uL = su_[L];
            vL = sv_[L];
            TL = sT_[L];
            uR = su_[R];
            vR = sv_[R];
            TR = sT_[R];
            h = ax == 0 ? 0.5 * (m.dx(m.level[L]) + m.dx(m.level[R])) : 0.5 * (m.dy(m.level[L]) + m.dy(m.level[R]));
            ca = ax == 0 ? 0.5 * (dudy_[L] + dudy_[R]) : 0.5 * (dudx_[L] + dudx_[R]);
            cb = ax == 0 ? 0.5 * (dvdy_[L] + dvdy_[R]) : 0.5 * (dvdx_[L] + dvdx_[R]);
        }
        const double u_f = 0.5 * (uL + uR), v_f = 0.5 * (vL + vR);
        const double du_n = (uR - uL) / h, dv_n = (vR - vL) / h, dT_n = (TR - TL) / h;
        const double len = face.length;
        if (ax == 0) {
            const double tau_xx = mu * (kMuDiag * du_n - kMuOff * cb);
            const double tau_xy = mu * (ca + dv_n);
            const double q_x = -k_cond * dT_n;
            Fv_[f] = {0.0, tau_xx * len, tau_xy * len, (u_f * tau_xx + v_f * tau_xy - q_x) * len};
        } else {
            const double tau_yy = mu * (kMuDiag * dv_n - kMuOff * ca);
            const double tau_yx = mu * (du_n + cb);
            const double q_y = -k_cond * dT_n;
            Fv_[f] = {0.0, tau_yx * len, tau_yy * len, (u_f * tau_yx + v_f * tau_yy - q_y) * len};
        }
    }
}

#ifdef FV_SIMD
double CellFluxEngine::viscous_speed(const CellMesh& m, int cb, int ce) const {
    if (dim_ == 3) return viscous_speed_3d(m, cb, ce);
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu();
    const double k_cond = eos_.transport_law().thermal_conductivity(mu, ns.prandtl());
    const Field& U = *us_;
    const LevelSizes hs(m);
    const int nblock = (ce - cb + kViscBlock - 1) / kViscBlock;
    double s_max = 0.0;
#pragma omp parallel if (ce - cb >= kOmpMinWork)
    {
        double local = 0.0;
        // blocks of consecutive leaves, four at a time with AVX2: a maximum does not depend on the grouping (every speed is > 0 or
        // NaN, which std::max(local, s) ignores)
#pragma omp for schedule(static) nowait
        for (int b = 0; b < nblock; ++b) {
            const int c0 = cb + b * kViscBlock, c1 = std::min(ce, c0 + kViscBlock);
            local = std::max(local, viscous_speed_range(eos_, mu, k_cond, U.data(), m.level.data(), hs.dx.data(), hs.dy.data(), c0, c1));
        }
#pragma omp critical(fv_cellvisc_speed)
        s_max = std::max(s_max, local);
    }
    return s_max;
}
#else
double CellFluxEngine::viscous_speed(const CellMesh& m, int cb, int ce) const {
    if (dim_ == 3) return viscous_speed_3d(m, cb, ce);
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu(), gamma = eos_.gamma;
    const double k_cond = eos_.transport_law().thermal_conductivity(mu, ns.prandtl());
    const Field& U = *us_;
    double s_max = 0.0;
#pragma omp parallel if (ce - cb >= kOmpMinWork)
    {
        double local = 0.0;
#pragma omp for schedule(static) nowait
        for (int c = cb; c < ce; ++c) {
            const int l = m.level[c];
            const double dx = m.dx(l), dy = m.dy(l);
            const double L = step_length_2d(dx, dy), inv_h2 = inv_h2_2d(dx, dy);
            const State W = eos_.cons_to_prim(U[c]);
            const double rho = W[0];
            const double nu_eff = viscous_nu_eff(mu, k_cond, gamma, rho);
            const double s = viscous_speed_bound(std::max(std::abs(W[1]), std::abs(W[2])), eos_.sound_speed(W[3], rho), L, inv_h2, nu_eff);
            local = std::max(local, s);
        }
#pragma omp critical(fv_cellvisc_speed)
        s_max = std::max(s_max, local);
    }
    return s_max;
}
#endif

// --- 3-D viscous terms: the expressions of NavierStokes::viscous_residual_3d (a level-0 octree gives its bits) ---

void CellFluxEngine::cell_quantities_3d(const CellMesh& m, const std::vector<int>& cells) {
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(cells.size());
    double* const G[3][3] = {{dudx_.data(), dudy_.data(), dudz_.data()},
                             {dvdx_.data(), dvdy_.data(), dvdz_.data()},
                             {dwdx_.data(), dwdy_.data(), dwdz_.data()}};
    double* const vel[3] = {su_.data(), sv_.data(), sw_.data()};
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int k = 0; k < n; ++k) {
        const int c = cells[static_cast<size_t>(k)];
        const int l = m.level[c];
        const State& Ui = U[c];
        const State W = eos.cons_to_prim(Ui);
        State nb[6];   // the six neighbours (the diffusive ghost at a physical edge): the padded state of the uniform solver
        for (int s = 0; s < 6; ++s) {
            const int code = nbr_[static_cast<size_t>(6) * c + s];
            nb[s] = code >= 0 ? cell_state(code) : bc_.viscous_ghost_3d(static_cast<Face>(s), Ui);
        }
        const double h[3] = {m.dx(l), m.dy(l), m.dz(l)};
        State grad[3];   // central differences of the conservative state (compute_gradients_3d)
        for (int b = 0; b < 3; ++b)
            for (int q = 0; q < kNVar; ++q) grad[b][q] = (nb[2 * b + 1][q] - nb[2 * b][q]) / (2.0 * h[b]);
        const double rho = W[0];
        sT_[c] = W[3] / rho;
        for (int a = 0; a < 3; ++a) {
            const double ua = W[kMom[a]];
            vel[a][c] = ua;
            for (int b = 0; b < 3; ++b) G[a][b][c] = (grad[b][kMom[a]] - ua * grad[b][0]) / rho;
        }
    }
}

void CellFluxEngine::viscous_fluxes_3d(const CellMesh& m, const std::vector<int>& faces) {
    const int nlist = static_cast<int>(faces.size());
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu();
    const double k_cond = eos_.transport_law().thermal_conductivity(mu, ns.prandtl());
    const Field& U = *us_;
    const bool bo2 = bc_.params().boundary_order == 2;
    if (Fv_.size() != m.faces.size()) Fv_.resize(m.faces.size());
    const double* const G[3][3] = {{dudx_.data(), dudy_.data(), dudz_.data()},
                                   {dvdx_.data(), dvdy_.data(), dvdz_.data()},
                                   {dwdx_.data(), dwdy_.data(), dwdz_.data()}};
    const double* const vel[3] = {su_.data(), sv_.data(), sw_.data()};
    // the cross terms (a, b) = du_a / dx_b of a face normal to axis d, in the order of the uniform solver (CrossGhosts order)
    static const int kPairs[3][4][2] = {{{0, 1}, {1, 1}, {2, 2}, {0, 2}}, {{0, 0}, {1, 0}, {2, 2}, {1, 2}}, {{0, 0}, {1, 1}, {2, 0}, {2, 1}}};
    const auto hsz = [&](int axis, int l) { return axis == 0 ? m.dx(l) : (axis == 1 ? m.dy(l) : m.dz(l)); };
#pragma omp parallel for schedule(static) if (nlist >= kOmpMinWork)
    for (int fi = 0; fi < nlist; ++fi) {
        const int f = faces[static_cast<size_t>(fi)];
        const CellFace& face = m.faces[f];
        const int d = face.axis, lv = face.level;
        double vL[3], vR[3], TL, TR, h;
        double Gc[3][3] = {};
        if (face.kind == CellFaceKind::Boundary) {
            const bool ghost_left = face.left < 0;
            const int c = ghost_left ? face.right : face.left;
            const State g = eos_.cons_to_prim(bc_.viscous_ghost_3d(face.side, U[c]));
            const double gT = g[3] / g[0];
            for (int a = 0; a < 3; ++a) {
                (ghost_left ? vL : vR)[a] = g[kMom[a]];
                (ghost_left ? vR : vL)[a] = vel[a][c];
            }
            TL = ghost_left ? gT : sT_[c];
            TR = ghost_left ? sT_[c] : gT;
            h = hsz(d, lv);
            if (bo2) {
                // second-order boundary face: d u_a / d x_b is the tangential difference of the face values (ghost and cell average)
                // of the two neighbours along b. Across a physical edge along b the neighbour is the b-ghost of the leaf and its
                // d-ghost is the edge entry of the padded state (BoundaryManager::fix_padded_corners: the rule of the lower axis
                // applied to the face ghost of the other one).
                double fval[6][3] = {};
                for (int b = 0; b < 3; ++b) {
                    if (b == d) continue;
                    for (int hi = 0; hi < 2; ++hi) {
                        const int si = 2 * b + hi;
                        const int code = nbr_[static_cast<size_t>(6) * c + si];
                        if (code == kOutside) throw std::logic_error("CellFluxEngine: a boundary face reads a leaf that is not in this rank's partial mesh");
                        State in, gh;
                        if (code >= 0) {
                            in = cell_state(code);
                            gh = bc_.viscous_ghost_3d(face.side, in);
                        } else {
                            const Face ts = static_cast<Face>(si);
                            in = bc_.viscous_ghost_3d(ts, U[c]);
                            gh = d < b ? bc_.viscous_ghost_3d(face.side, in) : bc_.viscous_ghost_3d(ts, bc_.viscous_ghost_3d(face.side, U[c]));
                        }
                        const State Wg = eos_.cons_to_prim(gh), Wi = eos_.cons_to_prim(in);
                        for (int a = 0; a < 3; ++a) fval[si][a] = 0.5 * (Wg[kMom[a]] + Wi[kMom[a]]);
                    }
                }
                for (const auto& pr : kPairs[d]) {
                    const int a = pr[0], b = pr[1];
                    Gc[a][b] = (fval[2 * b + 1][a] - fval[2 * b][a]) / (2.0 * hsz(b, lv));
                }
            } else {
                for (const auto& pr : kPairs[d]) {
                    const double x = G[pr[0]][pr[1]][c];
                    Gc[pr[0]][pr[1]] = 0.5 * (x + x);
                }
            }
        } else {
            const int L = face.left, R = face.right;
            for (int a = 0; a < 3; ++a) {
                vL[a] = vel[a][L];
                vR[a] = vel[a][R];
            }
            TL = sT_[L];
            TR = sT_[R];
            h = 0.5 * (hsz(d, m.level[L]) + hsz(d, m.level[R]));
            for (const auto& pr : kPairs[d]) Gc[pr[0]][pr[1]] = 0.5 * (G[pr[0]][pr[1]][L] + G[pr[0]][pr[1]][R]);
        }
        double Gn[3], vf[3];
        for (int a = 0; a < 3; ++a) {
            Gn[a] = (vR[a] - vL[a]) / h;
            vf[a] = 0.5 * (vL[a] + vR[a]);
        }
        const double dT = (TR - TL) / h;
        double tau[3];
        for (int a = 0; a < 3; ++a) {
            if (a == d) {
                double others = 0.0;
                for (int o = 0; o < 3; ++o)
                    if (o != d) others += Gc[o][o];
                tau[a] = mu * (kMuDiag * Gn[d] - kMuOff * others);
            } else {
                tau[a] = mu * (Gn[a] + Gc[d][a]);   // du_a/dx_d + du_d/dx_a
            }
        }
        const double q_d = -k_cond * dT;
        const double area = face.length;
        Fv_[f] = {0.0, tau[0] * area, tau[1] * area, (vf[0] * tau[0] + vf[1] * tau[1] + vf[2] * tau[2] - q_d) * area, tau[2] * area};
    }
}

double CellFluxEngine::viscous_speed_3d(const CellMesh& m, int cb, int ce) const {
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu(), gamma = eos_.gamma;
    const double k_cond = eos_.transport_law().thermal_conductivity(mu, ns.prandtl());
    const Field& U = *us_;
    double s_max = 0.0;
#pragma omp parallel if (ce - cb >= kOmpMinWork)
    {
        double local = 0.0;
#pragma omp for schedule(static) nowait
        for (int c = cb; c < ce; ++c) {
            const int l = m.level[c];
            const double dx = m.dx(l), dy = m.dy(l), dz = m.dz(l);
            // the time-step length of the leaf, 2 / sum 1/h (Simulation::dt_length), and sum 1/h^2 (viscous_residual_3d)
            const double inv_h2 = inv_h2_3d(dx, dy, dz);
            const double L = step_length_3d(dx, dy, dz);
            const State W = eos_.cons_to_prim(U[c]);
            const double rho = W[0];
            const double nu_eff = viscous_nu_eff(mu, k_cond, gamma, rho);
            const double s = viscous_speed_bound(std::max({std::abs(W[1]), std::abs(W[2]), std::abs(W[4])}), eos_.sound_speed(W[3], rho), L, inv_h2, nu_eff);
            local = std::max(local, s);
        }
#pragma omp critical(fv_cellvisc_speed3)
        s_max = std::max(s_max, local);
    }
    return s_max;
}

void CellFluxEngine::gather(const CellMesh& m, int cb, int ce, bool replace, Field& R) {
    R.resize(static_cast<size_t>(ce - cb));
    const bool visc = phys_.ns != nullptr;
    const Field& U = *us_;
    if (phys_.source && phys_.source_unsteady) update_source(m, cb, ce);
    if (dim_ == 3) {   // convective, viscous, body force, sources (the 2-D order of the terms)
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
        for (int c = cb; c < ce; ++c) {
            // side integrals: sum of (face flux * face area), one face or four (summed in pairs, in slot order: a fixed order);
            // the viscous face fluxes are stored times the area already
            State G[6], H[6];
            for (int s = 0; s < 6; ++s) {
                const int b = m.side_ptr[6 * c + s], e = m.side_ptr[6 * c + s + 1];
                State part[4], vpart[4];
                for (int k = b; k < e; ++k) {
                    const int fi = m.side_face[k];
                    const CellFace& f = m.faces[fi];
                    const bool reg = replace && f.level > m.level[c];
                    const State& Fk = reg ? reg_[static_cast<size_t>(reg_slot_[static_cast<size_t>(fi)])] : F_[fi];
                    for (int q = 0; q < kNVar; ++q) part[k - b][q] = Fk[q] * f.length;
                    if (visc) vpart[k - b] = reg ? regv_[static_cast<size_t>(reg_slot_[static_cast<size_t>(fi)])] : Fv_[fi];
                }
                if (e - b == 1) {
                    G[s] = part[0];
                    if (visc) H[s] = vpart[0];
                } else {
                    for (int q = 0; q < kNVar; ++q) G[s][q] = (part[0][q] + part[1][q]) + (part[2][q] + part[3][q]);
                    if (visc)
                        for (int q = 0; q < kNVar; ++q) H[s][q] = (vpart[0][q] + vpart[1][q]) + (vpart[2][q] + vpart[3][q]);
                }
            }
            const double inv_vol = 1.0 / m.volume(c);   // the uniform solver's 1 / cell_volume()
            State& r = R[c - cb];
            for (int q = 0; q < kNVar; ++q)
                r[q] = (((0.0 - (G[1][q] - G[0][q])) - (G[3][q] - G[2][q])) - (G[5][q] - G[4][q])) * inv_vol;
            if (visc)   // NavierStokes::viscous_residual_3d, accumulated
                for (int q = 0; q < kNVar; ++q)
                    r[q] += (((0.0 + (H[1][q] - H[0][q])) + (H[3][q] - H[2][q])) + (H[5][q] - H[4][q])) * inv_vol;
            if (!phys_.rhs_sources && phys_.body) {
                const ConstantBodyForce& bf = *phys_.body;
                r[1] += U[c][0] * bf.gx;
                r[2] += U[c][0] * bf.gy;
                r[3] += U[c][1] * bf.gx + U[c][2] * bf.gy;
                if (bf.gz != 0.0) {
                    r[3] += U[c][4] * bf.gz;
                    r[4] += U[c][0] * bf.gz;
                }
            }
            if (!phys_.rhs_sources && phys_.source)
                for (int q = 0; q < kNVar; ++q) r[q] += src_[c][q];
        }
        if (phys_.rhs_sources) phys_.rhs_sources->apply_adaptive(m, U, R, cb, ce, stage_time, phys_.source ? &src_ : nullptr);
        return;
    }
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
    for (int c = cb; c < ce; ++c) {
        // side integrals: sum of (face flux * face length), one or two faces per side
        State G[4], H[4];
        for (int s = 0; s < 4; ++s) {
            const int b = m.side_ptr[4 * c + s], e = m.side_ptr[4 * c + s + 1];
            for (int k = b; k < e; ++k) {
                const int fi = m.side_face[k];
                const CellFace& f = m.faces[fi];
                // a finer face on the side of this leaf: its coarse side uses the registered time average
                const bool reg = replace && f.level > m.level[c];
                const State& Fk = reg ? reg_[static_cast<size_t>(reg_slot_[static_cast<size_t>(fi)])] : F_[fi];
                if (k == b)
                    for (int q = 0; q < kNVar; ++q) G[s][q] = Fk[q] * f.length;
                else
                    for (int q = 0; q < kNVar; ++q) G[s][q] += Fk[q] * f.length;
                if (visc) {
                    const State& Vk = reg ? regv_[static_cast<size_t>(reg_slot_[static_cast<size_t>(fi)])] : Fv_[fi];
                    if (k == b)
                        H[s] = Vk;
                    else
                        for (int q = 0; q < kNVar; ++q) H[s][q] += Vk[q];
                }
            }
        }
        const double inv_area = 1.0 / m.area(c);   // the same multiplication by 1 / area as the uniform solver's
        State& r = R[c - cb];
        for (int q = 0; q < kNVar; ++q) r[q] = ((0.0 - (G[1][q] - G[0][q])) - (G[3][q] - G[2][q])) * inv_area;
        if (visc)
            for (int q = 0; q < kNVar; ++q) r[q] += ((0.0 + (H[1][q] - H[0][q])) + (H[3][q] - H[2][q])) * inv_area;
        if (!phys_.rhs_sources && phys_.body) {
            const ConstantBodyForce& bf = *phys_.body;
            r[1] += U[c][0] * bf.gx;
            r[2] += U[c][0] * bf.gy;
            r[3] += U[c][1] * bf.gx + U[c][2] * bf.gy;
            if (bf.gz != 0.0) {
                r[3] += U[c][4] * bf.gz;
                r[4] += U[c][0] * bf.gz;
            }
        }
        if (!phys_.rhs_sources && phys_.source)
            for (int q = 0; q < kNVar; ++q) r[q] += src_[c][q];
    }
    if (phys_.rhs_sources)
        phys_.rhs_sources->apply_adaptive(m, U, R, cb, ce, stage_time, phys_.source ? &src_ : nullptr);
}

double CellFluxEngine::residual(const CellMesh& m, const Field& U, Field& R) {
    const int N = m.num_leaves(), nf = static_cast<int>(m.faces.size());
    if (N != nleaf_ || static_cast<int>(stencil_.size()) != nf)
        throw std::logic_error("CellFluxEngine::residual: build() was not called for this mesh");
    prepare(m, U, -1);
    double u_target[6];
    mass_flow_targets(m, u_target);
    double s = fluxes(m, all_faces_, u_target);
    if (phys_.ns) {
        std::vector<int> all(static_cast<size_t>(N));
        for (int c = 0; c < N; ++c) all[c] = c;
        cell_quantities(m, all);
        viscous_fluxes(m, all_faces_);
        s = std::max(s, viscous_speed(m, 0, N));
    }
    gather(m, 0, N, false, R);
    return s;
}

double CellFluxEngine::wave_speed(const CellMesh& m, const Field& U) {
    double s = 0.0;
    for (int l = 0; l <= m.max_level; ++l)
        if (m.level_begin[l + 1] > m.level_begin[l]) s = std::max(s, compute_level_fluxes(m, U, l));
    return s;
}

double CellFluxEngine::compute_level_fluxes(const CellMesh& m, const Field& U, int l) {
    if (m.num_leaves() != nleaf_ || static_cast<int>(stencil_.size()) != static_cast<int>(m.faces.size()))
        throw std::logic_error("CellFluxEngine: build() was not called for this mesh");
    ProfileScope ps_prep(Section::AmrPrepare);
    prepare(m, U, l);
    ps_prep.stop();
    double u_target[6];
    mass_flow_targets(m, u_target);
    ProfileScope ps_conv(Section::AmrConvective);
    double s = fluxes(m, lface_[static_cast<size_t>(l)], u_target);
    ps_conv.stop();
    if (phys_.ns) {
        ProfileScope ps_cells(Section::AmrViscousCells);
        cell_quantities(m, vcells_[static_cast<size_t>(l)]);
        ps_cells.stop();
        ProfileScope ps_faces(Section::AmrViscousFaces);
        viscous_fluxes(m, lface_[static_cast<size_t>(l)]);
        s = std::max(s, viscous_speed(m, lown_b_[static_cast<size_t>(l)], lown_e_[static_cast<size_t>(l)]));
    }
    return s;
}

void CellFluxEngine::gather_level(const CellMesh& m, int l, bool replace, Field& Rl) {
    gather(m, lown_b_[static_cast<size_t>(l)], lown_e_[static_cast<size_t>(l)], replace, Rl);
}

void CellFluxEngine::clear_registers(const CellMesh& /*m*/, int l) {
    for (int s = reg_level_begin_[static_cast<size_t>(l)]; s < reg_level_begin_[static_cast<size_t>(l) + 1]; ++s) {
        reg_[static_cast<size_t>(s)] = State{};
        regv_[static_cast<size_t>(s)] = State{};
    }
}

void CellFluxEngine::accumulate_registers(const CellMesh& m, int l, double w) {
    const std::vector<int>& faces = lface_[static_cast<size_t>(l)];
    const int nlist = static_cast<int>(faces.size());
    const bool visc = phys_.ns != nullptr;
#pragma omp parallel for schedule(static) if (nlist >= kOmpMinWork)
    for (int fi = 0; fi < nlist; ++fi) {
        const int f = faces[static_cast<size_t>(fi)];
        if (m.faces[f].kind == CellFaceKind::CoarseFine) {
            const size_t s = static_cast<size_t>(reg_slot_[static_cast<size_t>(f)]);
            for (int q = 0; q < kNVar; ++q) {
                reg_[s][q] += w * F_[f][q];
                if (visc) regv_[s][q] += w * Fv_[f][q];
            }
        }
    }
}

void CellFluxEngine::save_registers(int l, Field& conv, Field& visc) const {
    const auto b = static_cast<size_t>(reg_level_begin_[static_cast<size_t>(l)]), e = static_cast<size_t>(reg_level_begin_[static_cast<size_t>(l) + 1]);
    conv.assign(reg_.begin() + b, reg_.begin() + e);
    visc.assign(regv_.begin() + b, regv_.begin() + e);
}

void CellFluxEngine::restore_registers(int l, const Field& conv, const Field& visc) {
    const auto b = static_cast<size_t>(reg_level_begin_[static_cast<size_t>(l)]);
    std::copy(conv.begin(), conv.end(), reg_.begin() + b);
    std::copy(visc.begin(), visc.end(), regv_.begin() + b);
}

void CellFluxEngine::solid_reset(const CellMesh& m, Field& U) const {
    if (!has_body()) return;
    const int N = m.num_leaves();
    // ghost leaves from the fluid, the deeper solid leaves at rest at the reference state (owned leaves); before the wall is ready
    // there are no ghost leaves and every solid leaf is reset
    if (ghost_ready_) fill_ghost_leaves(U, U, owned_mask_.empty() ? nullptr : &owned_mask_);
    const State deep = eos_.prim_to_cons({ibm_rho_ref_, 0.0, 0.0, ibm_p_ref_, 0.0});
    for (int c = 0; c < N; ++c)
        if (solid_[c] && !(ghost_ready_ && ghost_leaf_mask_[static_cast<size_t>(c)]) && (owned_mask_.empty() || owned_mask_[c])) U[c] = deep;
}

}   // namespace fv
