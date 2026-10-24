#include "fv/Physics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <stdexcept>

namespace fv {

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

void compute_gradients(const Field& P, int nx, int ny, double dx, double dy, Field& gx, Field& gy) {
    const int W = nx + 2;
    gx.resize(static_cast<size_t>(nx) * ny);
    gy.resize(static_cast<size_t>(nx) * ny);
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int c = j * nx + i;
            for (int q = 0; q < kNVar; ++q) {
                gx[c][q] = (P[(j + 1) * W + i + 2][q] - P[(j + 1) * W + i][q]) / (2.0 * dx);
                gy[c][q] = (P[(j + 2) * W + i + 1][q] - P[j * W + i + 1][q]) / (2.0 * dy);
            }
        }
}

void compute_gradients_3d(const Field& P, const Mesh& m, Field& gx, Field& gy, Field& gz) {
    const int nx = m.nx, ny = m.ny, nz = m.nz, n = nx * ny * nz;
    const int PX = nx + 2, PY = ny + 2;
    auto at = [&](int ii, int jj, int kk) -> const State& { return P[(static_cast<size_t>(kk) * PY + jj) * PX + ii]; };
    gx.resize(n);
    gy.resize(n);
    gz.resize(n);
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int kj = 0; kj < ny * nz; ++kj) {
        const int j = kj % ny, k = kj / ny;
        for (int i = 0; i < nx; ++i) {
            const int c = m.cell(i, j, k);
            for (int q = 0; q < kNVar; ++q) {
                gx[c][q] = (at(i + 2, j + 1, k + 1)[q] - at(i, j + 1, k + 1)[q]) / (2.0 * m.dx);
                gy[c][q] = (at(i + 1, j + 2, k + 1)[q] - at(i + 1, j, k + 1)[q]) / (2.0 * m.dy);
                gz[c][q] = (at(i + 1, j + 1, k + 2)[q] - at(i + 1, j + 1, k)[q]) / (2.0 * m.dz);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Navier-Stokes viscous residual
// ---------------------------------------------------------------------------

double NavierStokes::viscous_residual(const Field& U, const Field& gX, const Field& gY, const Mesh& mesh,
                                      const Field& U_pad, Field& Rv, const CrossGhosts* cg) const {
    constexpr double MU_DIAG = 4.0 / 3.0;
    constexpr double MU_OFF = 2.0 / 3.0;

    const int nx = mesh.nx, ny = mesh.ny, n = nx * ny;
    const double dx = mesh.dx, dy = mesh.dy;
    const double gamma = eos_.gamma;
    const double k_cond = mu_ * gamma / (Pr_ * (gamma - 1.0));

    cons_to_prim(eos_, U, W_);
    cons_to_prim(eos_, U_pad, W_pad_);
    const Field& W = W_;
    const Field& Wp = W_pad_;
    const int PW = nx + 2;

    // Per-cell scalars: u, v, T, du/dx, du/dy, dv/dx, dv/dy
    scratch_.resize(static_cast<size_t>(7) * n);
    double* su = scratch_.data();
    double* sv = su + n;
    double* sT = sv + n;
    double* dudx = sT + n;
    double* dudy = dudx + n;
    double* dvdx = dudy + n;
    double* dvdy = dvdx + n;
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) {
        const double rho = W[c][0], u = W[c][1], v = W[c][2];
        su[c] = u;
        sv[c] = v;
        sT[c] = W[c][3] / rho;
        const double drx = gX[c][0], dry = gY[c][0];
        dudx[c] = (gX[c][1] - u * drx) / rho;
        dudy[c] = (gY[c][1] - u * dry) / rho;
        dvdx[c] = (gX[c][2] - v * drx) / rho;
        dvdy[c] = (gY[c][2] - v * dry) / rho;
    }

    // Ghost values from the padded state (wall BCs); temperature is adiabatic
    auto gU = [&](int jj, int ii) { return Wp[jj * PW + ii][1]; };
    auto gV = [&](int jj, int ii) { return Wp[jj * PW + ii][2]; };
    auto gT = [&](int jj, int ii) { return Wp[jj * PW + ii][3] / Wp[jj * PW + ii][0]; };

    Rv.assign(n, State{0, 0, 0, 0});

    // X-faces: F(j, i) for i = 0..nx; left cell i-1, right cell i
    auto xflux = [&](int j, int i) -> State {
        const bool lo = (i == 0), hi = (i == nx);
        const int cL = j * nx + (lo ? 0 : i - 1);
        const int cR = j * nx + (hi ? nx - 1 : i);
        const double uL = lo ? gU(j + 1, 0) : su[cL], uR = hi ? gU(j + 1, nx + 1) : su[cR];
        const double vL = lo ? gV(j + 1, 0) : sv[cL], vR = hi ? gV(j + 1, nx + 1) : sv[cR];
        const double TL = lo ? gT(j + 1, 0) : sT[cL], TR = hi ? gT(j + 1, nx + 1) : sT[cR];
        const double u_f = 0.5 * (uL + uR);
        const double v_f = 0.5 * (vL + vR);
        const double du_dx = (uR - uL) / dx;
        const double dv_dx = (vR - vL) / dx;
        const double dT_dx = (TR - TL) / dx;
        // Cross terms: neighbour values across MPI interfaces, the opposite edge cell on a
        // periodic axis, zero gradient at other physical boundaries
        const bool gl = lo && cg && cg->dir == 'x' && cg->lo_a, gr = hi && cg && cg->dir == 'x' && cg->hi_a;
        const bool px = cg && cg->periodic_x;
        const int xL = (lo && px) ? j * nx + nx - 1 : cL, xR = (hi && px) ? j * nx : cR;
        const double du_dy = 0.5 * ((gl ? cg->lo_a[j] : dudy[xL]) +
                                    (gr ? cg->hi_a[j] : dudy[xR]));
        const double dv_dy = 0.5 * ((gl ? cg->lo_b[j] : dvdy[xL]) +
                                    (gr ? cg->hi_b[j] : dvdy[xR]));
        const double tau_xx = mu_ * (MU_DIAG * du_dx - MU_OFF * dv_dy);
        const double tau_xy = mu_ * (du_dy + dv_dx);
        const double q_x = -k_cond * dT_dx;
        return {0.0, tau_xx * dy, tau_xy * dy, (u_f * tau_xx + v_f * tau_xy - q_x) * dy};
    };
    // Y-faces: G(j, i) for j = 0..ny; bottom cell j-1, top cell j
    auto yflux = [&](int j, int i) -> State {
        const bool lo = (j == 0), hi = (j == ny);
        const int cB = (lo ? 0 : j - 1) * nx + i;
        const int cT = (hi ? ny - 1 : j) * nx + i;
        const double uB = lo ? gU(0, i + 1) : su[cB], uT = hi ? gU(ny + 1, i + 1) : su[cT];
        const double vB = lo ? gV(0, i + 1) : sv[cB], vT = hi ? gV(ny + 1, i + 1) : sv[cT];
        const double TB = lo ? gT(0, i + 1) : sT[cB], TT = hi ? gT(ny + 1, i + 1) : sT[cT];
        const double u_f = 0.5 * (uB + uT);
        const double v_f = 0.5 * (vB + vT);
        const double dv_dy = (vT - vB) / dy;
        const double du_dy = (uT - uB) / dy;
        const double dT_dy = (TT - TB) / dy;
        const bool gb = lo && cg && cg->dir == 'y' && cg->lo_a, gt = hi && cg && cg->dir == 'y' && cg->hi_a;
        const bool py = cg && cg->periodic_y;
        const int yB = (lo && py) ? (ny - 1) * nx + i : cB, yT = (hi && py) ? i : cT;
        const double du_dx = 0.5 * ((gb ? cg->lo_a[i] : dudx[yB]) +
                                    (gt ? cg->hi_a[i] : dudx[yT]));
        const double dv_dx = 0.5 * ((gb ? cg->lo_b[i] : dvdx[yB]) +
                                    (gt ? cg->hi_b[i] : dvdx[yT]));
        const double tau_yy = mu_ * (MU_DIAG * dv_dy - MU_OFF * du_dx);
        const double tau_yx = mu_ * (du_dy + dv_dx);
        const double q_y = -k_cond * dT_dy;
        return {0.0, tau_yx * dx, tau_yy * dx, (u_f * tau_yx + v_f * tau_yy - q_y) * dx};
    };

    const double area = mesh.cell_area();
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int j = 0; j < ny; ++j) {
        State Fl = xflux(j, 0);
        for (int i = 0; i < nx; ++i) {
            const State Fr = xflux(j, i + 1);
            const State Gb = yflux(j, i);
            const State Gt = yflux(j + 1, i);
            State& r = Rv[j * nx + i];
            for (int q = 0; q < kNVar; ++q) r[q] = ((0.0 + (Fr[q] - Fl[q])) + (Gt[q] - Gb[q])) / area;
            Fl = Fr;
        }
    }

    // Speed that bounds the explicit time step. The convective eigenvalue (2 S / L) and the viscous one, from the
    // parabolic limit dt_visc = 1 / (2 max(nu, alpha_th) (1/dx^2 + 1/dy^2)) (0.25 h^2 / nu on square cells), share
    // one stability budget, and the viscous one counts twice (at CFL 0.5 both are at their marginal limit, so
    // their plain sum has no margin left: a manufactured solution with mu = 0.02 is unstable at CFL 0.5 with the
    // plain sum and stable up to 0.7 with this weight): dt (2 S / L + 2 / dt_visc) <= 2 CFL, so the solver divides
    // CFL L by S = (|u| + c) + L / dt_visc of the worst cell. A maximum over cells is the same on any partition
    // (MPI strips, adaptive-mesh levels), and it stays below the face wave speed as the viscosity goes to zero.
    const double L = std::min(dx, dy), inv_h2 = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    double s_max = 0.0;
#pragma omp parallel if (n >= kOmpMinWork)
    {
        double local = 0.0;   // thread-local maximum (OpenMP 2.0 has no max reduction); a max is order independent
#pragma omp for schedule(static) nowait
        for (int c = 0; c < n; ++c) {
            const double rho = W[c][0];
            const double nu_eff = std::max(mu_ / rho, k_cond / (rho * (gamma / (gamma - 1.0)))) + 1e-12;
            const double s = std::max(std::abs(W[c][1]), std::abs(W[c][2])) + eos_.sound_speed(W[c][3], rho) +
                             2.0 * L * inv_h2 * nu_eff;
            local = std::max(local, s);
        }
#pragma omp critical(fv_viscous_speed)
        s_max = std::max(s_max, local);
    }
    return s_max;
}

// 3-D viscous residual.  For a face normal to axis d the velocity-gradient tensor
// G[a][b] = du_a/dx_b uses compact differences across the face for b == d and
// face averages of the cell-centre gradients otherwise (the "cross" terms).
double NavierStokes::viscous_residual_3d(const Field& U, const Field& gX, const Field& gY, const Field& gZ,
                                         const Mesh& m, const Field& U_pad, Field& Rv, const CrossGhosts* cg,
                                         double length) const {
    constexpr double MU_DIAG = 4.0 / 3.0;
    constexpr double MU_OFF = 2.0 / 3.0;
    const int nx = m.nx, ny = m.ny, nz = m.nz, n = nx * ny * nz;
    const double h[3] = {m.dx, m.dy, m.dz};
    const double gamma = eos_.gamma;
    const double k_cond = mu_ * gamma / (Pr_ * (gamma - 1.0));
    const int PX = nx + 2, PY = ny + 2;

    cons_to_prim(eos_, U, W_);
    cons_to_prim(eos_, U_pad, W_pad_);
    const Field& W = W_;
    const Field& Wp = W_pad_;

    // Per cell: velocity (3), T, and G[a][b] (9)
    scratch_.resize(static_cast<size_t>(13) * n);
    double* vel[3] = {scratch_.data(), scratch_.data() + n, scratch_.data() + 2 * n};
    double* sT = scratch_.data() + 3 * n;
    double* G[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) G[a][b] = scratch_.data() + (4 + 3 * a + b) * static_cast<size_t>(n);
    const Field* grads[3] = {&gX, &gY, &gZ};
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int cc = 0; cc < n; ++cc) {
        const double rho = W[cc][0];
        sT[cc] = W[cc][3] / rho;
        for (int a = 0; a < 3; ++a) {
            const double ua = W[cc][kMom[a]];
            vel[a][cc] = ua;
            for (int b = 0; b < 3; ++b) G[a][b][cc] = ((*grads[b])[cc][kMom[a]] - ua * (*grads[b])[cc][0]) / rho;
        }
    }
    auto padded = [&](int ii, int jj, int kk) -> const State& {
        return Wp[(static_cast<size_t>(kk) * PY + jj) * PX + ii];
    };
    const bool periodic[3] = {cg && cg->periodic_x, cg && cg->periodic_y, cg && cg->periodic_z};

    // Face fluxes, one array per axis: F[d][face (i, j, k) with index t along d]
    const int nfx = (nx + 1) * ny * nz, nfy = nx * (ny + 1) * nz, nfz = nx * ny * (nz + 1);
    thread_local Field Fax[3];
    Fax[0].resize(nfx);
    Fax[1].resize(nfy);
    Fax[2].resize(nfz);
    for (int d = 0; d < 3; ++d) {
        const int ni = nx + (d == 0), nj = ny + (d == 1), nk = nz + (d == 2);
        const int n_d = d == 0 ? nx : (d == 1 ? ny : nz);
        const double area = m.face_area(d);
        Field& F = Fax[d];
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int kj = 0; kj < nj * nk; ++kj) {
            const int j = kj % nj, k = kj / nj;
            for (int i = 0; i < ni; ++i) {
                const int idx[3] = {i, j, k};
                const int t = idx[d];
                const bool lo = (t == 0), hi = (t == n_d);
                auto cell_at = [&](int tt) {
                    int c3[3] = {i, j, k};
                    c3[d] = tt;
                    return m.cell(c3[0], c3[1], c3[2]);
                };
                const int cL = cell_at(lo ? 0 : t - 1), cR = cell_at(hi ? n_d - 1 : t);
                // Values on each side (ghost from the padded state on boundary faces)
                double vL[3], vR[3], TL, TR;
                auto ghost = [&](bool high) -> const State& {
                    int p3[3] = {i + 1, j + 1, k + 1};
                    p3[d] = high ? n_d + 1 : 0;
                    return padded(p3[0], p3[1], p3[2]);
                };
                if (lo) {
                    const State& g = ghost(false);
                    for (int a = 0; a < 3; ++a) vL[a] = g[kMom[a]];
                    TL = g[3] / g[0];
                } else {
                    for (int a = 0; a < 3; ++a) vL[a] = vel[a][cL];
                    TL = sT[cL];
                }
                if (hi) {
                    const State& g = ghost(true);
                    for (int a = 0; a < 3; ++a) vR[a] = g[kMom[a]];
                    TR = g[3] / g[0];
                } else {
                    for (int a = 0; a < 3; ++a) vR[a] = vel[a][cR];
                    TR = sT[cR];
                }

                // Cross-term cells: MPI ghost value, periodic opposite cell, or own cell
                const int xL = (lo && periodic[d]) ? cell_at(n_d - 1) : cL;
                const int xR = (hi && periodic[d]) ? cell_at(0) : cR;
                const bool mpi_ax = cg && ((cg->dir == 'x' && d == 0) || (cg->dir == 'y' && d == 1));
                const int slab = d == 0 ? k * ny + j : k * nx + i;
                auto cross = [&](int a, int b, int which) {   // which: index into CrossGhosts a/b/c/d
                    double l = G[a][b][xL], r = G[a][b][xR];
                    if (mpi_ax) {
                        const double* gl[4] = {cg->lo_a, cg->lo_b, cg->lo_c, cg->lo_d};
                        const double* gr[4] = {cg->hi_a, cg->hi_b, cg->hi_c, cg->hi_d};
                        if (lo && gl[which]) l = gl[which][slab];
                        if (hi && gr[which]) r = gr[which][slab];
                    }
                    return 0.5 * (l + r);
                };
                // Normal derivatives (compact) and face-averaged velocities
                double Gn[3], vf[3];
                for (int a = 0; a < 3; ++a) {
                    Gn[a] = (vR[a] - vL[a]) / h[d];
                    vf[a] = 0.5 * (vL[a] + vR[a]);
                }
                const double dT = (TR - TL) / h[d];
                // Cross terms: G[d][a] (a != d) and G[o][o] (o != d); CrossGhosts order
                //   d = 0 (x-faces): a = du/dy, b = dv/dy, c = dw/dz, d = du/dz
                //   d = 1 (y-faces): a = du/dx, b = dv/dx, c = dw/dz, d = dv/dz
                double Gc[3][3] = {};
                if (d == 0) {
                    Gc[0][1] = cross(0, 1, 0);
                    Gc[1][1] = cross(1, 1, 1);
                    Gc[2][2] = cross(2, 2, 2);
                    Gc[0][2] = cross(0, 2, 3);
                } else if (d == 1) {
                    Gc[0][0] = cross(0, 0, 0);
                    Gc[1][0] = cross(1, 0, 1);
                    Gc[2][2] = cross(2, 2, 2);
                    Gc[1][2] = cross(1, 2, 3);
                } else {
                    Gc[0][0] = cross(0, 0, 0);
                    Gc[1][1] = cross(1, 1, 1);
                    Gc[2][0] = cross(2, 0, 2);
                    Gc[2][1] = cross(2, 1, 3);
                }
                double tau[3];
                for (int a = 0; a < 3; ++a) {
                    if (a == d) {
                        double others = 0.0;
                        for (int o = 0; o < 3; ++o)
                            if (o != d) others += Gc[o][o];
                        tau[a] = mu_ * (MU_DIAG * Gn[d] - MU_OFF * others);
                    } else {
                        tau[a] = mu_ * (Gn[a] + Gc[d][a]);   // du_a/dx_d + du_d/dx_a
                    }
                }
                const double q_d = -k_cond * dT;
                const int fi = d == 0 ? (k * ny + j) * (nx + 1) + i
                                      : (d == 1 ? (k * (ny + 1) + j) * nx + i : (k * ny + j) * nx + i);
                F[fi] = {0.0, tau[0] * area, tau[1] * area,
                         (vf[0] * tau[0] + vf[1] * tau[1] + vf[2] * tau[2] - q_d) * area, tau[2] * area};
            }
        }
    }

    Rv.assign(n, State{});
    const double vol = m.cell_volume();
    // Bind the thread_local arrays here: inside the parallel region each thread would see its own (empty) copy
    const Field& F0 = Fax[0];
    const Field& F1 = Fax[1];
    const Field& F2 = Fax[2];
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int kj = 0; kj < ny * nz; ++kj) {
        const int j = kj % ny, k = kj / ny;
        for (int i = 0; i < nx; ++i) {
            const State& fl = F0[(k * ny + j) * (nx + 1) + i];
            const State& fr = F0[(k * ny + j) * (nx + 1) + i + 1];
            const State& fb = F1[(k * (ny + 1) + j) * nx + i];
            const State& ft = F1[(k * (ny + 1) + j + 1) * nx + i];
            const State& ff = F2[(k * ny + j) * nx + i];
            const State& fk = F2[((k + 1) * ny + j) * nx + i];
            State& r = Rv[m.cell(i, j, k)];
            for (int q = 0; q < kNVar; ++q)
                r[q] = (((0.0 + (fr[q] - fl[q])) + (ft[q] - fb[q])) + (fk[q] - ff[q])) / vol;
        }
    }

    // Speed that bounds the explicit time step, as in the 2-D version. The parabolic limit of the 3-D heat
    // equation is dt_visc = 1 / (2 nu sum 1/h^2) (h^2/6nu on cubes; the 2-D 0.25 min(h)^2 / nu is the same
    // bound for square cells) and the convective length is L = 2 / sum(1/h), so L / dt_visc = 2 L nu sum 1/h^2.
    const double inv_h2 = 1.0 / (m.dx * m.dx) + 1.0 / (m.dy * m.dy) + 1.0 / (m.dz * m.dz);
    const double L = length > 0.0 ? length : 2.0 / (1.0 / m.dx + 1.0 / m.dy + 1.0 / m.dz);
    double s_max = 0.0;
#pragma omp parallel if (n >= kOmpMinWork)
    {
        double local = 0.0;
#pragma omp for schedule(static) nowait
        for (int cc = 0; cc < n; ++cc) {
            const double rho = W[cc][0];
            const double nu_eff = std::max(mu_ / rho, k_cond / (rho * (gamma / (gamma - 1.0)))) + 1e-12;
            const double s = std::max({std::abs(W[cc][1]), std::abs(W[cc][2]), std::abs(W[cc][4])}) +
                             eos_.sound_speed(W[cc][3], rho) + 2.0 * L * inv_h2 * nu_eff;
            local = std::max(local, s);
        }
#pragma omp critical(fv_viscous_speed)
        s_max = std::max(s_max, local);
    }
    return s_max;
}

// ---------------------------------------------------------------------------
// Body force
// ---------------------------------------------------------------------------

void ConstantBodyForce::add_to(const Field& U, Field& R) const {
    const int n = static_cast<int>(U.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) {
        R[c][1] += U[c][0] * gx;
        R[c][2] += U[c][0] * gy;
        R[c][3] += U[c][1] * gx + U[c][2] * gy;
        if (gz != 0.0) {
            R[c][3] += U[c][4] * gz;
            R[c][4] += U[c][0] * gz;
        }
    }
}

// ---------------------------------------------------------------------------
// Immersed boundary
// ---------------------------------------------------------------------------

ImmersedBoundary::ImmersedBoundary(const Mesh& mesh, const std::string& shape, double radius, double cx, double cy,
                                   double rho_ref, double p_ref, double cz)
    : nx_(mesh.nx), ny_(mesh.ny), nz_(mesh.nz), rho_ref_(rho_ref), p_ref_(p_ref),
      mask_(static_cast<size_t>(mesh.num_cells()), 0) {
    std::string s = shape;
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (s != "cylinder" && s != "sphere")
        throw std::invalid_argument("Unknown IBM shape: '" + shape + "' (use Cylinder or Sphere)");
    const bool sphere = mesh.is3d() && s == "sphere";
    shape_ = sphere ? Shape::Ball : Shape::Circle;
    cx_ = cx;
    cy_ = cy;
    cz_ = cz;
    r_ = radius;
    for (int k = 0; k < nz_; ++k)
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const double dxc = mesh.xc(i) - cx, dyc = mesh.yc(j) - cy;
                const double r2 =
                    sphere ? dxc * dxc + dyc * dyc + (mesh.zc(k) - cz) * (mesh.zc(k) - cz) : dxc * dxc + dyc * dyc;
                mask_[mesh.cell(i, j, k)] = std::sqrt(r2) <= radius;
            }
}

ImmersedBoundary::ImmersedBoundary(const Mesh& mesh, const Polygon2D& section, double rho_ref, double p_ref)
    : nx_(mesh.nx), ny_(mesh.ny), nz_(mesh.nz), rho_ref_(rho_ref), p_ref_(p_ref),
      mask_(static_cast<size_t>(mesh.num_cells()), 0) {
    shape_ = Shape::Polygon;
    poly_ = section;
    double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;   // bounding box: skip the far cells
    for (const auto& p : section.pts) {
        x0 = std::min(x0, p[0]);
        x1 = std::max(x1, p[0]);
        y0 = std::min(y0, p[1]);
        y1 = std::max(y1, p[1]);
    }
    for (int j = 0; j < ny_; ++j)
        for (int i = 0; i < nx_; ++i) {
            const double x = mesh.xc(i), y = mesh.yc(j);
            if (x < x0 || x > x1 || y < y0 || y > y1 || !section.contains(x, y)) continue;
            for (int k = 0; k < nz_; ++k) mask_[mesh.cell(i, j, k)] = 1;
        }
}

int ImmersedBoundary::solid_count() const {
    int n = 0;
    for (char m : mask_) n += m ? 1 : 0;
    return n;
}

void ImmersedBoundary::apply_forcing(Field& U, const EOS& eos) const {
    if (ghost_on_) {
        fill_ghosts_part([](int c) { return c; }, [&U](int c) -> const State& { return U[c]; }, U, eos, false);
        return;
    }
    const int n = static_cast<int>(U.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) {
        State W = eos.cons_to_prim(U[c]);
        if (mask_[c]) {
            W[1] = 0.0;
            W[2] = 0.0;
            W[4] = 0.0;
        }
        U[c] = eos.prim_to_cons(W);
    }
}

void ImmersedBoundary::apply_solid_reset(Field& U, const EOS& eos, const SolidResetHalo* h) const {
    if (ghost_on_) {   // ghost states from the fluid, the deeper solid cells at rest at the reference state
        fill_ghosts_part([](int c) { return c; }, [&U](int c) -> const State& { return U[c]; }, U, eos, true);
        return;
    }
    const int nx = nx_, ny = ny_, nz = nz_, n = nx * ny * nz;
    auto pressure = [&](const State& c) {
        return (eos.gamma - 1.0) * (c[3] - 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0]);
    };
    std::vector<double> rho(n), p(n);
    for (int c = 0; c < n; ++c) {
        rho[c] = U[c][0];
        p[c] = pressure(U[c]);
    }
    std::vector<double> new_rho = rho, new_p = p;
    std::vector<char> fringe(n, 0);
    auto id = [&](int k, int j, int i) { return (k * ny + j) * nx + i; };
    auto solid = [&](int k, int j, int i) { return mask_[id(k, j, i)] != 0; };

    // Neumann condition: copy rho/p from the adjacent fluid cell into the fringe
    // layer.  Passes run right, left, top, bottom (then back, front in 3-D; later
    // passes win), each reading the original rho/p.
    auto pull = [&](int c, double rho_n, double p_n) {
        new_rho[c] = rho_n;
        new_p[c] = p_n;
        fringe[c] = 1;
    };
    auto pull_in = [&](int k, int j, int i, int kn, int jn, int in) {
        if (solid(k, j, i) && !solid(kn, jn, in)) pull(id(k, j, i), rho[id(kn, jn, in)], p[id(kn, jn, in)]);
    };
    // Neighbour across an MPI interface (s-th cell of the halo slab)
    auto pull_halo = [&](int k, int j, int i, const Field* strip, const std::vector<char>* m, int s) {
        if (strip && m && solid(k, j, i) && !(*m)[s]) pull(id(k, j, i), (*strip)[s][0], pressure((*strip)[s]));
    };
    const bool hx = h && h->dir == 'x', hy = h && h->dir == 'y';

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i + 1 < nx; ++i) pull_in(k, j, i, k, j, i + 1);
            if (hx) pull_halo(k, j, nx - 1, h->hi, h->mask_hi, k * ny + j);
        }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) pull_in(k, j, i, k, j, i - 1);
            if (hx) pull_halo(k, j, 0, h->lo, h->mask_lo, k * ny + j);
        }
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j + 1 < ny; ++j)
            for (int i = 0; i < nx; ++i) pull_in(k, j, i, k, j + 1, i);
        if (hy)
            for (int i = 0; i < nx; ++i) pull_halo(k, ny - 1, i, h->hi, h->mask_hi, k * nx + i);
    }
    for (int k = 0; k < nz; ++k) {
        for (int j = 1; j < ny; ++j)
            for (int i = 0; i < nx; ++i) pull_in(k, j, i, k, j - 1, i);
        if (hy)
            for (int i = 0; i < nx; ++i) pull_halo(k, 0, i, h->lo, h->mask_lo, k * nx + i);
    }
    for (int k = 0; k + 1 < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) pull_in(k, j, i, k + 1, j, i);
    for (int k = 1; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) pull_in(k, j, i, k - 1, j, i);

    for (int c = 0; c < n; ++c) {
        if (!mask_[c]) continue;
        if (!fringe[c]) {
            new_rho[c] = rho_ref_;
            new_p[c] = p_ref_;
        }   // deep solid
        U[c] = {new_rho[c], 0.0, 0.0, new_p[c] / (eos.gamma - 1.0), 0.0};
    }
}

// ---------------------------------------------------------------------------
// Ghost-cell (sharp-interface) wall
// ---------------------------------------------------------------------------

void ImmersedBoundary::closest_wall_point(double x, double y, double& wx, double& wy, double& nx, double& ny) const {
    if (shape_ == Shape::Circle) {
        const double dx = x - cx_, dy = y - cy_, d = std::sqrt(dx * dx + dy * dy);
        nx = d > 0.0 ? dx / d : 1.0;
        ny = d > 0.0 ? dy / d : 0.0;
        wx = cx_ + r_ * nx;
        wy = cy_ + r_ * ny;
        return;
    }
    if (shape_ != Shape::Polygon) throw std::logic_error("ImmersedBoundary: no 2-D geometry for the ghost-cell wall");
    const auto& pts = poly_.pts;
    const size_t n = pts.size();
    double best = 1e300;
    size_t seg = 0;
    for (size_t k = 0; k < n; ++k) {
        const auto &a = pts[k], &b = pts[(k + 1) % n];
        const double ex = b[0] - a[0], ey = b[1] - a[1], l2 = ex * ex + ey * ey;
        double t = l2 > 0.0 ? ((x - a[0]) * ex + (y - a[1]) * ey) / l2 : 0.0;
        t = std::min(1.0, std::max(0.0, t));
        const double px = a[0] + t * ex, py = a[1] + t * ey, d2 = (x - px) * (x - px) + (y - py) * (y - py);
        if (d2 < best) {
            best = d2;
            wx = px;
            wy = py;
            seg = k;
        }
    }
    const double d = std::sqrt(best);
    const bool inside = poly_.contains(x, y);
    if (d > 1e-12) {   // outward: from the point to the wall if it is inside, from the wall to the point if outside
        nx = (inside ? wx - x : x - wx) / d;
        ny = (inside ? wy - y : y - wy) / d;
        return;
    }
    const auto &a = pts[seg], &b = pts[(seg + 1) % n];   // on the wall: the segment normal, pointing out
    const double ex = b[0] - a[0], ey = b[1] - a[1], l = std::sqrt(ex * ex + ey * ey);
    nx = ey / l;
    ny = -ex / l;
    if (poly_.contains(wx + 1e-6 * l * nx, wy + 1e-6 * l * ny)) {
        nx = -nx;
        ny = -ny;
    }
}

State ImmersedBoundary::Probe::at(const CellState& in, const EOS& eos) const {
    State r{};
    for (int k = 0; k < 4; ++k) {
        if (w[k] == 0.0) continue;
        const State W = eos.cons_to_prim(in(cell[k]));
        for (int q = 0; q < kNVar; ++q) r[q] += w[k] * W[q];
    }
    return r;
}

bool ImmersedBoundary::make_probe(const Mesh& m, double x, double y, Probe& pr) const {
    // bilinear in the four cell centres around (x, y); solid cells are left out and the weights renormalised
    double fx = (x - m.x0) / m.dx - 0.5, fy = (y - m.y0) / m.dy - 0.5;
    int i0 = static_cast<int>(std::floor(fx)), j0 = static_cast<int>(std::floor(fy));
    i0 = std::min(std::max(i0, 0), m.nx - 2);
    j0 = std::min(std::max(j0, 0), m.ny - 2);
    const double tx = std::min(1.0, std::max(0.0, fx - i0)), ty = std::min(1.0, std::max(0.0, fy - j0));
    const int ci[4] = {i0, i0 + 1, i0, i0 + 1}, cj[4] = {j0, j0, j0 + 1, j0 + 1};
    const double w[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
    double sum = 0.0;
    for (int k = 0; k < 4; ++k) {
        pr.cell[k] = m.cell(ci[k], cj[k]);
        pr.w[k] = mask_[pr.cell[k]] ? 0.0 : w[k];
        sum += pr.w[k];
    }
    if (sum < 1e-6) return false;
    for (double& v : pr.w) v /= sum;
    return true;
}

void ImmersedBoundary::enable_ghost_cells(const Mesh& m, bool slip) {
    if (m.is3d() || shape_ == Shape::None || shape_ == Shape::Ball)
        throw std::invalid_argument("ibm_method = ghost: 2-D cylinder or airfoil bodies only (for now)");
    const int nx = m.nx, ny = m.ny;
    const double h = std::max(m.dx, m.dy);
    slip_ = slip;
    ghost_.clear();
    surface_.clear();
    ghost_mask_.assign(mask_.size(), 0);
    // the image (resp. probe) point lies at least 1.5 cells into the fluid, so its four interpolation cells are fluid cells
    // on a flat or convex wall; further out where they are not (concave corners, thin trailing edges)
    const auto probe_out = [&](double wx, double wy, double nxw, double nyw, double d0, double& d, Probe& pr) {
        for (d = std::max(d0, 1.5 * h); d <= 4.0 * h + 1e-12; d += 0.5 * h)
            if (make_probe(m, wx + d * nxw, wy + d * nyw, pr)) return true;
        return false;
    };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const int c = m.cell(i, j);
            if (!mask_[c]) continue;
            bool near = false;   // a fluid cell within two cells (MUSCL stencils, viscous gradients) reads this one
            for (int dj = -2; dj <= 2 && !near; ++dj)
                for (int di = -2; di <= 2 && !near; ++di) {
                    const int ii = i + di, jj = j + dj;
                    if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && !mask_[m.cell(ii, jj)]) near = true;
                }
            if (!near) continue;
            Ghost g;
            g.cell = c;
            double wx, wy;
            closest_wall_point(m.xc(i), m.yc(j), wx, wy, g.nx, g.ny);
            const double dg = std::sqrt((m.xc(i) - wx) * (m.xc(i) - wx) + (m.yc(j) - wy) * (m.yc(j) - wy));
            double di;
            if (!probe_out(wx, wy, g.nx, g.ny, dg, di, g.image))
                throw std::runtime_error("ibm_method = ghost: no fluid cells around the image point of cell (" + std::to_string(i) +
                                         ", " + std::to_string(j) + "); refine the mesh near the body");
            g.ratio = dg / di;
            ghost_.push_back(g);
            ghost_mask_[c] = 1;
        }
    // surface samples for the loads, at most half a cell apart
    const auto add_point = [&](double wx, double wy, double nxw, double nyw, double ds) {
        SurfacePoint sp;
        sp.x = wx;
        sp.y = wy;
        sp.nx = nxw;
        sp.ny = nyw;
        sp.ds = ds;
        if (!probe_out(wx, wy, nxw, nyw, 0.0, sp.d, sp.probe)) return;
        if (make_probe(m, wx + 2.0 * sp.d * nxw, wy + 2.0 * sp.d * nyw, sp.probe2)) surface_.push_back(sp);
    };
    if (shape_ == Shape::Circle) {
        const double pi = std::acos(-1.0);
        const int n = std::max(64, static_cast<int>(std::ceil(2.0 * pi * r_ / (0.5 * h))));
        for (int k = 0; k < n; ++k) {
            const double th = 2.0 * pi * (k + 0.5) / n;
            add_point(cx_ + r_ * std::cos(th), cy_ + r_ * std::sin(th), std::cos(th), std::sin(th), 2.0 * pi * r_ / n);
        }
    } else {
        const auto& pts = poly_.pts;
        for (size_t k = 0; k < pts.size(); ++k) {
            const auto &a = pts[k], &b = pts[(k + 1) % pts.size()];
            const double ex = b[0] - a[0], ey = b[1] - a[1], l = std::sqrt(ex * ex + ey * ey);
            if (l <= 0.0) continue;
            double nxs = ey / l, nys = -ex / l;
            const double mx = 0.5 * (a[0] + b[0]), my = 0.5 * (a[1] + b[1]);
            if (poly_.contains(mx + 1e-6 * l * nxs, my + 1e-6 * l * nys)) {
                nxs = -nxs;
                nys = -nys;
            }
            const int ns = std::max(1, static_cast<int>(std::ceil(l / (0.5 * h))));
            for (int q = 0; q < ns; ++q) {
                const double t = (q + 0.5) / ns;
                add_point(a[0] + t * ex, a[1] + t * ey, nxs, nys, l / ns);
            }
        }
    }
    ghost_on_ = true;
}

void ImmersedBoundary::fill_ghosts_part(const CellIndex& local, const CellState& in, Field& U, const EOS& eos, bool reset) const {
    // every ghost state is computed from fluid cells only, then written: the result does not depend on the order
    const int ng = static_cast<int>(ghost_.size());
    std::vector<State> val(static_cast<size_t>(ng));
    std::vector<int> dst(static_cast<size_t>(ng));
#pragma omp parallel for schedule(static) if (ng >= kOmpMinWork)
    for (int k = 0; k < ng; ++k) {
        const Ghost& g = ghost_[static_cast<size_t>(k)];
        dst[static_cast<size_t>(k)] = local(g.cell);
        if (dst[static_cast<size_t>(k)] < 0) continue;
        const State W = g.image.at(in, eos);
        const double un = W[1] * g.nx + W[2] * g.ny;
        double ux, uy;
        if (slip_) {   // tangential velocity kept, normal velocity reflected through zero at the wall
            ux = W[1] - un * g.nx - g.ratio * un * g.nx;
            uy = W[2] - un * g.ny - g.ratio * un * g.ny;
        } else {   // whole velocity through zero at the wall
            ux = -g.ratio * W[1];
            uy = -g.ratio * W[2];
        }
        val[static_cast<size_t>(k)] = eos.prim_to_cons({W[0], ux, uy, W[3], 0.0});
    }
    for (int k = 0; k < ng; ++k)
        if (dst[static_cast<size_t>(k)] >= 0) U[static_cast<size_t>(dst[static_cast<size_t>(k)])] = val[static_cast<size_t>(k)];
    // deeper solid cells: at rest, or (reset) at the reference state
    const State deep = eos.prim_to_cons({rho_ref_, 0.0, 0.0, p_ref_, 0.0});
    const int n = static_cast<int>(mask_.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) {
        if (!mask_[c] || ghost_mask_[c]) continue;
        const int l = local(c);
        if (l < 0) continue;
        if (reset) {
            U[static_cast<size_t>(l)] = deep;
            continue;
        }
        State W = eos.cons_to_prim(U[static_cast<size_t>(l)]);
        W[1] = W[2] = W[4] = 0.0;
        U[static_cast<size_t>(l)] = eos.prim_to_cons(W);
    }
}

std::vector<int> ImmersedBoundary::ghost_inputs(const CellIndex& local) const {
    std::vector<int> v;
    for (const Ghost& g : ghost_)
        if (local(g.cell) >= 0)
            for (int k = 0; k < 4; ++k)
                if (g.image.w[k] != 0.0) v.push_back(g.image.cell[k]);
    return v;
}

std::vector<int> ImmersedBoundary::surface_inputs(int part, int parts) const {
    std::vector<int> v;
    for (size_t k = static_cast<size_t>(part); k < surface_.size(); k += static_cast<size_t>(parts))
        for (const Probe* pr : {&surface_[k].probe, &surface_[k].probe2})
            for (int q = 0; q < 4; ++q)
                if (pr->w[q] != 0.0) v.push_back(pr->cell[q]);
    return v;
}

std::array<double, 3> ImmersedBoundary::surface_loads(const Field& U, const EOS& eos, double mu, double cx, double cy) const {
    return surface_loads_part(0, 1, [&U](int c) -> const State& { return U[c]; }, eos, mu, cx, cy);
}

std::array<double, 3> ImmersedBoundary::surface_loads_part(int part, int parts, const CellState& in, const EOS& eos, double mu,
                                                           double cx, double cy) const {
    double fx = 0.0, fy = 0.0, mz = 0.0;
    for (size_t k = static_cast<size_t>(part); k < surface_.size(); k += static_cast<size_t>(parts)) {
        const SurfacePoint& sp = surface_[k];
        const State W1 = sp.probe.at(in, eos), W2 = sp.probe2.at(in, eos);
        const double pw = 2.0 * W1[3] - W2[3];   // wall pressure, linear extrapolation from d and 2 d
        double tx = -pw * sp.nx, ty = -pw * sp.ny;
        if (mu > 0.0) {   // wall shear: slope at the wall of the quadratic through 0 (no-slip), u_t(d), u_t(2 d)
            const double un1 = W1[1] * sp.nx + W1[2] * sp.ny, un2 = W2[1] * sp.nx + W2[2] * sp.ny;
            tx += mu * (4.0 * (W1[1] - un1 * sp.nx) - (W2[1] - un2 * sp.nx)) / (2.0 * sp.d);
            ty += mu * (4.0 * (W1[2] - un1 * sp.ny) - (W2[2] - un2 * sp.ny)) / (2.0 * sp.d);
        }
        const double ex = tx * sp.ds, ey = ty * sp.ds;
        fx += ex;
        fy += ey;
        mz += (sp.x - cx) * ey - (sp.y - cy) * ex;
    }
    return {fx, fy, mz};
}

}   // namespace fv
