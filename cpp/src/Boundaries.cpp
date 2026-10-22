#include "fv/Boundaries.hpp"

#include <cctype>
#include <stdexcept>

namespace fv {

namespace {
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
}   // namespace

BCType parse_bc(const std::string& s_in) {
    const std::string s = lower(s_in);
    if (s == "wall" || s == "inviscid wall (slip)") return BCType::Wall;
    if (s == "no_slip" || s == "viscous wall (no-slip)") return BCType::NoSlip;
    if (s == "moving_wall" || s == "moving wall") return BCType::MovingWall;
    if (s == "vel_inlet" || s == "velocity inlet") return BCType::VelInlet;
    if (s == "massflow_inlet" || s == "mass flow inlet") return BCType::MassflowInlet;
    if (s == "outlet" || s == "pressure outlet") return BCType::Outlet;
    if (s == "transmissive") return BCType::Transmissive;
    if (s == "periodic") return BCType::Periodic;
    if (s == "nonreflecting" || s == "non_reflecting" || s == "non-reflecting outlet") return BCType::NonReflecting;
    throw std::invalid_argument("Unknown boundary condition type: '" + s_in + "'");
}

std::string bc_name(BCType t) {
    switch (t) {
    case BCType::Wall: return "wall";
    case BCType::NoSlip: return "no_slip";
    case BCType::MovingWall: return "moving_wall";
    case BCType::VelInlet: return "vel_inlet";
    case BCType::MassflowInlet: return "massflow_inlet";
    case BCType::Outlet: return "outlet";
    case BCType::Transmissive: return "transmissive";
    case BCType::NonReflecting: return "nonreflecting";
    default: return "periodic";
    }
}

void BoundaryManager::set_boundary_types(BCType left, BCType right, BCType bottom, BCType top) {
    types_[0] = left;
    types_[1] = right;
    types_[2] = bottom;
    types_[3] = top;
}

// ---------------------------------------------------------------------------
// Convective ghost cells
// ---------------------------------------------------------------------------

void BoundaryManager::apply_boundary_conditions(const Field& U, Field& UL, Field& UR, const TangentialHalos* th) const {
    const int nx = mesh_.nx, ny = mesh_.ny, nz = mesh_.nz;
    const Mesh& m = mesh_;
    UL.resize(m.num_faces());   // every face is written below (interior + sides)
    UR.resize(m.num_faces());

    // Interior x-faces
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < ny * nz; ++kj) {
        const int j = kj % ny, k = kj / ny;
        for (int i = 1; i < nx; ++i) {
            const int f = m.xface(i, j, k);
            UL[f] = U[m.cell(i - 1, j, k)];
            UR[f] = U[m.cell(i, j, k)];
        }
    }
    // Interior y-faces
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < (ny - 1) * nz; ++kj) {
        const int j = kj % (ny - 1) + 1, k = kj / (ny - 1);
        for (int i = 0; i < nx; ++i) {
            const int f = m.yface(i, j, k);
            UL[f] = U[m.cell(i, j - 1, k)];
            UR[f] = U[m.cell(i, j, k)];
        }
    }
    // Interior z-faces (3-D)
    if (m.is3d()) {
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
        for (int kj = 0; kj < ny * (nz - 1); ++kj) {
            const int j = kj % ny, k = kj / ny + 1;
            for (int i = 0; i < nx; ++i) {
                const int f = m.zface(i, j, k);
                UL[f] = U[m.cell(i, j, k - 1)];
                UR[f] = U[m.cell(i, j, k)];
            }
        }
    }

    // Sides in a fixed order: right, top, left, bottom (then back, front in 3-D).
    apply_side(U, UL, UR, Face::Right, th);
    apply_side(U, UL, UR, Face::Top, th);
    apply_side(U, UL, UR, Face::Left, th);
    apply_side(U, UL, UR, Face::Bottom, th);
    if (m.is3d()) {
        apply_side(U, UL, UR, Face::Back, th);
        apply_side(U, UL, UR, Face::Front, th);
    }
}

State BoundaryManager::ghost_state(Face side, const State& Ui, double u_target) const {
    const BCType bc = type(side);
    if (bc == BCType::Transmissive) return Ui;
    if (bc == BCType::Periodic || bc == BCType::NonReflecting)
        throw std::logic_error("BoundaryManager::ghost_state: no single-cell ghost state for this boundary type");
    const int axis = face_axis(side);
    const double nxn = axis == 0 ? 1.0 : 0.0;
    const double nyn = axis == 1 ? 1.0 : 0.0;
    const double nzn = axis == 2 ? 1.0 : 0.0;
    const State Wi = eos_.cons_to_prim(Ui);
    State Wg = Wi;
    switch (bc) {
    case BCType::Wall: {
        const double Vn = Wi[1] * nxn + Wi[2] * nyn + Wi[4] * nzn;
        Wg[1] = Wi[1] - 2.0 * Vn * nxn;
        Wg[2] = Wi[2] - 2.0 * Vn * nyn;
        Wg[4] = Wi[4] - 2.0 * Vn * nzn;
        break;
    }
    case BCType::NoSlip:
        Wg[1] = -Wi[1];
        Wg[2] = -Wi[2];
        Wg[4] = -Wi[4];
        break;
    case BCType::MovingWall:
    case BCType::VelInlet:
        Wg[1] = 2.0 * params_.u_in - Wi[1];
        Wg[2] = 2.0 * params_.v_in - Wi[2];
        Wg[4] = -Wi[4];   // wall / inlet velocity has no z-component
        break;
    case BCType::MassflowInlet:
        Wg[1] = 2.0 * u_target - Wi[1];
        Wg[2] = 2.0 * params_.v_in - Wi[2];
        Wg[4] = -Wi[4];
        break;
    case BCType::Outlet: {
        const double Vn = Wi[1] * nxn + Wi[2] * nyn + Wi[4] * nzn;
        const bool outflow = Vn >= 0;
        Wg[1] = outflow ? Wi[1] : 0.0;
        Wg[2] = outflow ? Wi[2] : 0.0;
        Wg[4] = outflow ? Wi[4] : 0.0;
        Wg[3] = std::max(params_.p_out, 1e-5);
        break;
    }
    default: break;
    }
    return eos_.prim_to_cons(Wg);
}

void BoundaryManager::apply_side(const Field& U, Field& UL, Field& UR, Face side, const TangentialHalos* th) const {
    const int n = mesh_.boundary_count(side);
    if (n == 0) return;
    const BCType bc = type(side);
    const bool isR = (side == Face::Right || side == Face::Top || side == Face::Back);   // ghost goes into U_R
    const int axis = face_axis(side);
    const int nx = mesh_.nx, ny = mesh_.ny, nz = mesh_.nz;

    // Mass-flow inlet needs the patch-averaged interior density.
    double u_target = 0.0;
    if (bc == BCType::MassflowInlet) {
        double sum = 0.0, count = n;
        for (int k = 0; k < n; ++k) sum += eos_.cons_to_prim(U[mesh_.boundary_cell(side, k)])[0];
        if (reducers_[static_cast<int>(side)]) reducers_[static_cast<int>(side)](sum, count);
        const double rho_avg = std::max(sum / count, 1e-5);
        u_target = (params_.rho_in * params_.u_in) / rho_avg;
    }

    for (int k = 0; k < n; ++k) {
        const int f = mesh_.boundary_face(side, k);
        const int cell = mesh_.boundary_cell(side, k);
        const State& Ui = U[cell];
        State Ug;

        if (bc == BCType::Periodic) {
            int opp;
            switch (side) {
            case Face::Left: opp = cell + (nx - 1); break;
            case Face::Right: opp = cell - (nx - 1); break;
            case Face::Bottom: opp = cell + (ny - 1) * nx; break;
            case Face::Top: opp = cell - (ny - 1) * nx; break;
            case Face::Front: opp = cell + (nz - 1) * ny * nx; break;
            default: opp = cell - (nz - 1) * ny * nx; break;
            }
            Ug = U[opp];
        } else if (bc != BCType::NonReflecting) {
            Ug = ghost_state(side, Ui, u_target);
        } else {
            const State Wi = eos_.cons_to_prim(Ui);
            State Wg = Wi;
            switch (bc) {
            case BCType::NonReflecting: {
                // Characteristic outlet (Rudy & Strikwerda 1980; Poinsot & Lele 1992; transverse
                // terms: Lodato, Domingo & Vervisch 2008). The ghost differs from the cell only in
                // the incoming acoustic characteristic w1 = p - rho c u_out, whose amplitude is
                //   L1 = K (p - p_out) - (1 - beta) T1,   K = sigma c (1 - M^2) / L,
                // with T1 the transverse terms of the w1 equation. Across one cell the jump is
                // dw1 = -h L1 / (c - u_out); the K part alone gives -sigma (1 + M) h / L (p - p_out).
                // Outgoing acoustic, entropy and vorticity waves are extrapolated (as transmissive).
                const double s = isR ? 1.0 : -1.0;   // outward normal along the axis
                const int qv = kMom[axis];
                const double u_out = s * Wi[qv];
                const double c = eos_.sound_speed(Wi[3], Wi[0]);
                const double M = std::min(std::max(u_out / c, -1.0), 1.0);
                if (M >= 1.0) break;   // supersonic outflow: nothing enters
                const double hs[3] = {mesh_.dx, mesh_.dy, mesh_.dz};
                const double h = hs[axis];
                const double K = params_.nrbc_sigma * c * (1.0 - M * M) / params_.length[axis];
                const double beta = params_.nrbc_beta >= 0.0 ? params_.nrbc_beta : std::abs(M);
                double T1 = 0.0;   // sum over tangential axes t: v_t dp/dt + rho c^2 dv_t/dt - rho c s v_t du_n/dt
                if (beta < 1.0) {
                    const int ijk[3] = {cell % nx, (cell / nx) % ny, cell / (nx * ny)};
                    const int n3[3] = {nx, ny, nz};
                    for (int t = 0; t < 3; ++t) {
                        if (t == axis || n3[t] == 1) continue;
                        const bool wrap = type(static_cast<Face>(2 * t)) == BCType::Periodic &&
                                          type(static_cast<Face>(2 * t + 1)) == BCType::Periodic;
                        // Across an MPI interface the neighbour comes from the strip halo
                        const bool mpi_t = th && t == (th->dir == 'y' ? 1 : 0);
                        const Field* h_lo = mpi_t ? th->lo : nullptr;
                        const Field* h_hi = mpi_t ? th->hi : nullptr;
                        auto prim = [&](int d) {
                            if (d < 0 || d >= n3[t]) {
                                const Field& H = d < 0 ? *h_lo : *h_hi;
                                const int D = th->depth, layer = d < 0 ? D - 1 : 0;
                                const size_t hidx = th->dir == 'y'
                                                        ? (static_cast<size_t>(layer) * nz + ijk[2]) * nx + ijk[0]
                                                        : (static_cast<size_t>(ijk[2]) * ny + ijk[1]) * D + layer;
                                return eos_.cons_to_prim(H[hidx]);
                            }
                            int idx[3] = {ijk[0], ijk[1], ijk[2]};
                            idx[t] = d;
                            return eos_.cons_to_prim(U[mesh_.cell(idx[0], idx[1], idx[2])]);
                        };
                        int lo = ijk[t] - 1, hi = ijk[t] + 1;   // central, wrapped or one-sided at the ends
                        double span = 2.0;
                        if (wrap) {
                            lo = (lo + n3[t]) % n3[t];
                            hi = hi % n3[t];
                        } else {
                            if (lo < 0 && !h_lo) {
                                lo = 0;
                                span -= 1.0;
                            }
                            if (hi >= n3[t] && !h_hi) {
                                hi = n3[t] - 1;
                                span -= 1.0;
                            }
                            if (span <= 0.0) continue;
                        }
                        const State Wl = prim(lo), Wh = prim(hi);
                        const double ht = span * hs[t];
                        const int qt = kMom[t];
                        T1 += Wi[qt] * (Wh[3] - Wl[3]) / ht + Wi[0] * c * c * (Wh[qt] - Wl[qt]) / ht -
                              Wi[0] * c * s * Wi[qt] * (Wh[qv] - Wl[qv]) / ht;
                    }
                }
                const double dw1 = -h * (K * (Wi[3] - params_.p_out) - (1.0 - beta) * T1) / (c - u_out);
                Wg[3] = Wi[3] + 0.5 * dw1;
                Wg[qv] = Wi[qv] - s * dw1 / (2.0 * Wi[0] * c);
                Wg[0] = Wi[0] + 0.5 * dw1 / (c * c);
                break;
            }
            default: break;
            }
            Ug = eos_.prim_to_cons(Wg);
        }

        if (isR) {
            UL[f] = Ui;
            UR[f] = Ug;
        } else {
            UL[f] = Ug;
            UR[f] = Ui;
        }
    }
}

// ---------------------------------------------------------------------------
// Diffusive ghost cells
// ---------------------------------------------------------------------------

void BoundaryManager::get_padded_U(const Field& U, Field& P) const {
    if (mesh_.is3d()) {
        padded_3d(U, P);
        return;
    }
    const int nx = mesh_.nx, ny = mesh_.ny;
    const int W = nx + 2;
    P.resize(static_cast<size_t>(ny + 2) * W);

    // Edge padding (zero-gradient everywhere, corners included)
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
    for (int jj = 0; jj < ny + 2; ++jj) {
        const int j = std::min(std::max(jj - 1, 0), ny - 1);
        for (int ii = 0; ii < W; ++ii) {
            const int i = std::min(std::max(ii - 1, 0), nx - 1);
            P[jj * W + ii] = U[j * nx + i];
        }
    }

    auto at = [&](int jj, int ii) -> State& { return P[jj * W + ii]; };
    const double u_in = params_.u_in, v_in = params_.v_in;

    // Bottom
    switch (type(Face::Bottom)) {
    case BCType::NoSlip:
        for (int i = 0; i < nx; ++i) {
            const State& in = U[i];
            at(0, i + 1)[1] = -in[1];
            at(0, i + 1)[2] = -in[2];
        }
        break;
    case BCType::MovingWall:
        for (int i = 0; i < nx; ++i) {
            const State& in = U[i];
            at(0, i + 1)[1] = 2.0 * in[0] * u_in - in[1];
            at(0, i + 1)[2] = -in[2];
        }
        break;
    case BCType::Wall:
        for (int i = 0; i < nx; ++i) at(0, i + 1)[2] = -U[i][2];
        break;
    case BCType::Periodic:
        for (int i = 0; i < nx; ++i) at(0, i + 1) = U[(ny - 1) * nx + i];
        break;
    default: break;
    }
    // Top
    switch (type(Face::Top)) {
    case BCType::NoSlip:
        for (int i = 0; i < nx; ++i) {
            const State& in = U[(ny - 1) * nx + i];
            at(ny + 1, i + 1)[1] = -in[1];
            at(ny + 1, i + 1)[2] = -in[2];
        }
        break;
    case BCType::Wall:
        for (int i = 0; i < nx; ++i) at(ny + 1, i + 1)[2] = -U[(ny - 1) * nx + i][2];
        break;
    case BCType::Periodic:
        for (int i = 0; i < nx; ++i) at(ny + 1, i + 1) = U[i];
        break;
    case BCType::MovingWall:
        for (int i = 0; i < nx; ++i) {
            const State& in = U[(ny - 1) * nx + i];
            at(ny + 1, i + 1)[1] = 2.0 * in[0] * u_in - in[1];
            at(ny + 1, i + 1)[2] = -in[2];
        }
        break;
    default: break;
    }
    // Left
    switch (type(Face::Left)) {
    case BCType::NoSlip:
        for (int j = 0; j < ny; ++j) {
            const State& in = U[j * nx];
            at(j + 1, 0)[1] = -in[1];
            at(j + 1, 0)[2] = -in[2];
        }
        break;
    case BCType::Wall:
        for (int j = 0; j < ny; ++j) at(j + 1, 0)[1] = -U[j * nx][1];
        break;
    case BCType::Periodic:
        for (int j = 0; j < ny; ++j) at(j + 1, 0) = U[j * nx + nx - 1];
        break;
    case BCType::MovingWall:
        for (int j = 0; j < ny; ++j) {
            const State& in = U[j * nx];
            at(j + 1, 0)[1] = -in[1];
            at(j + 1, 0)[2] = 2.0 * in[0] * v_in - in[2];
        }
        break;
    default: break;
    }
    // Right
    switch (type(Face::Right)) {
    case BCType::NoSlip:
        for (int j = 0; j < ny; ++j) {
            const State& in = U[j * nx + nx - 1];
            at(j + 1, nx + 1)[1] = -in[1];
            at(j + 1, nx + 1)[2] = -in[2];
        }
        break;
    case BCType::Wall:
        for (int j = 0; j < ny; ++j) at(j + 1, nx + 1)[1] = -U[j * nx + nx - 1][1];
        break;
    case BCType::Periodic:
        for (int j = 0; j < ny; ++j) at(j + 1, nx + 1) = U[j * nx];
        break;
    case BCType::MovingWall:
        for (int j = 0; j < ny; ++j) {
            const State& in = U[j * nx + nx - 1];
            at(j + 1, nx + 1)[1] = -in[1];
            at(j + 1, nx + 1)[2] = 2.0 * in[0] * v_in - in[2];
        }
        break;
    default: break;
    }
}

// 3-D padded state, (nz+2) x (ny+2) x (nx+2): edge replication, then the same
// momentum rules as in 2-D on each side's ghost layer (sides in the order
// bottom, top, left, right, front, back).
void BoundaryManager::padded_3d(const Field& U, Field& P) const {
    const Mesh& m = mesh_;
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const int PX = nx + 2, PY = ny + 2;
    P.resize(static_cast<size_t>(nz + 2) * PY * PX);
    auto at = [&](int ii, int jj, int kk) -> State& { return P[(static_cast<size_t>(kk) * PY + jj) * PX + ii]; };

#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kk = 0; kk < nz + 2; ++kk) {
        const int k = std::min(std::max(kk - 1, 0), nz - 1);
        for (int jj = 0; jj < PY; ++jj) {
            const int j = std::min(std::max(jj - 1, 0), ny - 1);
            for (int ii = 0; ii < PX; ++ii) at(ii, jj, kk) = U[m.cell(std::min(std::max(ii - 1, 0), nx - 1), j, k)];
        }
    }

    const double uw[3] = {params_.u_in, params_.v_in, 0.0};   // moving-wall velocity
    for (Face side : {Face::Bottom, Face::Top, Face::Left, Face::Right, Face::Front, Face::Back}) {
        const BCType bc = type(side);
        if (bc != BCType::NoSlip && bc != BCType::Wall && bc != BCType::MovingWall && bc != BCType::Periodic) continue;
        const int d = face_axis(side);
        const bool hi = (side == Face::Right || side == Face::Top || side == Face::Back);
        const int n_d = d == 0 ? nx : (d == 1 ? ny : nz);
        const int g = hi ? n_d + 1 : 0;                           // ghost layer (padded index along d)
        const int in = hi ? n_d - 1 : 0;                          // adjacent interior layer (cell index)
        const int op = hi ? 0 : n_d - 1;                          // opposite interior layer (periodic)
        const int na = d == 0 ? ny : nx, nb = d == 2 ? ny : nz;   // transverse extents
        for (int b = 0; b < nb; ++b)
            for (int a = 0; a < na; ++a) {
                int i, j, k;   // interior cell, padded ghost (ii, jj, kk)
                int ii, jj, kk, io, jo, ko;
                if (d == 0) {
                    i = in;
                    j = a;
                    k = b;
                    ii = g;
                    jj = j + 1;
                    kk = k + 1;
                    io = op;
                    jo = j;
                    ko = k;
                } else if (d == 1) {
                    i = a;
                    j = in;
                    k = b;
                    ii = i + 1;
                    jj = g;
                    kk = k + 1;
                    io = i;
                    jo = op;
                    ko = k;
                } else {
                    i = a;
                    j = b;
                    k = in;
                    ii = i + 1;
                    jj = j + 1;
                    kk = g;
                    io = i;
                    jo = j;
                    ko = op;
                }
                State& gh = at(ii, jj, kk);
                const State& c = U[m.cell(i, j, k)];
                switch (bc) {
                case BCType::NoSlip:
                    for (int q : kMom) gh[q] = -c[q];
                    break;
                case BCType::Wall: gh[kMom[d]] = -c[kMom[d]]; break;
                case BCType::MovingWall:
                    for (int t = 0; t < 3; ++t) gh[kMom[t]] = (t == d) ? -c[kMom[t]] : 2.0 * c[0] * uw[t] - c[kMom[t]];
                    break;
                default:   // periodic
                    gh = U[m.cell(io, jo, ko)];
                    break;
                }
            }
    }
}

}   // namespace fv
