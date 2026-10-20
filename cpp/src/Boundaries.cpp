#include "fv/Boundaries.hpp"

#include <cctype>
#include <stdexcept>

namespace fv {

namespace {
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
}  // namespace

BCType parse_bc(const std::string& s_in) {
    const std::string s = lower(s_in);
    if (s == "wall" || s == "inviscid wall (slip)")             return BCType::Wall;
    if (s == "no_slip" || s == "viscous wall (no-slip)")        return BCType::NoSlip;
    if (s == "moving_wall" || s == "moving wall")               return BCType::MovingWall;
    if (s == "vel_inlet" || s == "velocity inlet")              return BCType::VelInlet;
    if (s == "massflow_inlet" || s == "mass flow inlet")        return BCType::MassflowInlet;
    if (s == "outlet" || s == "pressure outlet")                return BCType::Outlet;
    if (s == "transmissive")                                    return BCType::Transmissive;
    if (s == "periodic")                                        return BCType::Periodic;
    throw std::invalid_argument("Unknown boundary condition type: '" + s_in + "'");
}

std::string bc_name(BCType t) {
    switch (t) {
        case BCType::Wall:          return "wall";
        case BCType::NoSlip:        return "no_slip";
        case BCType::MovingWall:    return "moving_wall";
        case BCType::VelInlet:      return "vel_inlet";
        case BCType::MassflowInlet: return "massflow_inlet";
        case BCType::Outlet:        return "outlet";
        case BCType::Transmissive:  return "transmissive";
        default:                    return "periodic";
    }
}

void BoundaryManager::set_boundary_types(BCType left, BCType right, BCType bottom, BCType top) {
    types_[0] = left;  types_[1] = right;
    types_[2] = bottom; types_[3] = top;
}

// ---------------------------------------------------------------------------
// Convective ghost cells
// ---------------------------------------------------------------------------

void BoundaryManager::apply_boundary_conditions(const Field& U, Field& UL, Field& UR) const {
    const int nx = mesh_.nx, ny = mesh_.ny, nz = mesh_.nz;
    const Mesh& m = mesh_;
    UL.resize(m.num_faces());   // every face is written below (interior + sides)
    UR.resize(m.num_faces());

    // Interior x-faces
#pragma omp parallel for schedule(static) if(nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < ny * nz; ++kj) {
        const int j = kj % ny, k = kj / ny;
        for (int i = 1; i < nx; ++i) {
            const int f = m.xface(i, j, k);
            UL[f] = U[m.cell(i - 1, j, k)];
            UR[f] = U[m.cell(i, j, k)];
        }
    }
    // Interior y-faces
#pragma omp parallel for schedule(static) if(nx * ny * nz >= kOmpMinWork)
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
#pragma omp parallel for schedule(static) if(nx * ny * nz >= kOmpMinWork)
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
    apply_side(U, UL, UR, Face::Right);
    apply_side(U, UL, UR, Face::Top);
    apply_side(U, UL, UR, Face::Left);
    apply_side(U, UL, UR, Face::Bottom);
    if (m.is3d()) {
        apply_side(U, UL, UR, Face::Back);
        apply_side(U, UL, UR, Face::Front);
    }
}

void BoundaryManager::apply_side(const Field& U, Field& UL, Field& UR, Face side) const {
    const int n = mesh_.boundary_count(side);
    if (n == 0) return;
    const BCType bc   = type(side);
    const bool   isR  = (side == Face::Right || side == Face::Top || side == Face::Back);   // ghost goes into U_R
    const int    axis = face_axis(side);
    const double nxn  = axis == 0 ? 1.0 : 0.0;
    const double nyn  = axis == 1 ? 1.0 : 0.0;
    const double nzn  = axis == 2 ? 1.0 : 0.0;
    const int    nx = mesh_.nx, ny = mesh_.ny, nz = mesh_.nz;

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
        const int    f    = mesh_.boundary_face(side, k);
        const int    cell = mesh_.boundary_cell(side, k);
        const State& Ui   = U[cell];
        State        Ug;

        if (bc == BCType::Periodic) {
            int opp;
            switch (side) {
                case Face::Left:   opp = cell + (nx - 1); break;
                case Face::Right:  opp = cell - (nx - 1); break;
                case Face::Bottom: opp = cell + (ny - 1) * nx; break;
                case Face::Top:    opp = cell - (ny - 1) * nx; break;
                case Face::Front:  opp = cell + (nz - 1) * ny * nx; break;
                default:           opp = cell - (nz - 1) * ny * nx; break;
            }
            Ug = U[opp];
        } else if (bc == BCType::Transmissive) {
            Ug = Ui;
        } else {
            const State Wi = eos_.cons_to_prim(Ui);
            State       Wg = Wi;
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
            Ug = eos_.prim_to_cons(Wg);
        }

        if (isR) { UL[f] = Ui; UR[f] = Ug; }
        else     { UL[f] = Ug; UR[f] = Ui; }
    }
}

// ---------------------------------------------------------------------------
// Diffusive ghost cells
// ---------------------------------------------------------------------------

void BoundaryManager::get_padded_U(const Field& U, Field& P) const {
    if (mesh_.is3d()) { padded_3d(U, P); return; }
    const int nx = mesh_.nx, ny = mesh_.ny;
    const int W  = nx + 2;
    P.resize(static_cast<size_t>(ny + 2) * W);

    // Edge padding (zero-gradient everywhere, corners included)
#pragma omp parallel for schedule(static) if(nx * ny >= kOmpMinWork)
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
            for (int i = 0; i < nx; ++i) { const State& in = U[i]; at(0, i + 1)[1] = -in[1]; at(0, i + 1)[2] = -in[2]; }
            break;
        case BCType::MovingWall:
            for (int i = 0; i < nx; ++i) { const State& in = U[i]; at(0, i + 1)[1] = 2.0 * in[0] * u_in - in[1]; at(0, i + 1)[2] = -in[2]; }
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
            for (int i = 0; i < nx; ++i) { const State& in = U[(ny - 1) * nx + i]; at(ny + 1, i + 1)[1] = -in[1]; at(ny + 1, i + 1)[2] = -in[2]; }
            break;
        case BCType::Wall:
            for (int i = 0; i < nx; ++i) at(ny + 1, i + 1)[2] = -U[(ny - 1) * nx + i][2];
            break;
        case BCType::Periodic:
            for (int i = 0; i < nx; ++i) at(ny + 1, i + 1) = U[i];
            break;
        case BCType::MovingWall:
            for (int i = 0; i < nx; ++i) { const State& in = U[(ny - 1) * nx + i]; at(ny + 1, i + 1)[1] = 2.0 * in[0] * u_in - in[1]; at(ny + 1, i + 1)[2] = -in[2]; }
            break;
        default: break;
    }
    // Left
    switch (type(Face::Left)) {
        case BCType::NoSlip:
            for (int j = 0; j < ny; ++j) { const State& in = U[j * nx]; at(j + 1, 0)[1] = -in[1]; at(j + 1, 0)[2] = -in[2]; }
            break;
        case BCType::Wall:
            for (int j = 0; j < ny; ++j) at(j + 1, 0)[1] = -U[j * nx][1];
            break;
        case BCType::Periodic:
            for (int j = 0; j < ny; ++j) at(j + 1, 0) = U[j * nx + nx - 1];
            break;
        case BCType::MovingWall:
            for (int j = 0; j < ny; ++j) { const State& in = U[j * nx]; at(j + 1, 0)[1] = -in[1]; at(j + 1, 0)[2] = 2.0 * in[0] * v_in - in[2]; }
            break;
        default: break;
    }
    // Right
    switch (type(Face::Right)) {
        case BCType::NoSlip:
            for (int j = 0; j < ny; ++j) { const State& in = U[j * nx + nx - 1]; at(j + 1, nx + 1)[1] = -in[1]; at(j + 1, nx + 1)[2] = -in[2]; }
            break;
        case BCType::Wall:
            for (int j = 0; j < ny; ++j) at(j + 1, nx + 1)[1] = -U[j * nx + nx - 1][1];
            break;
        case BCType::Periodic:
            for (int j = 0; j < ny; ++j) at(j + 1, nx + 1) = U[j * nx];
            break;
        case BCType::MovingWall:
            for (int j = 0; j < ny; ++j) { const State& in = U[j * nx + nx - 1]; at(j + 1, nx + 1)[1] = -in[1]; at(j + 1, nx + 1)[2] = 2.0 * in[0] * v_in - in[2]; }
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

#pragma omp parallel for schedule(static) if(nx * ny * nz >= kOmpMinWork)
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
        const int d  = face_axis(side);
        const bool hi = (side == Face::Right || side == Face::Top || side == Face::Back);
        const int n_d = d == 0 ? nx : (d == 1 ? ny : nz);
        const int g = hi ? n_d + 1 : 0;              // ghost layer (padded index along d)
        const int in = hi ? n_d - 1 : 0;             // adjacent interior layer (cell index)
        const int op = hi ? 0 : n_d - 1;             // opposite interior layer (periodic)
        const int na = d == 0 ? ny : nx, nb = d == 2 ? ny : nz;   // transverse extents
        for (int b = 0; b < nb; ++b)
            for (int a = 0; a < na; ++a) {
                int i, j, k;   // interior cell, padded ghost (ii, jj, kk)
                int ii, jj, kk, io, jo, ko;
                if (d == 0)      { i = in; j = a; k = b; ii = g; jj = j + 1; kk = k + 1; io = op; jo = j; ko = k; }
                else if (d == 1) { i = a; j = in; k = b; ii = i + 1; jj = g; kk = k + 1; io = i; jo = op; ko = k; }
                else             { i = a; j = b; k = in; ii = i + 1; jj = j + 1; kk = g; io = i; jo = j; ko = op; }
                State& gh = at(ii, jj, kk);
                const State& c = U[m.cell(i, j, k)];
                switch (bc) {
                    case BCType::NoSlip:
                        for (int q : kMom) gh[q] = -c[q];
                        break;
                    case BCType::Wall:
                        gh[kMom[d]] = -c[kMom[d]];
                        break;
                    case BCType::MovingWall:
                        for (int t = 0; t < 3; ++t)
                            gh[kMom[t]] = (t == d) ? -c[kMom[t]] : 2.0 * c[0] * uw[t] - c[kMom[t]];
                        break;
                    default:   // periodic
                        gh = U[m.cell(io, jo, ko)];
                        break;
                }
            }
    }
}

}  // namespace fv
