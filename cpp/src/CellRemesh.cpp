#include "fv/CellRemesh.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fv {

void CellIndicator::diffuse(const CellMesh& m, std::vector<double>& xi) const {
    const int N = m.num_leaves();
    std::vector<double> add(static_cast<size_t>(N));
    for (int it = 0; it < diffusion; ++it) {
        std::fill(add.begin(), add.end(), 0.0);
        for (const CellFace& f : m.faces) {
            if (f.kind == CellFaceKind::Boundary) continue;
            const int l = f.left, r = f.right;
            bool spread;
            if (m.level[l] > m.level[r])
                spread = xi[l] > diffusion_trigger;   // the finer leaf decides
            else if (m.level[l] < m.level[r])
                spread = xi[r] > diffusion_trigger;
            else
                spread = xi[l] > diffusion_trigger || xi[r] > diffusion_trigger;
            if (spread) {
                add[l] += diffusion_coeff;
                add[r] += diffusion_coeff;
            }
        }
        for (int c = 0; c < N; ++c) xi[c] += add[c];
    }
}

std::vector<double> CellIndicator::xi(const CellMesh& m, const Field& U, const EOS& eos) const {
    const int N = m.num_leaves();
    std::vector<double> x(static_cast<size_t>(N), 0.0);
    std::vector<double> rho, pr, vel;
    if (use_density) {
        rho.resize(static_cast<size_t>(N));
        for (int c = 0; c < N; ++c) rho[c] = U[c][0];
    }
    if (use_pressure || use_velocity) {
        pr.resize(static_cast<size_t>(N));
        vel.resize(static_cast<size_t>(N));
        for (int c = 0; c < N; ++c) {
            const State W = eos.cons_to_prim(U[c]);
            pr[c] = W[3];
            vel[c] = std::sqrt(W[1] * W[1] + W[2] * W[2] + W[4] * W[4]);
        }
    }
    const auto flag = [&](const std::vector<double>& v, double floor_value, int l, int r) {
        const double cl = v[l], cr = v[r];
        double vmin = std::min(std::abs(cl), std::abs(cr));
        if (vmin < value_floor) vmin = floor_value;
        if (std::abs(cr - cl) / vmin >= criterion) x[l] = x[r] = 1.0;
    };
    for (const CellFace& f : m.faces) {
        if (f.kind == CellFaceKind::Boundary) continue;
        if (use_density) flag(rho, value_floor, f.left, f.right);
        if (use_pressure) flag(pr, value_floor, f.left, f.right);
        if (use_velocity) flag(vel, velocity_floor, f.left, f.right);
    }
    diffuse(m, x);
    return x;
}

void CellAdaptor::set_body(std::function<ImmersedBoundary(const Mesh&)> factory, int margin) {
    body_ = std::move(factory);
    margin_ = margin;
    band_.clear();
}

void CellAdaptor::build_band() {
    const int M = mesh_.max_level, NX = mesh_.nx_at(M), NY = mesh_.ny_at(M);
    const ImmersedBoundary ib = body_(generate_cartesian_mesh(mesh_.Lx, mesh_.Ly, NX, NY, mesh_.x0, mesh_.y0));
    const std::vector<char>& mk = ib.mask();
    // dilation by `margin_` cells, separably
    std::vector<char> tmp(mk.size(), 0);
    band_.assign(mk.size(), 0);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            if (mk[static_cast<size_t>(j) * NX + i])
                for (int d = -margin_; d <= margin_; ++d)
                    if (i + d >= 0 && i + d < NX) tmp[static_cast<size_t>(j) * NX + i + d] = 1;
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            if (tmp[static_cast<size_t>(j) * NX + i])
                for (int d = -margin_; d <= margin_; ++d)
                    if (j + d >= 0 && j + d < NY) band_[static_cast<size_t>(j + d) * NX + i] = 1;
}

bool CellAdaptor::in_body_band(int c) {
    if (!body_) return false;
    if (band_.empty()) build_band();
    const int M = mesh_.max_level, NX = mesh_.nx_at(M), r = 1 << (M - mesh_.level[c]);
    for (int j = mesh_.cj[c] * r; j < (mesh_.cj[c] + 1) * r; ++j)
        for (int i = mesh_.ci[c] * r; i < (mesh_.ci[c] + 1) * r; ++i)
            if (band_[static_cast<size_t>(j) * NX + i]) return true;
    return false;
}

bool CellAdaptor::remesh() {
    using Clock = std::chrono::steady_clock;
    const auto sec = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = Clock::now();
    const int N = mesh_.num_leaves();
    last_xi = ind_.xi(mesh_, mesh_.U, eos_);
    std::vector<char> refine(static_cast<size_t>(N), 0), coarsen(static_cast<size_t>(N), 0);
    for (int c = 0; c < N; ++c) {
        refine[c] = mesh_.level[c] < mesh_.max_level && last_xi[c] >= ind_.split;
        coarsen[c] = mesh_.level[c] > 0 && last_xi[c] < ind_.join;
        if (in_body_band(c)) {
            refine[c] = mesh_.level[c] < mesh_.max_level;
            coarsen[c] = 0;
        }
    }
    const auto t1 = Clock::now();
    const bool changed = mesh_.adapt(refine, coarsen);
    const auto t2 = Clock::now();
    if (changed) engine_.build(mesh_);
    const auto t3 = Clock::now();
    t_indicator += sec(t0, t1);
    t_adapt += sec(t1, t2);
    t_build += sec(t2, t3);
    return changed;
}

int CellAdaptor::init_adapt(const std::function<void(CellMesh&)>& initial) {
    int passes = 0;
    for (; passes < mesh_.max_level + 1; ++passes) {
        initial(mesh_);
        if (!remesh()) {
            ++passes;
            break;
        }
    }
    initial(mesh_);
    return passes;
}

}   // namespace fv
