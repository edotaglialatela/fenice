#include "fv/TimeScheme.hpp"

#include <cctype>
#include <stdexcept>

#include "fv/Profile.hpp"

namespace fv {

namespace {

// Guard against S_max -> 0 in the CFL time-step formula.
constexpr double kDtEps = 1e-10;

inline int isize(const Field& f) { return static_cast<int>(f.size()); }

// out = a + s * b
void axpy(Field& out, const Field& a, double s, const Field& b) {
    ProfileScope profile(Section::UpdateAxpy);
    const int n = isize(a);
    out.resize(a.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) out[c][q] = a[c][q] + s * b[c][q];
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}   // namespace

TimeSchemeType parse_time_scheme(const std::string& s_in) {
    const std::string s = lower(s_in);
    if (s == "explicit euler" || s == "euler" || s == "explicit_euler") return TimeSchemeType::ExplicitEuler;
    if (s == "ssp runge-kutta 2" || s == "ssprk2" || s == "rk2") return TimeSchemeType::SSPRK2;
    if (s == "ssp runge-kutta 3" || s == "ssprk3" || s == "rk3") return TimeSchemeType::SSPRK3;
    if (s == "runge-kutta 4" || s == "rk4") return TimeSchemeType::RungeKutta4;
    throw std::invalid_argument("Unknown time scheme: '" + s_in + "'");
}

double TimeScheme::advance_fixed_dt(Field& U, const ResidualFn& res, double dt) {
    const double s = res(U, R1_);
    axpy(U, U, dt, R1_);
    return s;
}

// ---------------------------------------------------------------------------
// Explicit Euler
// ---------------------------------------------------------------------------

StepResult ExplicitEuler::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max) {
    stage_time = t;
    const double s = res(U, R1_);
    double dt = cfl * min_delta / (s + kDtEps);
    dt = std::min(dt, t_max - t);
    if (dt_max > 0.0) dt = std::min(dt, dt_max);
    axpy(U, U, dt, R1_);
    return {dt, s};
}

double ExplicitEuler::advance_fixed_dt(Field& U, const ResidualFn& res, double dt) {
    const double s = res(U, R1_);
    axpy(U, U, dt, R1_);
    return s;
}

// ---------------------------------------------------------------------------
// SSP-RK2 (Shu-Osher form)
// ---------------------------------------------------------------------------

namespace {
// U <- 0.5*U + 0.5*(U1 + dt*R2)
void ssp_combine(Field& U, const Field& U1, double dt, const Field& R2) {
    ProfileScope profile(Section::UpdateSspCombine);
    const int n = isize(U);
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) U[c][q] = 0.5 * U[c][q] + 0.5 * (U1[c][q] + dt * R2[c][q]);
}
}   // namespace

StepResult SSPRK2::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max) {
    stage_time = t;
    const double s1 = res(U, R1_);
    double dt = cfl * min_delta / (s1 + kDtEps);
    dt = std::min(dt, t_max - t);
    if (dt_max > 0.0) dt = std::min(dt, dt_max);
    axpy(tmp_, U, dt, R1_);
    stage_time = t + dt;
    const double s2 = res(tmp_, R2_);
    ssp_combine(U, tmp_, dt, R2_);
    return {dt, std::max(s1, s2)};
}

double SSPRK2::advance_fixed_dt(Field& U, const ResidualFn& res, double dt) {
    const double s = res(U, R1_);
    axpy(tmp_, U, dt, R1_);
    res(tmp_, R2_);
    ssp_combine(U, tmp_, dt, R2_);
    return s;
}

// ---------------------------------------------------------------------------
// SSP-RK3 (Shu-Osher form)
//   U1 = U + dt R(U);  U2 = 3/4 U + 1/4 (U1 + dt R(U1));  U^{n+1} = 1/3 U + 2/3 (U2 + dt R(U2))
// ---------------------------------------------------------------------------

namespace {
// out = a * u0 + b * (v + dt * R); out may be v or u0 (element-wise)
void ssp3_combine(Field& out, const Field& u0, double a, const Field& v, double b, double dt, const Field& R) {
    ProfileScope profile(Section::UpdateSsp3Combine);
    const int n = isize(out);
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) out[c][q] = a * u0[c][q] + b * (v[c][q] + dt * R[c][q]);
}
}   // namespace

StepResult SSPRK3::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max) {
    stage_time = t;
    const double s1 = res(U, R1_);
    double dt = cfl * min_delta / (s1 + kDtEps);
    dt = std::min(dt, t_max - t);
    if (dt_max > 0.0) dt = std::min(dt, dt_max);
    axpy(tmp_, U, dt, R1_);
    stage_time = t + dt;
    const double s2 = res(tmp_, R2_);
    ssp3_combine(tmp_, U, 0.75, tmp_, 0.25, dt, R2_);
    stage_time = t + 0.5 * dt;
    const double s3 = res(tmp_, R3_);
    ssp3_combine(U, U, 1.0 / 3.0, tmp_, 2.0 / 3.0, dt, R3_);
    return {dt, std::max(s1, std::max(s2, s3))};
}

double SSPRK3::advance_fixed_dt(Field& U, const ResidualFn& res, double dt) {
    const double s = res(U, R1_);
    axpy(tmp_, U, dt, R1_);
    res(tmp_, R2_);
    ssp3_combine(tmp_, U, 0.75, tmp_, 0.25, dt, R2_);
    res(tmp_, R3_);
    ssp3_combine(U, U, 1.0 / 3.0, tmp_, 2.0 / 3.0, dt, R3_);
    return s;
}

// ---------------------------------------------------------------------------
// Classical RK4
// ---------------------------------------------------------------------------

namespace {
// U <- U + (dt/6)*(k1 + 2k2 + 2k3 + k4)
void rk4_combine(Field& U, double dt, const Field& k1, const Field& k2, const Field& k3, const Field& k4) {
    ProfileScope profile(Section::UpdateRk4Combine);
    const int n = isize(U);
    const double w = dt / 6.0;
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) U[c][q] = U[c][q] + w * (k1[c][q] + 2.0 * k2[c][q] + 2.0 * k3[c][q] + k4[c][q]);
}
}   // namespace

StepResult RungeKutta4::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max) {
    stage_time = t;
    const double s1 = res(U, R1_);
    double dt = cfl * min_delta / (s1 + kDtEps);
    dt = std::min(dt, t_max - t);
    if (dt_max > 0.0) dt = std::min(dt, dt_max);
    axpy(tmp_, U, 0.5 * dt, R1_);
    stage_time = t + 0.5 * dt;
    const double s2 = res(tmp_, R2_);
    axpy(tmp_, U, 0.5 * dt, R2_);
    const double s3 = res(tmp_, R3_);
    axpy(tmp_, U, dt, R3_);
    stage_time = t + dt;
    const double s4 = res(tmp_, R4_);
    rk4_combine(U, dt, R1_, R2_, R3_, R4_);
    return {dt, std::max(std::max(s1, s2), std::max(s3, s4))};
}

double RungeKutta4::advance_fixed_dt(Field& U, const ResidualFn& res, double dt) {
    const double s = res(U, R1_);
    axpy(tmp_, U, 0.5 * dt, R1_);
    res(tmp_, R2_);
    axpy(tmp_, U, 0.5 * dt, R2_);
    res(tmp_, R3_);
    axpy(tmp_, U, dt, R3_);
    res(tmp_, R4_);
    rk4_combine(U, dt, R1_, R2_, R3_, R4_);
    return s;
}

std::unique_ptr<TimeScheme> make_time_scheme(TimeSchemeType type, double cfl) {
    switch (type) {
    case TimeSchemeType::SSPRK2: return std::make_unique<SSPRK2>(cfl);
    case TimeSchemeType::SSPRK3: return std::make_unique<SSPRK3>(cfl);
    case TimeSchemeType::RungeKutta4: return std::make_unique<RungeKutta4>(cfl);
    default: return std::make_unique<ExplicitEuler>(cfl);
    }
}

}   // namespace fv
