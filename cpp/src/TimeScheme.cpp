#include "fv/TimeScheme.hpp"

#include <cctype>
#include <stdexcept>

namespace fv {

namespace {

// Guard against S_max -> 0 in the CFL time-step formula.
constexpr double kDtEps = 1e-10;
// Absolute floor for the BDF2 inner-loop relative convergence check.
constexpr double kInnerResFloor = 1e-14;

inline int isize(const Field& f) { return static_cast<int>(f.size()); }

// out = a + s * b
void axpy(Field& out, const Field& a, double s, const Field& b) {
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
    if (s == "runge-kutta 4" || s == "rk4") return TimeSchemeType::RungeKutta4;
    if (s == "bdf2 dual time" || s == "bdf2") return TimeSchemeType::BDF2DualTime;
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

StepResult ExplicitEuler::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) {
    stage_time = t;
    const double s = res(U, R1_);
    double dt = cfl * min_delta / (s + kDtEps);
    dt = std::min(dt, t_max - t);
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
    const int n = isize(U);
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) U[c][q] = 0.5 * U[c][q] + 0.5 * (U1[c][q] + dt * R2[c][q]);
}
}   // namespace

StepResult SSPRK2::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) {
    stage_time = t;
    const double s1 = res(U, R1_);
    double dt = cfl * min_delta / (s1 + kDtEps);
    dt = std::min(dt, t_max - t);
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
// Classical RK4
// ---------------------------------------------------------------------------

namespace {
// U <- U + (dt/6)*(k1 + 2k2 + 2k3 + k4)
void rk4_combine(Field& U, double dt, const Field& k1, const Field& k2, const Field& k3, const Field& k4) {
    const int n = isize(U);
    const double w = dt / 6.0;
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) U[c][q] = U[c][q] + w * (k1[c][q] + 2.0 * k2[c][q] + 2.0 * k3[c][q] + k4[c][q]);
}
}   // namespace

StepResult RungeKutta4::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) {
    stage_time = t;
    const double s1 = res(U, R1_);
    double dt = cfl * min_delta / (s1 + kDtEps);
    dt = std::min(dt, t_max - t);
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

// ---------------------------------------------------------------------------
// BDF2 dual time
// ---------------------------------------------------------------------------

BDF2DualTime::BDF2DualTime(double cfl, double dt_phys, int max_inner_iter, double inner_tol_rel)
    : TimeScheme(cfl), dt_phys_(dt_phys), max_inner_iter_(max_inner_iter), inner_tol_rel_(inner_tol_rel) {
    if (dt_phys <= 0.0) throw std::invalid_argument("dt_phys must be positive, got " + std::to_string(dt_phys));
}

StepResult BDF2DualTime::step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) {
    const double dt_phys = std::min(dt_phys_, t_max - t);
    stage_time = t + dt_phys;   // the implicit stage: the residual of every inner iteration is at the new time
    const bool bdf1 = first_step_;   // BDF1 bootstrap: U^{n-1} := U^n
    const double alpha = bdf1 ? 1.0 / dt_phys : 1.5 / dt_phys;
    const Field& U_n = U;
    const int n = isize(U);

    Field U_star = U;
    Field& R = R1_;
    Field& R_star = R2_;
    R_star.resize(U.size());
    double S_last = 1.0;
    double R0 = -1.0;
    inner_iter_last_ = max_inner_iter_;

    for (int k = 0; k < max_inner_iter_; ++k) {
        double S = res(U_star, R);
        S = global_norm(S);
        S_last = S;

        double local_max = 0.0;
#pragma omp parallel if (n >= kOmpMinWork)
        {
            double lm = 0.0;
#pragma omp for schedule(static) nowait
            for (int c = 0; c < n; ++c)
                for (int q = 0; q < kNVar; ++q) {
                    const double src = bdf1 ? (U_star[c][q] - U_n[c][q]) / dt_phys
                                            : (3.0 * U_star[c][q] - 4.0 * U_n[c][q] + U_prev_[c][q]) / (2.0 * dt_phys);
                    const double r = R[c][q] - src;
                    R_star[c][q] = r;
                    if (std::abs(r) > lm) lm = std::abs(r);
                }
#pragma omp critical(fv_bdf2_max)
            if (lm > local_max) local_max = lm;
        }
        const double res_norm = global_norm(local_max);
        if (R0 < 0.0) R0 = res_norm;
        if (res_norm < std::max(inner_tol_rel_ * R0, kInnerResFloor)) {
            inner_iter_last_ = k + 1;
            break;
        }
        const double dt_pseudo = cfl * min_delta / (S + alpha * min_delta + kDtEps);
        axpy(U_star, U_star, dt_pseudo, R_star);
    }

    U_prev_ = U;
    first_step_ = false;
    U.swap(U_star);
    return {dt_phys, S_last};
}

std::unique_ptr<TimeScheme> make_time_scheme(TimeSchemeType type, double cfl, double dt_phys, int max_inner_iter,
                                             double inner_tol_rel) {
    switch (type) {
    case TimeSchemeType::SSPRK2: return std::make_unique<SSPRK2>(cfl);
    case TimeSchemeType::RungeKutta4: return std::make_unique<RungeKutta4>(cfl);
    case TimeSchemeType::BDF2DualTime:
        return std::make_unique<BDF2DualTime>(cfl, dt_phys, max_inner_iter, inner_tol_rel);
    default: return std::make_unique<ExplicitEuler>(cfl);
    }
}

}   // namespace fv
