// Time integration schemes.
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "fv/Core.hpp"

namespace fv {

// R = residual(U); returns the maximum wave speed S_max used for the CFL dt.
using ResidualFn = std::function<double(const Field& U, Field& R)>;

// Outcome of one time step.
struct StepResult {
    double dt = 0.0;      // time step taken [s]
    double s_max = 0.0;   // maximum wave speed seen [m/s]
};

// Available time integrators.
enum class TimeSchemeType { ExplicitEuler, SSPRK2, RungeKutta4, BDF2DualTime };

// Scheme from its GUI label ("SSP Runge-Kutta 2") or alias ("ssprk2").
TimeSchemeType parse_time_scheme(const std::string& s);

// Base class of the time integrators; dt = cfl * min(dx, dy) / S_max.
class TimeScheme {
public:
    // Integrator with CFL number `cfl`.
    explicit TimeScheme(double cfl) : cfl(cfl) {}
    virtual ~TimeScheme() = default;

    // Advance U in place by one adaptive (CFL) step, clamped to t_max.
    virtual StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) = 0;

    // Advance U in place by an imposed dt (AMR subcycling); returns S_max of
    // the first stage.  Default: first-order Euler.
    virtual double advance_fixed_dt(Field& U, const ResidualFn& res, double dt);

    // What advance_fixed_dt does, stage by stage: U_new = U + dt * sum_i weights[i] * R_i, and R_i is evaluated at
    // time t + offsets[i] * dt (offsets are 0, 1/2 or 1; the stage time is t + 0.5 * dt, t + dt as in step()).
    // The cell-based AMR uses them to accumulate the fine fluxes seen by a coarse neighbour. The base class is
    // forward Euler.
    virtual std::vector<double> stage_weights() const { return {1.0}; }
    virtual std::vector<double> stage_offsets() const { return {0.0}; }

    // Inner pseudo-time iterations of the last step (BDF2 only), else -1.
    virtual int inner_iter_last() const { return -1; }

    double cfl;   // CFL number

    // Absolute time of the stage whose residual is being evaluated (set by every step() before each call of the
    // residual: t, t + dt, ... of the Runge-Kutta stages; t + dt for BDF2). Only the time-dependent source terms
    // read it; the fixed-dt path of the AMR subcycling does not set it.
    double stage_time = 0.0;

protected:
    Field R1_, R2_, R3_, R4_, tmp_;
};

// First-order forward Euler: U^{n+1} = U^n + dt R(U^n).
class ExplicitEuler : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    double advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
};

// Two-stage strong-stability-preserving Runge-Kutta (Shu-Osher form), 2nd order.
class SSPRK2 : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    double advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
    std::vector<double> stage_weights() const override { return {0.5, 0.5}; }
    std::vector<double> stage_offsets() const override { return {0.0, 1.0}; }
};

// Classical four-stage Runge-Kutta, 4th order.
class RungeKutta4 : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    double advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
    std::vector<double> stage_weights() const override { return {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0}; }
    std::vector<double> stage_offsets() const override { return {0.0, 0.5, 0.5, 1.0}; }
};

// BDF2 physical time stepping with an explicit-Euler pseudo-time inner loop.
class BDF2DualTime : public TimeScheme {
public:
    // Fixed physical step `dt_phys`; the inner loop stops after `max_inner_iter` iterations
    // or when the modified residual drops by `inner_tol_rel`.  Throws if dt_phys <= 0.
    BDF2DualTime(double cfl, double dt_phys, int max_inner_iter = 50, double inner_tol_rel = 1e-3);
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    int inner_iter_last() const override { return inner_iter_last_; }

    // Global max across MPI ranks (identity in serial).
    std::function<double(double)> global_norm = [](double x) { return x; };

private:
    double dt_phys_;
    int max_inner_iter_;
    double inner_tol_rel_;
    Field U_prev_;
    bool first_step_ = true;
    int inner_iter_last_ = 0;
};

// Factory: the integrator for `type` (BDF2 uses dt_phys / max_inner_iter / inner_tol_rel).
std::unique_ptr<TimeScheme> make_time_scheme(TimeSchemeType type, double cfl, double dt_phys, int max_inner_iter,
                                             double inner_tol_rel);

}   // namespace fv
