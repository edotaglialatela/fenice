// Time integration schemes.
#pragma once

#include <functional>
#include <memory>

#include "fv/Core.hpp"

namespace fv {

// R = residual(U); returns the maximum wave speed S_max used for the CFL dt.
using ResidualFn = std::function<double(const Field& U, Field& R)>;

// Outcome of one time step.
struct StepResult {
    double dt    = 0.0;   // time step taken [s]
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

    // Inner pseudo-time iterations of the last step (BDF2 only), else -1.
    virtual int inner_iter_last() const { return -1; }

    // Independent copy (own scratch buffers) for concurrent AMR patch updates.
    virtual std::unique_ptr<TimeScheme> clone() const = 0;

    double cfl;   // CFL number

protected:
    Field R1_, R2_, R3_, R4_, tmp_;
};

// First-order forward Euler: U^{n+1} = U^n + dt R(U^n).
class ExplicitEuler : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    std::unique_ptr<TimeScheme> clone() const override { return std::make_unique<ExplicitEuler>(*this); }
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    double     advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
};

// Two-stage strong-stability-preserving Runge-Kutta (Shu-Osher form), 2nd order.
class SSPRK2 : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    std::unique_ptr<TimeScheme> clone() const override { return std::make_unique<SSPRK2>(*this); }
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    double     advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
};

// Classical four-stage Runge-Kutta, 4th order.
class RungeKutta4 : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    std::unique_ptr<TimeScheme> clone() const override { return std::make_unique<RungeKutta4>(*this); }
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    double     advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
};

// BDF2 physical time stepping with an explicit-Euler pseudo-time inner loop.
class BDF2DualTime : public TimeScheme {
public:
    // Fixed physical step `dt_phys`; the inner loop stops after `max_inner_iter` iterations
    // or when the modified residual drops by `inner_tol_rel`.  Throws if dt_phys <= 0.
    BDF2DualTime(double cfl, double dt_phys, int max_inner_iter = 50, double inner_tol_rel = 1e-3);
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max) override;
    int        inner_iter_last() const override { return inner_iter_last_; }
    std::unique_ptr<TimeScheme> clone() const override { return std::make_unique<BDF2DualTime>(*this); }

    // Global max across MPI ranks (identity in serial).
    std::function<double(double)> global_norm = [](double x) { return x; };

private:
    double dt_phys_;
    int    max_inner_iter_;
    double inner_tol_rel_;
    Field  U_prev_;
    bool   first_step_ = true;
    int    inner_iter_last_ = 0;
};

// Factory: the integrator for `type` (BDF2 uses dt_phys / max_inner_iter / inner_tol_rel).
std::unique_ptr<TimeScheme> make_time_scheme(TimeSchemeType type, double cfl, double dt_phys,
                                             int max_inner_iter, double inner_tol_rel);

}  // namespace fv
