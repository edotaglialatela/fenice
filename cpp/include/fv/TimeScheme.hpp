// Time integration schemes.
#pragma once

#include <array>
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
enum class TimeSchemeType { ExplicitEuler, SSPRK2, SSPRK3 };

// Scheme from its GUI label ("SSP Runge-Kutta 2") or alias ("ssprk2").
TimeSchemeType parse_time_scheme(const std::string& s);

// Base class of the time integrators; dt = cfl * min(dx, dy) / S_max.
class TimeScheme {
public:
    // Integrator with CFL number `cfl`.
    explicit TimeScheme(double cfl) : cfl(cfl) {}
    virtual ~TimeScheme() = default;

    // Advance U in place by one adaptive (CFL) step, clamped to t_max.
    virtual StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max = 0.0) = 0;

    // Advance U in place by an imposed dt (AMR subcycling); returns S_max of
    // the first stage.  Default: first-order Euler.
    virtual double advance_fixed_dt(Field& U, const ResidualFn& res, double dt);

    // What advance_fixed_dt does, stage by stage: U_new = U + dt * sum_i weights[i] * R_i. The cell-based AMR uses
    // the weights to accumulate the fine fluxes seen by a coarse neighbour. The base class is forward Euler.
    virtual std::vector<double> stage_weights() const { return {1.0}; }
    // Time of the residual R_i evaluated by advance_fixed_dt: t + offsets[i] * dt (0, 1/2 or 1; the stage time
    // is t, t + 0.5 * dt or t + dt exactly as in step()).
    virtual std::vector<double> stage_offsets() const { return {0.0}; }

    // The work fields of the scheme (stage residuals, stage state) exchanged with `w`. A caller that advances fields of several
    // sizes in turn (the levels of an adaptive mesh) keeps one set per size: with a single set every change of size would
    // resize the fields, and growing one fills it with zeros on one thread.
    void swap_work(std::array<Field, 4>& w) {
        R1_.swap(w[0]);
        R2_.swap(w[1]);
        R3_.swap(w[2]);
        tmp_.swap(w[3]);
    }

    double cfl;   // CFL number

    // Absolute time of the stage whose residual is being evaluated (set by every step() before each call of the
    // residual: t, t + dt, ... of the Runge-Kutta stages). Only the time-dependent source terms
    // read it; the fixed-dt path of the AMR subcycling does not set it.
    double stage_time = 0.0;

protected:
    Field R1_, R2_, R3_, tmp_;
};

// First-order forward Euler: U^{n+1} = U^n + dt R(U^n).
class ExplicitEuler : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max = 0.0) override;
    double advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
};

// Two-stage strong-stability-preserving Runge-Kutta (Shu-Osher form), 2nd order.
class SSPRK2 : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max = 0.0) override;
    double advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
    std::vector<double> stage_weights() const override { return {0.5, 0.5}; }
    std::vector<double> stage_offsets() const override { return {0.0, 1.0}; }
};

// Three-stage strong-stability-preserving Runge-Kutta (Shu-Osher form), 3rd order, CFL coefficient 1. Stage times t,
// t + dt, t + dt / 2.
class SSPRK3 : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max = 0.0) override;
    double advance_fixed_dt(Field& U, const ResidualFn& res, double dt) override;
    std::vector<double> stage_weights() const override { return {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}; }
    std::vector<double> stage_offsets() const override { return {0.0, 1.0, 0.5}; }
};

// Factory: the integrator for `type`.
std::unique_ptr<TimeScheme> make_time_scheme(TimeSchemeType type, double cfl);

}   // namespace fv
