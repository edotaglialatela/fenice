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
enum class TimeSchemeType { ExplicitEuler, SSPRK2, SSPRK3, MusclHancock };

// Scheme from its GUI label ("SSP Runge-Kutta 2") or alias ("ssprk2").
TimeSchemeType parse_time_scheme(const std::string& s);

// Opt-in repair of a stage that leaves the admissible set (positivity_fallback). After each stage `mark` scans the new state and
// adds its cells with density or pressure <= 0 to the set the residual treats with first-order faces; it returns true if the set
// grew. The stage is then recomputed (the residual reads the set), and scanned again, until nothing new is found; `clear` empties
// the set before the next stage. A scheme without a guard runs the code it always had.
struct StageGuard {
    std::function<bool(const Field& U)> mark;
    std::function<void()> clear;
};

// Base class of the time integrators; dt = cfl * min(dx, dy) / S_max.
class TimeScheme {
public:
    // Install (or remove, with nullptr) the guard of the stages; it must outlive the scheme's steps.
    void set_guard(const StageGuard* g) { guard_ = g; }

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
    void swap_work(std::array<Field, 5>& w) {
        R1_.swap(w[0]);
        R2_.swap(w[1]);
        R3_.swap(w[2]);
        tmp_.swap(w[3]);
        g_.swap(w[4]);
    }

    double cfl;   // CFL number

    // Absolute time of the stage whose residual is being evaluated (set by every step() before each call of the
    // residual: t, t + dt, ... of the Runge-Kutta stages). Only the time-dependent source terms
    // read it; the fixed-dt path of the AMR subcycling does not set it.
    double stage_time = 0.0;

    // Half of the time step the residual about to be evaluated belongs to, when it must advance its face states by it before the Riemann
    // solves (the predictor of MusclHancock); 0 for every other scheme and stage, which then run the code they always had.
    double predictor_half_dt = 0.0;

    // The wave speed the next step's dt is taken from when it is not a function of the state alone (MusclHancock: the previous step's), 0
    // for a scheme that takes it from its own residual. A checkpoint carries it (key S), so a resumed run takes the steps it would have taken.
    virtual double lagged_speed() const { return 0.0; }
    virtual void set_lagged_speed(double) {}

protected:
    // Repeat a guarded stage while its result has new non-positive cells (at most kMaxRepairs times); `again` recomputes the
    // candidate `cand` of the stage with the marked cells in force. The set is emptied at the end.
    void repair(const Field& cand, const std::function<void()>& again);
    static constexpr int kMaxRepairs = 8;

    Field R1_, R2_, R3_, tmp_;
    Field g_;                            // candidate of a guarded stage (the stage's inputs stay intact for the repeat)
    const StageGuard* guard_ = nullptr;
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

// MUSCL-Hancock: one residual per step. The residual advances the limited face states of every cell by half a step with the cell's own
// flux difference (predictor, `predictor_half_dt`), solves the Riemann problems on those, and U^{n+1} = U^n + dt R. The step is known
// before the residual runs, so it comes from the wave speed of the previous step (the first step seeds it with one plain residual);
// a step is therefore not a function of the state alone, so the speed is part of a checkpoint (lagged_speed). On the adaptive mesh the step
// is an imposed one (advance_fixed_dt: the residual of a level at t + dt / 2, weight 1) and the mesh's own wave-speed bound plays the lag.
class MusclHancock : public TimeScheme {
public:
    using TimeScheme::TimeScheme;
    StepResult step(Field& U, const ResidualFn& res, double min_delta, double t, double t_max, double dt_max = 0.0) override;
    std::vector<double> stage_offsets() const override { return {0.5}; }
    double lagged_speed() const override { return s_prev_; }
    void set_lagged_speed(double s) override { s_prev_ = s; }

private:
    double s_prev_ = 0.0;   // wave speed of the previous step (0: none yet)
};

// Factory: the integrator for `type`.
std::unique_ptr<TimeScheme> make_time_scheme(TimeSchemeType type, double cfl);

}   // namespace fv
