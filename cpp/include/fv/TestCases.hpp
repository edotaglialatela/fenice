// Built-in benchmark cases.
#pragma once

#include <functional>

#include "fv/Boundaries.hpp"
#include "fv/Params.hpp"

namespace fv {

// A benchmark: default parameters, initial field and optional forced BCs.
struct TestCase {
    std::string key;                 // registry key, e.g. "Sod Shock Tube"
    bool        requires_viscosity = false;   // warn when run without Navier-Stokes
    bool        three_d = false;   // initialize() expects a 3-D mesh (otherwise 2-D, extruded along z)
    Params      defaults;            // case defaults (override baseline_defaults)
    // Initial conservative field on `mesh`; `mu` is the run's viscosity.
    std::function<Field(const Mesh& mesh, const EOS& eos, double mu)> initialize;
    // Optional: force case-specific BCs (overrides the configured ones).
    std::function<void(BoundaryManager& bm, bool verbose)> configure_boundaries;
};

// All built-in cases, in GUI order.
const std::vector<TestCase>& test_registry();
const TestCase&              get_test_case(const std::string& key);   // throws if unknown

// Exact Taylor-Green vortex [rho, u, v, p] on the periodic square [0, L]^2 (k = 2 pi / L,
// U0 = 1, rho0 = 1, p0 = 100), decaying as exp(-2 nu k^2 t) with nu = mu / rho0.
State taylor_green_exact(double x, double y, double t, double L, double mu);

// Initial 3-D Taylor-Green vortex [rho, u, v, p, w] on [0, 2 pi]^3 (U0 = 1, rho0 = 1, Ma = 0.1):
// u = sin x cos y cos z, v = -cos x sin y cos z, w = 0.
State taylor_green_3d_initial(double x, double y, double z, double gamma);

// Exact isentropic vortex [rho, u, v, p] (Shu 1998: strength 5, free stream (1, 1)) on the
// periodic [0, Lx] x [0, Ly], centred at the domain centre at t = 0 and advected unchanged.
State isentropic_vortex_exact(double x, double y, double t, double Lx, double Ly, double gamma);

}  // namespace fv
