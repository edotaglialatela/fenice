# Trixi.jl side of compare_trixi.py: the two benchmark setups, parametrised, timed after a warm-up run.
#
#   julia --project=<env> --threads=N trixi_case.jl vortex <level> <polydeg> <outdir>
#   julia --project=<env> --threads=N trixi_case.jl tgv    <level> <polydeg> <outdir>
#
# The setups are those of the Trixi.jl examples elixir_euler_vortex.jl (tree_2d_dgsem) and
# elixir_navierstokes_taylor_green_vortex.jl (tree_3d_dgsem), unchanged except for the refinement level, the polynomial
# degree and the output. The last stdout line is "RESULT {json}"; compare_trixi.py parses it.
using Trixi
using OrdinaryDiffEqLowStorageRK
using Printf

function vortex(level, polydeg, tend)
    equations = CompressibleEulerEquations2D(1.4)
    function ic(x, t, equations::CompressibleEulerEquations2D)
        # Shu (1997) isentropic vortex, as in the example (the centre does not move: exact only at t = 20 k)
        inicenter = SVector(0, 0)
        iniamplitude = 5
        rho = 1
        vel = SVector(1, 1)
        p = 25.0
        rt = p / rho
        cent = SVector(x[1] - inicenter[1], x[2] - inicenter[2])
        cent = SVector(-cent[2], cent[1])
        r2 = cent[1]^2 + cent[2]^2
        du = iniamplitude / (2 * pi) * exp(0.5 * (1 - r2))
        dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2
        rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
        vel = vel + du * cent
        p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
        return prim2cons(SVector(rho, vel[1], vel[2], p), equations)
    end
    solver = DGSEM(polydeg = polydeg, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))
    mesh = TreeMesh((-10.0, -10.0), (10.0, 10.0), initial_refinement_level = level, n_cells_max = 10^7,
                    periodicity = true)
    semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver; boundary_conditions = boundary_condition_periodic)
    ode = semidiscretize(semi, (0.0, tend))
    analysis_callback = AnalysisCallback(semi, interval = 10^9)   # only the final analysis
    callbacks = CallbackSet(analysis_callback, StepsizeCallback(cfl = 1.1))
    t = @elapsed sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
                              dt = 1.0, ode_default_options()..., callback = callbacks)
    return sol, analysis_callback, t, (nx = 2^level * (polydeg + 1),)
end

function tgv(level, polydeg, tend, outdir)
    prandtl = 0.72
    mu = 6.25e-4   # Re = 1600
    equations = CompressibleEulerEquations3D(1.4)
    equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu, Prandtl = prandtl,
                                                              gradient_variables = GradientVariablesPrimitive())
    function ic(x, t, equations::CompressibleEulerEquations3D)
        A = 1.0
        Ms = 0.1
        rho = 1.0
        v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
        v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
        p = (A / Ms)^2 * rho / equations.gamma
        p = p + 1.0 / 16.0 * A^2 * rho * (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
                                           cos(2 * x[2]) * cos(2 * x[3]))
        return prim2cons(SVector(rho, v1, v2, 0.0, p), equations)
    end
    solver = DGSEM(polydeg = polydeg, surface_flux = flux_hllc,
                   volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
    mesh = TreeMesh((-1.0, -1.0, -1.0) .* pi, (1.0, 1.0, 1.0) .* pi, initial_refinement_level = level,
                    n_cells_max = 10^7, periodicity = true)
    semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), ic, solver;
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
    ode = semidiscretize(semi, (0.0, tend))
    # the analysis integrals (kinetic energy, enstrophy) go to <outdir>/analysis.dat every `interval` steps
    analysis_callback = AnalysisCallback(semi, interval = 20, save_analysis = true, output_directory = outdir,
                                         analysis_filename = "analysis.dat",
                                         extra_analysis_integrals = (energy_kinetic, enstrophy))
    callbacks = CallbackSet(analysis_callback)
    t = @elapsed sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.Threaded()); abstol = 1e-8, reltol = 1e-8,
                              ode_default_options()..., callback = callbacks)
    return sol, analysis_callback, t, (nx = 2^level * (polydeg + 1),)
end

function main()
    kind, level, polydeg, outdir = ARGS[1], parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4]
    mkpath(outdir)
    if kind == "vortex"
        vortex(level, polydeg, 0.2)              # warm-up: compiles every method
        sol, ac, t, info = vortex(level, polydeg, 20.0)
        l2, linf = ac(sol)                        # the error of the final state against the initial condition
        res = Dict("kind" => kind, "level" => level, "polydeg" => polydeg, "nx" => info.nx, "wall" => t,
                   "l2_rho" => l2[1], "linf_rho" => linf[1], "threads" => Threads.nthreads(),
                   "steps" => length(sol.t) - 1, "trixi" => string(pkgversion(Trixi)))
    else
        tgv(level, polydeg, 0.2, outdir)
        sol, ac, t, info = tgv(level, polydeg, 20.0, outdir)
        res = Dict("kind" => kind, "level" => level, "polydeg" => polydeg, "nx" => info.nx, "wall" => t,
                   "threads" => Threads.nthreads(), "analysis" => joinpath(outdir, "analysis.dat"),
                   "trixi" => string(pkgversion(Trixi)))
    end
    println("RESULT {", join(("\"$k\": " * (v isa AbstractString ? "\"$v\"" : string(v)) for (k, v) in res), ", "), "}")
end

main()
