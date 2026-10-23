cmake_minimum_required(VERSION 3.20)
project(Fenice LANGUAGES C CXX)   # C: required by FindMPI's C component

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(FV_USE_OPENMP "Multi-threaded kernels (OpenMP)" ON)
option(FV_USE_MPI    "MPI domain decomposition (needs an MPI SDK, e.g. MS-MPI)" ON)
option(FV_BUILD_TESTS "Build the C++ unit tests" ON)

add_library(fenice_core STATIC
  src/CaseFile.cpp
  src/Core.cpp
  src/Geometry.cpp
  src/Boundaries.cpp
  src/Schemes.cpp
  src/Physics.cpp
  src/TimeScheme.cpp
  src/SolutionMapping.cpp
  src/CellMesh.cpp
  src/CellFlux.cpp
  src/CellAdvance.cpp
  src/CellRemesh.cpp
  src/CellParallel.cpp
  src/IO.cpp
  src/Params.cpp
  src/TestCases.cpp
  src/MPIDecomposition.cpp
  src/Simulation.cpp
)
target_include_directories(fenice_core PUBLIC include)
target_include_directories(fenice_core PRIVATE third_party/tomlplusplus)

# TinyExpr (case-file expressions): C, built with the same strict floating point as the solver
add_library(fenice_tinyexpr STATIC third_party/tinyexpr/tinyexpr.c)
target_include_directories(fenice_tinyexpr PUBLIC third_party/tinyexpr)
if(MSVC)
  target_compile_options(fenice_tinyexpr PRIVATE /fp:precise /W0)
else()
  target_compile_options(fenice_tinyexpr PRIVATE -fno-fast-math -ffp-contract=off -w)
endif()
# Usual maths precedence: -a^2 = -(a^2) and a^b^c = a^(b^c) (TinyExpr's default would make
# exp(-x^2) an exp(+x^2)); a compile switch of the vendored, unmodified source
target_compile_definitions(fenice_tinyexpr PRIVATE TE_POW_FROM_RIGHT)
target_link_libraries(fenice_core PUBLIC fenice_tinyexpr)
# Project root: logo, cases/ folder, default results/ folder
get_filename_component(FV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
target_compile_definitions(fenice_core PUBLIC FV_PROJECT_ROOT="${FV_ROOT}")

if(MSVC)
  # /fp:precise (no contraction / fast-math) keeps results bit-identical to the regression references
  target_compile_options(fenice_core PUBLIC /W3 /utf-8 /fp:precise /permissive-)
  target_compile_definitions(fenice_core PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
  # Optimisation comes from the build type (Release: -O3), so Debug builds stay debuggable
  target_compile_options(fenice_core PUBLIC -Wall -Wextra -fno-fast-math -ffp-contract=off)
endif()

if(FV_USE_OPENMP)
  if(APPLE AND NOT DEFINED OpenMP_CXX_FLAGS)
    # Apple clang ships no OpenMP runtime: use Homebrew's libomp when it is installed
    # (explicit -DOpenMP_CXX_FLAGS=... on the command line still take precedence)
    find_program(FV_BREW brew)
    if(FV_BREW)
      execute_process(COMMAND ${FV_BREW} --prefix libomp OUTPUT_VARIABLE FV_LIBOMP
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(FV_LIBOMP AND EXISTS "${FV_LIBOMP}/lib/libomp.dylib")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${FV_LIBOMP}/include" CACHE STRING "OpenMP flags")
      set(OpenMP_CXX_LIB_NAMES omp CACHE STRING "OpenMP libraries")
      set(OpenMP_omp_LIBRARY "${FV_LIBOMP}/lib/libomp.dylib" CACHE FILEPATH "Homebrew libomp")
    endif()
  endif()
  find_package(OpenMP COMPONENTS CXX)   # C++ only: no misleading "OpenMP_C not found"
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fenice_core PUBLIC OpenMP::OpenMP_CXX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # MinGW: FindOpenMP lists no OpenMP libraries (the -fopenmp driver flag pulls in libgomp
      # itself), so the flag must also reach the link step or every omp_*/GOMP_* is undefined
      target_link_options(fenice_core PUBLIC -fopenmp)
    endif()
  else()
    message(WARNING "OpenMP not found: building single-threaded kernels (macOS: brew install libomp)")
  endif()
endif()

if(FV_USE_MPI)
  find_package(MPI COMPONENTS C)
  if(MPI_C_FOUND)
    target_link_libraries(fenice_core PUBLIC MPI::MPI_C)
    # Only the C API is used: keep mpi.h from pulling in the deprecated C++ bindings, which
    # Open MPI 4 / MPICH would then need libmpi_cxx for (undefined MPI::Comm::Comm() on Ubuntu 24.04)
    target_compile_definitions(fenice_core PUBLIC FV_USE_MPI OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
    message(STATUS "MPI enabled: ${MPI_C_INCLUDE_DIRS}")
  else()
    message(WARNING "MPI SDK not found: building without MPI (install the MS-MPI SDK or set MSMPI_INC / MSMPI_LIB64)")
  endif()
endif()

add_executable(fenice apps/main.cpp)
target_link_libraries(fenice PRIVATE fenice_core)

# Wall-time gate of the cell-based AMR (residual cost per cell, face engine vs uniform)
add_executable(bench_cellflux apps/bench_cellflux.cpp)
target_link_libraries(bench_cellflux PRIVATE fenice_core)

# Cost and bit fingerprint of the Riemann solvers
add_executable(bench_riemann apps/bench_riemann.cpp)
target_link_libraries(bench_riemann PRIVATE fenice_core)

if(FV_BUILD_TESTS)
  enable_testing()
  add_executable(fenice_tests tests/test_main.cpp)
  target_link_libraries(fenice_tests PRIVATE fenice_core)
  add_test(NAME fenice_unit_tests COMMAND fenice_tests)
endif()
