cmake_minimum_required(VERSION 3.20)
project(FVSolver LANGUAGES C CXX)   # C: required by FindMPI's C component

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(FV_USE_OPENMP "Multi-threaded kernels (OpenMP)" ON)
option(FV_USE_MPI    "MPI domain decomposition (needs an MPI SDK, e.g. MS-MPI)" ON)
option(FV_BUILD_TESTS "Build the C++ unit tests" ON)

add_library(fvcore STATIC
  src/Core.cpp
  src/Boundaries.cpp
  src/Schemes.cpp
  src/Physics.cpp
  src/TimeScheme.cpp
  src/SolutionMapping.cpp
  src/AMR.cpp
  src/IO.cpp
  src/Params.cpp
  src/TestCases.cpp
  src/MPIDecomposition.cpp
  src/Simulation.cpp
)
target_include_directories(fvcore PUBLIC include)
# Project root: logo, default results/ folder
get_filename_component(FV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
target_compile_definitions(fvcore PUBLIC FV_PROJECT_ROOT="${FV_ROOT}")

if(MSVC)
  # /fp:precise (no contraction / fast-math) keeps results bit-identical to the regression references
  target_compile_options(fvcore PUBLIC /W3 /utf-8 /fp:precise /permissive-)
  target_compile_definitions(fvcore PUBLIC NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(fvcore PUBLIC -Wall -Wextra -O3 -fno-fast-math -ffp-contract=off)
endif()

if(FV_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fvcore PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found: building single-threaded kernels")
  endif()
endif()

if(FV_USE_MPI)
  find_package(MPI COMPONENTS C)
  if(MPI_C_FOUND)
    target_link_libraries(fvcore PUBLIC MPI::MPI_C)
    target_compile_definitions(fvcore PUBLIC FV_USE_MPI)
    message(STATUS "MPI enabled: ${MPI_C_INCLUDE_DIRS}")
  else()
    message(WARNING "MPI SDK not found: building without MPI (install the MS-MPI SDK or set MSMPI_INC / MSMPI_LIB64)")
  endif()
endif()

add_executable(fvsolver apps/main.cpp)
target_link_libraries(fvsolver PRIVATE fvcore)

if(FV_BUILD_TESTS)
  enable_testing()
  add_executable(fv_tests tests/test_main.cpp)
  target_link_libraries(fv_tests PRIVATE fvcore)
  add_test(NAME fv_unit_tests COMMAND fv_tests)
endif()
