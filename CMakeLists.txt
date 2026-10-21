# Top-level entry point: the presets (CMakePresets.json: release, debug), cpp/build.ps1 and
# IDEs (CLion, VS Code, Visual Studio) configure the repository root. The solver itself is
# defined in cpp/CMakeLists.txt.
cmake_minimum_required(VERSION 3.20)
project(Fenice LANGUAGES C CXX)

# Executables at the top of the build folder: build/release/fenice, build/release/fenice_tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()   # so ctest runs from the top-level build directory
add_subdirectory(cpp)
