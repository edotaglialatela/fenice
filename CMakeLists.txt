# Top-level entry point, so IDEs (CLion, VS Code, Visual Studio) can open the repository
# root as a CMake project. The solver itself is defined in cpp/CMakeLists.txt, which can
# also be configured on its own (cmake -S cpp ..., as build.ps1 and the README do).
cmake_minimum_required(VERSION 3.20)
project(Fenice LANGUAGES C CXX)

enable_testing()   # so ctest runs from the top-level build directory
add_subdirectory(cpp)
