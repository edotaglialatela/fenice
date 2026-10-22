// Core data types, Cartesian mesh and equations of state.
//
// Conservative state U = [rho, rho*u, rho*v, rho*E, rho*w], primitive W = [rho, u, v, p, w].
// The z-momentum comes last so that 2-D runs (nz = 1, w = 0) keep the
// [rho, rho*u, rho*v, rho*E] layout in components 0-3.
// Cell indexing is row-major: cell = (k*ny + j)*nx + i (z outer, x inner).
// Face ordering: x-normal faces (k*ny + j)*(nx+1) + i first, then y-normal faces
// (k*(ny+1) + j)*nx + i, then (3-D only) z-normal faces (k*ny + j)*nx + i.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace fv {

// Number of conservative variables.
constexpr int kNVar = 5;
// One cell state: conservative [rho, rho*u, rho*v, rho*E, rho*w] or primitive [rho, u, v, p, w].
using State = std::array<double, kNVar>;
// A state per cell (row-major, cell = (k*ny + j)*nx + i) or per face.
using Field = std::vector<State>;
// Component of the momentum (conservative) / velocity (primitive) along axis 0, 1, 2.
constexpr int kMom[3] = {1, 2, 4};

// Loops shorter than this run single-threaded (OpenMP fork/join would cost
// more than the work, e.g. on small AMR patches).
constexpr int kOmpMinWork = 4096;

// Cardinal faces of a (sub)domain; Front / Back are the z-min / z-max sides (3-D only).
enum class Face { Left = 0, Right = 1, Bottom = 2, Top = 3, Front = 4, Back = 5 };
// The four in-plane faces, in enum order (2-D domains, AMR patches).
constexpr std::array<Face, 4> kAllFaces = {Face::Left, Face::Right, Face::Bottom, Face::Top};
// All six faces of a 3-D domain.
constexpr std::array<Face, 6> kAllFaces3D = {Face::Left, Face::Right, Face::Bottom, Face::Top, Face::Front, Face::Back};

// Face on the other side of a cell / patch (Left <-> Right, Bottom <-> Top, Front <-> Back).
inline Face opposite(Face f) {
    switch (f) {
    case Face::Left: return Face::Right;
    case Face::Right: return Face::Left;
    case Face::Bottom: return Face::Top;
    case Face::Top: return Face::Bottom;
    case Face::Front: return Face::Back;
    default: return Face::Front;
    }
}
// Axis normal to a face: 0 (x) for Left/Right, 1 (y) for Bottom/Top, 2 (z) for Front/Back.
inline int face_axis(Face f) { return static_cast<int>(f) / 2; }

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

// Uniform Cartesian mesh.  Topology is implicit (structured), so no per-face
// arrays are stored; (x0, y0, z0) is the global origin, non-zero for AMR patches
// and MPI sub-domains.  nz = 1 is a 2-D mesh: no z-faces, areas and volumes in 2-D.
struct Mesh {
    int nx = 0, ny = 0, nz = 1;            // cells in x, y, z
    double Lx = 0.0, Ly = 0.0, Lz = 1.0;   // physical extent
    double dx = 0.0, dy = 0.0, dz = 1.0;   // cell size (Lx/nx, Ly/ny, Lz/nz)
    double x0 = 0.0, y0 = 0.0, z0 = 0.0;   // global coordinates of the lower corner

    // True for a 3-D mesh (nz > 1).
    bool is3d() const { return nz > 1; }
    // Number of cells, nx*ny*nz.
    int num_cells() const { return nx * ny * nz; }
    // Number of x-normal faces, (nx+1)*ny*nz; they come first in face order.
    int num_xfaces() const { return (nx + 1) * ny * nz; }
    // Number of y-normal faces, nx*(ny+1)*nz.
    int num_yfaces() const { return nx * (ny + 1) * nz; }
    // Number of z-normal faces: nx*ny*(nz+1) in 3-D, none in 2-D.
    int num_zfaces() const { return is3d() ? nx * ny * (nz + 1) : 0; }
    // Total faces.
    int num_faces() const { return num_xfaces() + num_yfaces() + num_zfaces(); }
    // Cell area dx*dy of a 2-D cell.
    double cell_area() const { return dx * dy; }
    // Cell volume: dx*dy*dz in 3-D, dx*dy (per unit depth) in 2-D.
    double cell_volume() const { return is3d() ? dx * dy * dz : dx * dy; }
    // Face area for normal `axis`: dy*dz, dx*dz, dx*dy in 3-D; dy, dx in 2-D.
    double face_area(int axis) const {
        if (!is3d()) return axis == 0 ? dy : dx;
        return axis == 0 ? dy * dz : (axis == 1 ? dx * dz : dx * dy);
    }
    // Normal axis (0, 1, 2) of face f.
    int face_axis(int f) const { return f < num_xfaces() ? 0 : (f < num_xfaces() + num_yfaces() ? 1 : 2); }
    // Global x of the centre of column i.
    double xc(int i) const { return (i + 0.5) * dx + x0; }
    // Global y of the centre of row j.
    double yc(int j) const { return (j + 0.5) * dy + y0; }
    // Global z of the centre of layer k.
    double zc(int k) const { return (k + 0.5) * dz + z0; }

    // Cell index of (i, j, k).
    int cell(int i, int j, int k = 0) const { return (k * ny + j) * nx + i; }
    // Index of the x-face left of cell (i, j, k) (i = nx: right boundary).
    int xface(int i, int j, int k = 0) const { return (k * ny + j) * (nx + 1) + i; }
    // Index of the y-face below cell (i, j, k) (j = ny: top boundary).
    int yface(int i, int j, int k = 0) const { return num_xfaces() + (k * (ny + 1) + j) * nx + i; }
    // Index of the z-face in front of cell (i, j, k) (k = nz: back boundary); 3-D only.
    int zface(int i, int j, int k) const { return num_xfaces() + num_yfaces() + (k * ny + j) * nx + i; }

    // Global index of the k-th boundary face on side `b`.  k runs over the two
    // transverse directions: kz*ny + j for left/right, kz*nx + i for bottom/top,
    // j*nx + i for front/back (in 2-D: row j, column i).
    int boundary_face(Face b, int k) const {
        switch (b) {
        case Face::Left: return k * (nx + 1);
        case Face::Right: return k * (nx + 1) + nx;
        case Face::Bottom: return num_xfaces() + (k / nx) * (ny + 1) * nx + k % nx;
        case Face::Top: return num_xfaces() + ((k / nx) * (ny + 1) + ny) * nx + k % nx;
        case Face::Front: return num_xfaces() + num_yfaces() + k;
        default: return num_xfaces() + num_yfaces() + nz * ny * nx + k;
        }
    }
    // Interior cell adjacent to the k-th boundary face on side `b`.
    int boundary_cell(Face b, int k) const {
        switch (b) {
        case Face::Left: return k * nx;
        case Face::Right: return k * nx + nx - 1;
        case Face::Bottom: return (k / nx) * ny * nx + k % nx;
        case Face::Top: return ((k / nx) * ny + ny - 1) * nx + k % nx;
        case Face::Front: return k;
        default: return (nz - 1) * ny * nx + k;
        }
    }
    // Number of boundary faces on side b (0 for front/back in 2-D).
    int boundary_count(Face b) const {
        switch (b) {
        case Face::Left:
        case Face::Right: return ny * nz;
        case Face::Bottom:
        case Face::Top: return nx * nz;
        default: return is3d() ? nx * ny : 0;
        }
    }
};

// Uniform nx x ny (2-D) mesh of [x0, x0+Lx] x [y0, y0+Ly].
Mesh generate_cartesian_mesh(double Lx, double Ly, int nx, int ny, double x0 = 0.0, double y0 = 0.0);
// Uniform nx x ny x nz mesh of [x0, x0+Lx] x [y0, y0+Ly] x [z0, z0+Lz] (nz = 1: 2-D).
Mesh generate_cartesian_mesh_3d(double Lx, double Ly, double Lz, int nx, int ny, int nz, double x0 = 0.0,
                                double y0 = 0.0, double z0 = 0.0);

// ---------------------------------------------------------------------------
// Equations of state
// ---------------------------------------------------------------------------

// Numerical floor applied to density and pressure.
constexpr double kEosEps = 1e-10;

// Equation of state: calorically perfect gas, p = (gamma-1) rho e, or stiffened
// gas, p + p_inf = (gamma-1) rho e (p_inf = 0 recovers the ideal gas).
struct EOS {
    enum class Type { IdealGas, StiffenedGas };
    Type type = Type::IdealGas;
    double gamma = 1.4;   // heat capacity ratio
    double p_inf = 0.0;   // stiffness pressure [Pa] (stiffened gas only)

    // Ideal gas (default: air, gamma = 1.4).
    static EOS ideal(double g = 1.4) { return EOS{Type::IdealGas, g, 0.0}; }
    // Stiffened gas (default: liquid water, gamma = 4.4, p_inf = 6e8 Pa).
    static EOS stiffened(double g = 4.4, double pinf = 6.0e8) { return EOS{Type::StiffenedGas, g, pinf}; }

    // Primitive [rho, u, v, p, w] -> conservative [rho, rho*u, rho*v, rho*E, rho*w].
    State prim_to_cons(const State& W) const {
        const double rho = W[0], u = W[1], v = W[2], p = W[3], w = W[4];
        const double ke = 0.5 * rho * (u * u + v * v + w * w);
        const double e = (type == Type::IdealGas) ? p / (gamma - 1.0) : (p + gamma * p_inf) / (gamma - 1.0);
        return {rho, rho * u, rho * v, e + ke, rho * w};
    }

    // Conservative -> primitive; density and pressure are floored at kEosEps.
    State cons_to_prim(const State& U) const {
        const double rho = std::max(U[0], kEosEps);
        const double u = U[1] / rho, v = U[2] / rho, w = U[4] / rho;
        double p;
        if (type == Type::IdealGas) {
            p = (gamma - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v + w * w));
        } else {
            const double e_kin = 0.5 * rho * (u * u + v * v + w * w);
            p = (gamma - 1.0) * (U[3] - e_kin) - gamma * p_inf;
        }
        return {rho, u, v, std::max(p, kEosEps), w};
    }

    // Isentropic speed of sound: sqrt(gamma p / rho) or sqrt(gamma (p + p_inf) / rho).
    double sound_speed(double p, double rho) const {
        if (type == Type::IdealGas) return std::sqrt(gamma * std::max(p, kEosEps) / std::max(rho, kEosEps));
        return std::sqrt(gamma * (p + p_inf) / std::max(rho, kEosEps));
    }
};

// Field-wide conversions (OpenMP-parallel).
void cons_to_prim(const EOS& eos, const Field& U, Field& W);
// Field-wide primitive -> conservative conversion (OpenMP-parallel).
void prim_to_cons(const EOS& eos, const Field& W, Field& U);

}   // namespace fv
