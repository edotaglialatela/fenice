// Output and checkpoint I/O: VTK XML (.vtr/.vtm/.pvd)
// and a dependency-free NumPy .npy/.npz reader/writer for checkpoints.
#pragma once

#include <cstdint>
#include <functional>
#include <map>

#include "fv/Core.hpp"

namespace fv {

class AMRTree;

// ---------------------------------------------------------------------------
// VTK exporter
// ---------------------------------------------------------------------------

// One snapshot registered in the .pvd time series.
struct CollectionEntry {
    double time = 0.0;      // physical time [s]
    std::string filename;   // .vtr / .vtm file, relative to the output folder
};

// Writes VTK snapshots and the .pvd index; every file is written atomically (temp + rename).
class VTKExporter {
public:
    VTKExporter() = default;
    // Exporter writing into `output_dir` (created if missing); `binary`: inline base64
    // Float32 data arrays (about half the size of ASCII, faster to write), else ASCII text.
    VTKExporter(std::string output_dir, bool verbose, bool binary = true);

    // One .vtr snapshot (RectilinearGrid: Density, Pressure, Temperature_nondim, Mach, Velocity).
    void export_vtr(const std::string& prefix, const EOS& eos, const Field& U, const Mesh& mesh, double time, int step);
    // One .vtm multi-block snapshot of all AMR leaves (+ per-leaf .vtr in a subfolder). On `size` > 1
    // ranks every rank calls this: each writes the leaves it owns, `sync` is a barrier, and rank 0
    // writes the .vtm (and is the only one to record the snapshot in `collection`).
    void export_amr(const std::string& prefix, const EOS& eos, const AMRTree& tree, double time, int step, int rank = 0,
                    int size = 1, const std::function<void()>& sync = {});
    // .pvd time-series index of every snapshot written so far.
    void write_pvd(const std::string& prefix) const;

    // Output folder.
    const std::string& output_dir() const { return dir_; }
    std::vector<CollectionEntry> collection;   // snapshots written so far

private:
    std::string dir_;
    bool binary_ = true;
};

// ---------------------------------------------------------------------------
// NumPy .npy / .npz
// ---------------------------------------------------------------------------

// One NumPy array: dtype string, shape and raw little-endian bytes.
struct NpyArray {
    std::string descr;            // e.g. "<f8", "<i8", "<U64"
    std::vector<int64_t> shape;   // empty for a scalar
    std::vector<uint8_t> data;    // C-order element bytes

    // Number of elements.
    int64_t size() const;
    std::vector<double> as_doubles() const;   // numeric dtypes
    // First element as double (scalars).
    double scalar() const { return as_doubles().at(0); }
    std::vector<std::string> as_strings() const;   // '<U' dtypes

    // float64 array of the given shape.
    static NpyArray from_doubles(const std::vector<double>& v, std::vector<int64_t> shape);
    // float64 scalar.
    static NpyArray from_double(double v);
    // int64 scalar.
    static NpyArray from_int(int64_t v);
    // 1-D unicode array ('<U<width>').
    static NpyArray from_strings(const std::vector<std::string>& v);
};

// Arrays of an .npz file by key (file name without '.npy').
using NpzArchive = std::map<std::string, NpyArray>;

// Uncompressed (ZIP_STORED) archive, readable by numpy.load.
void write_npz(const std::string& path, const NpzArchive& arrays);
// Reads archives written by numpy.savez (stored entries, zip64 aware).
NpzArchive read_npz(const std::string& path);

// State field as an (n, ncomp) float64 array: ncomp = 4 ([rho, rho*u, rho*v, rho*E]) for
// 2-D fields, kNVar (+ rho*w) for 3-D ones.
NpyArray field_to_npy(const Field& U, int ncomp = 4);
// (n, 4) or (n, 5) array back to a state field (missing rho*w = 0); throws on other shapes.
Field npy_to_field(const NpyArray& a);

}   // namespace fv
