// Output and checkpoint I/O: VTK XML (.vtr/.vtm/.pvd)
// and a dependency-free NumPy .npy/.npz reader/writer for checkpoints.
#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <map>

#include "fv/Core.hpp"

namespace fv {

class CellMesh;

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
    // One snapshot of the given leaves of an adaptive mesh (a VTK_QUAD each: the fields of export_vtr and AMR_Level).
    // One rank: one .vtu. On `size` > 1 ranks every rank calls this with the leaves it owns: each writes a .vtu piece in a
    // subfolder, `sync` is a barrier, and rank 0 writes the .pvtu that lists the pieces (and is the only one to record the
    // snapshot in `collection`).
    void export_cells(const std::string& prefix, const EOS& eos, const CellMesh& mesh, const std::vector<int>& leaves,
                      double time, int step, int rank = 0, int size = 1, const std::function<void()>& sync = {});
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

// Renames `from` over `to` (atomic replace). On Windows a reader that has `to` open (GUI monitor, ParaView, antivirus)
// makes the rename fail for a moment: it is retried a few times, then the error is thrown.
void rename_replace(const std::string& from, const std::string& to);

// Uncompressed (ZIP_STORED) archive, readable by numpy.load.
void write_npz(const std::string& path, const NpzArchive& arrays);
// An .npz archive (uncompressed, as write_npz makes it) written entry by entry. An entry of float64 can be reserved with its shape
// and filled later in pieces, in any order, so that an array that no single process holds goes to the file as its parts arrive.
// The bytes are those write_npz gives for the same entries in the same order. Entries or archives of 4 GB and more are written in
// the zip64 form.
class NpzWriter {
public:
    // `zip64_from`: sizes and offsets from this value on take the zip64 form (the limit of the format; lower only in tests).
    explicit NpzWriter(const std::string& path, uint64_t zip64_from = 0xFFFFFFFFull);
    // A whole entry.
    void add(const std::string& key, const NpyArray& a);
    // A float64 entry of the given shape whose values come later through put(); returns its handle.
    int reserve(const std::string& key, const std::vector<int64_t>& shape);
    // The `n` values of the reserved entry from element `at` on (C order). The pieces of an entry must cover it exactly once.
    void put(int entry, uint64_t at, const double* v, size_t n);
    // Checks that every reserved entry is complete, then writes the checksums and the directory. Must be called.
    void close();

private:
    struct Piece {
        uint64_t at = 0, n = 0;   // first element and number of elements
        uint32_t crc = 0;
    };
    struct Entry {
        std::string name;
        uint64_t offset = 0, size = 0;     // position of the local header; length of the .npy bytes
        uint64_t data = 0, elements = 0;   // reserved entries: position of the first value, number of values
        uint32_t crc = 0;
        bool reserved = false;
        std::vector<Piece> pieces;
    };
    void write_at(uint64_t at, const char* bytes, size_t n);
    std::string path_;
    std::ofstream out_;
    uint64_t limit_ = 0, written_ = 0;
    std::vector<Entry> entries_;
    bool closed_ = false;
};
// Reads archives written by numpy.savez (stored entries, zip64 aware).
NpzArchive read_npz(const std::string& path);

// An .npz file opened for reading parts of it: only the central directory is read; an entry, or chosen rows of a 2-D float64
// entry, are read when asked (a restart of a distributed run reads the rows of its own leaves, not the whole state).
class NpzFile {
public:
    explicit NpzFile(const std::string& path);
    bool has(const std::string& key) const { return entries_.count(key) > 0; }
    // The whole entry.
    NpyArray read(const std::string& key) const;
    // Length of the first axis of an entry (reads its header only).
    int64_t length(const std::string& key) const;
    // The rows `rows` (indices into the first axis, any order) of a 2-D float64 entry: ncol doubles per row, row after row, in the
    // order of `rows`; ncol returns the number of columns.
    std::vector<double> read_rows(const std::string& key, const std::vector<int64_t>& rows, int& ncol) const;

private:
    struct Entry {
        uint64_t data = 0, size = 0;   // position and length of the .npy bytes in the file
    };
    const Entry& entry(const std::string& key) const;
    std::string path_;
    std::map<std::string, Entry> entries_;
};

// State field as an (n, ncomp) float64 array: ncomp = 4 ([rho, rho*u, rho*v, rho*E]) for
// 2-D fields, kNVar (+ rho*w) for 3-D ones.
NpyArray field_to_npy(const Field& U, int ncomp = 4);
// (n, 4) or (n, 5) array back to a state field (missing rho*w = 0); throws on other shapes.
Field npy_to_field(const NpyArray& a);

}   // namespace fv
