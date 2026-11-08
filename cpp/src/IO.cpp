#include "fv/IO.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "fv/CellMesh.hpp"
#include "fv/SolutionMapping.hpp"

namespace fs = std::filesystem;

namespace fv {

void rename_replace(const std::string& from, const std::string& to) {
    for (int attempt = 0;; ++attempt) {
        std::error_code ec;
        fs::rename(from, to, ec);
        if (!ec) return;
        if (attempt >= 9) throw fs::filesystem_error("Cannot rename " + from + " to " + to, from, to, ec);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

namespace {

void write_file_atomic(const fs::path& path, const std::string& content) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write " + tmp.string());
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    rename_replace(tmp.string(), path.string());   // atomic replace: readers never see partial files
}

// Replace NaN/Inf/subnormal values so ParaView's float32 parser accepts them.
inline double sanitize(double x) {
    if (std::isnan(x)) return 0.0;
    if (std::isinf(x)) return x > 0 ? 1e30 : -1e30;
    if (std::abs(x) < 1e-30) return 0.0;
    return x;
}

// v in scientific notation with `prec` digits, the same text as printf("%.<prec>e")
// but several times faster (VTK output time was dominated by snprintf).
inline void append_sci(std::string& s, double v, int prec) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific, prec);
    s.append(buf, r.ptr);
}

void append_base64(std::string& s, const uint8_t* d, size_t n) {
    static const char* abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const size_t at = s.size();
    s.resize(at + (n + 2) / 3 * 4);   // written through a pointer: one append per character was the cost
    char* o = &s[at];
    size_t i = 0;
    for (; i + 2 < n; i += 3, o += 4) {
        const uint32_t v = (uint32_t(d[i]) << 16) | (uint32_t(d[i + 1]) << 8) | d[i + 2];
        o[0] = abc[v >> 18];
        o[1] = abc[(v >> 12) & 63];
        o[2] = abc[(v >> 6) & 63];
        o[3] = abc[v & 63];
    }
    if (i < n) {
        const uint32_t v = (uint32_t(d[i]) << 16) | (i + 1 < n ? uint32_t(d[i + 1]) << 8 : 0u);
        o[0] = abc[v >> 18];
        o[1] = abc[(v >> 12) & 63];
        o[2] = i + 1 < n ? abc[(v >> 6) & 63] : '=';
        o[3] = '=';
    }
}

// One Float32 <DataArray> of tuples of `ncomp` values.  ASCII: `prec` digits, one tuple
// per line.  Binary: VTK inline base64 of a UInt32 byte count followed by the
// little-endian floats, encoded as a single stream.
void data_array(std::string& s, const char* attrs, const std::vector<double>& d, int ncomp, int prec, bool binary) {
    s += "        <DataArray type=\"Float32\" ";
    s += attrs;
    s += binary ? " format=\"binary\">\n" : " format=\"ascii\">\n";
    if (binary) {
        std::vector<uint8_t> raw(4 + d.size() * 4);
        const uint32_t bytes = static_cast<uint32_t>(d.size() * 4);
        std::memcpy(raw.data(), &bytes, 4);
        for (size_t k = 0; k < d.size(); ++k) {
            const float f = static_cast<float>(d[k]);
            std::memcpy(raw.data() + 4 + 4 * k, &f, 4);
        }
        s += "          ";
        append_base64(s, raw.data(), raw.size());
        s += '\n';
    } else {
        for (size_t k = 0; k < d.size(); ++k) {
            append_sci(s, d[k], prec);
            s += (k + 1) % ncomp == 0 ? '\n' : ' ';
        }
    }
    s += "        </DataArray>\n";
}

// One RectilinearGrid piece; `level` < 0 means no AMR_Level field.
std::string build_vtr(const EOS& eos, const Field& U, int nx, int ny, double x0, double x1, double y0, double y1,
                      int level, bool binary, int nz = 1, double z0 = 0.0, double z1 = 0.0) {
    const int n = nx * ny * nz;
    const bool three_d = nz > 1;
    std::vector<double> rho(n), u(n), v(n), w(n), p(n), T(n), mach(n);
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) {
        const State W = eos.cons_to_prim(U[c]);
        rho[c] = sanitize(W[0]);
        u[c] = sanitize(W[1]);
        v[c] = sanitize(W[2]);
        p[c] = sanitize(W[3]);
        w[c] = sanitize(W[4]);
        T[c] = sanitize(p[c] / (rho[c] + 1e-12));
        const double cs = eos.sound_speed(p[c], rho[c]);
        mach[c] = sanitize(std::sqrt(u[c] * u[c] + v[c] * v[c] + (three_d ? w[c] * w[c] : 0.0)) / (cs + 1e-12));
    }

    std::string s;
    s.reserve(static_cast<size_t>(n) * 110 + 4096);
    char head[256];
    s += "<?xml version=\"1.0\"?>\n";
    s += "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    std::snprintf(head, sizeof(head), "  <RectilinearGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", nx, ny, three_d ? nz : 0);
    s += head;
    std::snprintf(head, sizeof(head), "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", nx, ny, three_d ? nz : 0);
    s += head;

    s += "      <Coordinates>\n";
    const std::vector<double> xs = linspace(x0, x1, nx + 1), ys = linspace(y0, y1, ny + 1);
    const std::vector<double> zs = three_d ? linspace(z0, z1, nz + 1) : std::vector<double>{0.0};
    const std::pair<const char*, const std::vector<double>*> coords[] = {{"x", &xs}, {"y", &ys}, {"z", &zs}};
    for (const auto& c : coords) {
        std::snprintf(head, sizeof(head), "Name=\"%s\"", c.first);
        data_array(s, head, *c.second, 1, 6, binary);
    }
    s += "      </Coordinates>\n";

    s += "      <CellData Scalars=\"Density\" Vectors=\"Velocity\">\n";
    auto scalar = [&](const char* name, const std::vector<double>& d) {
        std::snprintf(head, sizeof(head), "Name=\"%s\" NumberOfComponents=\"1\"", name);
        data_array(s, head, d, 1, 5, binary);
    };
    scalar("Density", rho);
    scalar("Pressure", p);
    scalar("Temperature_nondim", T);
    scalar("Mach", mach);
    std::vector<double> vel(static_cast<size_t>(n) * 3);
    for (int c = 0; c < n; ++c) {
        vel[3 * c] = u[c];
        vel[3 * c + 1] = v[c];
        vel[3 * c + 2] = three_d ? w[c] : 0.0;
    }
    data_array(s, "Name=\"Velocity\" NumberOfComponents=\"3\"", vel, 3, 5, binary);
    if (level >= 0) scalar("AMR_Level", std::vector<double>(n, static_cast<double>(level)));
    s += "      </CellData>\n";
    s += "    </Piece>\n";
    s += "  </RectilinearGrid>\n";
    s += "</VTKFile>\n";
    return s;
}

// An Int32 / UInt8 <DataArray> written as ASCII text (connectivity, offsets, cell types).
void int_array(std::string& s, const char* type, const char* name, const std::vector<int>& d) {
    s += "        <DataArray type=\"";
    s += type;
    s += "\" Name=\"";
    s += name;
    s += "\" format=\"ascii\">\n          ";
    char buf[16];
    for (size_t k = 0; k < d.size(); ++k) {
        const auto r = std::to_chars(buf, buf + sizeof(buf), d[k]);
        s.append(buf, r.ptr);
        s += (k + 1) % 16 == 0 ? "\n          " : " ";
    }
    s += "\n        </DataArray>\n";
}

// One UnstructuredGrid piece of the given leaves: a VTK_QUAD per leaf (a VTK_HEXAHEDRON in 3-D: the four corners of the bottom
// face counter-clockwise, then the same four on top), the cell fields of build_vtr and the level.
std::string build_vtu(const EOS& eos, const CellMesh& m, const std::vector<int>& leaves, bool binary) {
    const int n = static_cast<int>(leaves.size());
    const bool three_d = m.dim == 3;
    const int corners = three_d ? 8 : 4;
    std::vector<double> rho(n), u(n), v(n), w(n, 0.0), p(n), T(n), mach(n), lev(n), pts(static_cast<size_t>(n) * 3 * corners);
    for (int k = 0; k < n; ++k) {
        const size_t c = static_cast<size_t>(leaves[static_cast<size_t>(k)]);
        const State W = eos.cons_to_prim(m.U[c]);
        rho[k] = sanitize(W[0]);
        u[k] = sanitize(W[1]);
        v[k] = sanitize(W[2]);
        if (three_d) w[k] = sanitize(W[4]);
        p[k] = sanitize(W[3]);
        T[k] = sanitize(p[k] / (rho[k] + 1e-12));
        const double cs = eos.sound_speed(p[k], rho[k]);
        mach[k] = sanitize(std::sqrt(three_d ? u[k] * u[k] + v[k] * v[k] + w[k] * w[k] : u[k] * u[k] + v[k] * v[k]) / (cs + 1e-12));
        const int l = m.level[c];
        lev[k] = l;
        const double x0 = m.x0 + m.ci[c] * m.dx(l), x1 = m.x0 + (m.ci[c] + 1) * m.dx(l);
        const double y0 = m.y0 + m.cj[c] * m.dy(l), y1 = m.y0 + (m.cj[c] + 1) * m.dy(l);
        if (three_d) {
            const double z0 = m.z0 + m.ck[c] * m.dz(l), z1 = m.z0 + (m.ck[c] + 1) * m.dz(l);
            const double q[24] = {x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1};
            std::copy(q, q + 24, pts.begin() + 24 * static_cast<std::ptrdiff_t>(k));
        } else {
            const double q[12] = {x0, y0, 0.0, x1, y0, 0.0, x1, y1, 0.0, x0, y1, 0.0};
            std::copy(q, q + 12, pts.begin() + 12 * static_cast<std::ptrdiff_t>(k));
        }
    }
    std::string s;
    s.reserve(static_cast<size_t>(n) * (three_d ? 700 : 400) + 4096);
    char head[256];
    s += "<?xml version=\"1.0\"?>\n";
    s += "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    s += "  <UnstructuredGrid>\n";
    std::snprintf(head, sizeof(head), "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", corners * n, n);
    s += head;
    s += "      <Points>\n";
    data_array(s, "Name=\"Points\" NumberOfComponents=\"3\"", pts, 3, 7, binary);
    s += "      </Points>\n";
    s += "      <Cells>\n";
    std::vector<int> conn(static_cast<size_t>(n) * corners), off(static_cast<size_t>(n)),
        type(static_cast<size_t>(n), three_d ? 12 : 9);
    for (int k = 0; k < corners * n; ++k) conn[static_cast<size_t>(k)] = k;
    for (int k = 0; k < n; ++k) off[static_cast<size_t>(k)] = corners * (k + 1);
    int_array(s, "Int32", "connectivity", conn);
    int_array(s, "Int32", "offsets", off);
    int_array(s, "UInt8", "types", type);
    s += "      </Cells>\n";
    s += "      <CellData Scalars=\"Density\" Vectors=\"Velocity\">\n";
    auto scalar = [&](const char* name, const std::vector<double>& d) {
        std::snprintf(head, sizeof(head), "Name=\"%s\" NumberOfComponents=\"1\"", name);
        data_array(s, head, d, 1, 5, binary);
    };
    scalar("Density", rho);
    scalar("Pressure", p);
    scalar("Temperature_nondim", T);
    scalar("Mach", mach);
    std::vector<double> vel(static_cast<size_t>(n) * 3, 0.0);
    for (int k = 0; k < n; ++k) {
        vel[3 * static_cast<size_t>(k)] = u[k];
        vel[3 * static_cast<size_t>(k) + 1] = v[k];
        vel[3 * static_cast<size_t>(k) + 2] = w[k];
    }
    data_array(s, "Name=\"Velocity\" NumberOfComponents=\"3\"", vel, 3, 5, binary);
    scalar("AMR_Level", lev);
    s += "      </CellData>\n";
    s += "    </Piece>\n";
    s += "  </UnstructuredGrid>\n";
    s += "</VTKFile>\n";
    return s;
}

std::string step_name(const std::string& prefix, int step, const char* ext) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%05d", step);
    return prefix + buf + ext;
}

}   // namespace

// ---------------------------------------------------------------------------
// VTKExporter
// ---------------------------------------------------------------------------

VTKExporter::VTKExporter(std::string output_dir, bool verbose, bool binary)
    : dir_(std::move(output_dir)), binary_(binary) {
    fs::create_directories(dir_);
    if (verbose) std::printf("[EXPORTER] Output directory: '%s' (%s VTK)\n", dir_.c_str(), binary ? "binary" : "ASCII");
}

void VTKExporter::export_vtr(const std::string& prefix, const EOS& eos, const Field& U, const Mesh& mesh, double time,
                             int step) {
    const std::string name = step_name(prefix, step, ".vtr");
    write_file_atomic(fs::path(dir_) / name, build_vtr(eos, U, mesh.nx, mesh.ny, 0.0, mesh.Lx, 0.0, mesh.Ly, -1,
                                                       binary_, mesh.nz, 0.0, mesh.Lz));
    collection.push_back({time, name});
}

void VTKExporter::export_cells(const std::string& prefix, const EOS& eos, const CellMesh& mesh, const std::vector<int>& leaves,
                               double time, int step, int rank, int size, const std::function<void()>& sync) {
    if (size <= 1) {
        const std::string name = step_name(prefix, step, ".vtu");
        write_file_atomic(fs::path(dir_) / name, build_vtu(eos, mesh, leaves, binary_));
        collection.push_back({time, name});
        return;
    }
    // several ranks: each writes a piece of the leaves it owns, rank 0 the .pvtu that lists them
    const std::string piece_dir = step_name(prefix, step, "");
    fs::create_directories(fs::path(dir_) / piece_dir);
    char piece[32];
    std::snprintf(piece, sizeof(piece), "r%04d.vtu", rank);
    write_file_atomic(fs::path(dir_) / piece_dir / piece, build_vtu(eos, mesh, leaves, binary_));
    if (sync) sync();   // all pieces are on disk before the index points at them
    if (rank != 0) return;
    const std::string name = step_name(prefix, step, ".pvtu");
    std::string s = "<?xml version=\"1.0\"?>\n"
                    "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                    "  <PUnstructuredGrid GhostLevel=\"0\">\n"
                    "    <PPoints><PDataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\"/></PPoints>\n"
                    "    <PCellData Scalars=\"Density\" Vectors=\"Velocity\">\n"
                    "      <PDataArray type=\"Float32\" Name=\"Density\" NumberOfComponents=\"1\"/>\n"
                    "      <PDataArray type=\"Float32\" Name=\"Pressure\" NumberOfComponents=\"1\"/>\n"
                    "      <PDataArray type=\"Float32\" Name=\"Temperature_nondim\" NumberOfComponents=\"1\"/>\n"
                    "      <PDataArray type=\"Float32\" Name=\"Mach\" NumberOfComponents=\"1\"/>\n"
                    "      <PDataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\"/>\n"
                    "      <PDataArray type=\"Float32\" Name=\"AMR_Level\" NumberOfComponents=\"1\"/>\n"
                    "    </PCellData>\n";
    char line[256];
    for (int r = 0; r < size; ++r) {
        std::snprintf(line, sizeof(line), "    <Piece Source=\"%s/r%04d.vtu\"/>\n", piece_dir.c_str(), r);
        s += line;
    }
    s += "  </PUnstructuredGrid>\n</VTKFile>\n";
    write_file_atomic(fs::path(dir_) / name, s);
    collection.push_back({time, name});
}

void VTKExporter::write_pvd(const std::string& prefix) const {
    std::string s = "<?xml version=\"1.0\"?>\n"
                    "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                    "  <Collection>\n";
    char line[512];
    for (const auto& e : collection) {
        std::snprintf(line, sizeof(line), "    <DataSet timestep=\"%.6e\" group=\"\" part=\"0\" file=\"%s\"/>\n",
                      e.time, e.filename.c_str());
        s += line;
    }
    s += "  </Collection>\n</VTKFile>\n";
    write_file_atomic(fs::path(dir_) / (prefix + ".pvd"), s);
    std::printf("[EXPORTER] Time-series index written: '%s.pvd'\n", prefix.c_str());
}

// ---------------------------------------------------------------------------
// .npy
// ---------------------------------------------------------------------------

int64_t NpyArray::size() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

std::vector<double> NpyArray::as_doubles() const {
    const int64_t n = size();
    std::vector<double> out(static_cast<size_t>(n));
    const std::string t = descr.size() >= 2 ? descr.substr(1) : descr;
    if (!descr.empty() && descr[0] == '>') throw std::runtime_error("Big-endian .npy arrays are not supported");
    auto get = [&](auto tag) {
        using T = decltype(tag);
        if (data.size() < static_cast<size_t>(n) * sizeof(T)) throw std::runtime_error("Truncated .npy data");
        for (int64_t k = 0; k < n; ++k) {
            T v;
            std::memcpy(&v, data.data() + k * sizeof(T), sizeof(T));
            out[static_cast<size_t>(k)] = static_cast<double>(v);
        }
    };
    if (t == "f8")
        get(double{});
    else if (t == "f4")
        get(float{});
    else if (t == "i8")
        get(int64_t{});
    else if (t == "i4")
        get(int32_t{});
    else if (t == "u8")
        get(uint64_t{});
    else if (t == "u4")
        get(uint32_t{});
    else if (t == "b1")
        get(uint8_t{});
    else
        throw std::runtime_error("Unsupported .npy dtype '" + descr + "'");
    return out;
}

std::vector<std::string> NpyArray::as_strings() const {
    if (descr.size() < 3 || descr[1] != 'U') throw std::runtime_error("Not a unicode .npy array: " + descr);
    const int64_t width = std::stoll(descr.substr(2));
    if (data.size() < static_cast<size_t>(size() * width * 4)) throw std::runtime_error("Truncated .npy data");
    std::vector<std::string> out;
    for (int64_t k = 0; k < size(); ++k) {
        std::string s;
        for (int64_t c = 0; c < width; ++c) {
            uint32_t cp;
            std::memcpy(&cp, data.data() + (k * width + c) * 4, 4);
            if (cp == 0) break;
            if (cp < 0x80)
                s += static_cast<char>(cp);
            else if (cp < 0x800) {
                s += static_cast<char>(0xC0 | (cp >> 6));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                s += static_cast<char>(0xE0 | (cp >> 12));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                s += static_cast<char>(0xF0 | (cp >> 18));
                s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                s += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
        out.push_back(s);
    }
    return out;
}

NpyArray NpyArray::from_doubles(const std::vector<double>& v, std::vector<int64_t> shape) {
    NpyArray a;
    a.descr = "<f8";
    a.shape = std::move(shape);
    a.data.resize(v.size() * sizeof(double));
    if (!v.empty()) std::memcpy(a.data.data(), v.data(), a.data.size());
    return a;
}

NpyArray NpyArray::from_double(double v) { return from_doubles({v}, {}); }

NpyArray NpyArray::from_int(int64_t v) {
    NpyArray a;
    a.descr = "<i8";
    a.data.resize(8);
    std::memcpy(a.data.data(), &v, 8);
    return a;
}

NpyArray NpyArray::from_strings(const std::vector<std::string>& v) {
    size_t width = 1;
    for (const auto& s : v) width = std::max(width, s.size());
    NpyArray a;
    a.descr = "<U" + std::to_string(width);
    a.shape = {static_cast<int64_t>(v.size())};
    a.data.assign(v.size() * width * 4, 0);
    for (size_t k = 0; k < v.size(); ++k)
        for (size_t c = 0; c < v[k].size(); ++c) {
            const uint32_t cp = static_cast<unsigned char>(v[k][c]);   // ASCII file names
            std::memcpy(a.data.data() + (k * width + c) * 4, &cp, 4);
        }
    return a;
}

NpyArray field_to_npy(const Field& U, int ncomp) {
    NpyArray a;
    a.descr = "<f8";
    a.shape = {static_cast<int64_t>(U.size()), ncomp};
    a.data.resize(U.size() * ncomp * sizeof(double));
    for (size_t c = 0; c < U.size(); ++c)
        std::memcpy(a.data.data() + c * ncomp * sizeof(double), U[c].data(), ncomp * sizeof(double));
    return a;
}

Field npy_to_field(const NpyArray& a) {
    if (a.shape.size() != 2 || (a.shape[1] != 4 && a.shape[1] != kNVar))
        throw std::runtime_error("Expected an (n, 4) or (n, 5) state array");
    const auto d = a.as_doubles();
    const int ncomp = static_cast<int>(a.shape[1]);
    Field U(static_cast<size_t>(a.shape[0]), State{});
    for (size_t c = 0; c < U.size(); ++c)
        for (int q = 0; q < ncomp; ++q) U[c][q] = d[c * ncomp + q];
    return U;
}

namespace {

std::string npy_bytes(const NpyArray& a) {
    std::string shape = "(";
    for (size_t k = 0; k < a.shape.size(); ++k) {
        shape += std::to_string(a.shape[k]);
        if (a.shape.size() == 1 || k + 1 < a.shape.size()) shape += ",";
        if (k + 1 < a.shape.size()) shape += " ";
    }
    shape += ")";
    std::string header = "{'descr': '" + a.descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    const size_t base = 10;   // magic(6) + version(2) + header_len(2)
    size_t total = base + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';

    std::string out = "\x93NUMPY";
    out += '\x01';
    out += '\x00';
    const uint16_t hl = static_cast<uint16_t>(header.size());
    out += static_cast<char>(hl & 0xFF);
    out += static_cast<char>(hl >> 8);
    out += header;
    out.append(reinterpret_cast<const char*>(a.data.data()), a.data.size());
    return out;
}

NpyArray parse_npy(const uint8_t* p, size_t n) {
    if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) throw std::runtime_error("Not a .npy entry");
    const int major = p[6];
    size_t hlen, off;
    if (major == 1) {
        hlen = p[8] | (p[9] << 8);
        off = 10;
    } else {
        hlen = p[8] | (p[9] << 8) | (static_cast<size_t>(p[10]) << 16) | (static_cast<size_t>(p[11]) << 24);
        off = 12;
    }
    const std::string h(reinterpret_cast<const char*>(p + off), hlen);

    NpyArray a;
    auto value_after = [&](const std::string& key) {
        const size_t k = h.find(key);
        if (k == std::string::npos) throw std::runtime_error("Malformed .npy header: " + h);
        return k + key.size();
    };
    size_t q = h.find('\'', value_after("'descr':"));
    a.descr = h.substr(q + 1, h.find('\'', q + 1) - q - 1);
    if (h.find("True", value_after("'fortran_order':")) == value_after("'fortran_order':") + 1)
        throw std::runtime_error("Fortran-ordered .npy arrays are not supported");
    const size_t s0 = h.find('(', value_after("'shape':")), s1 = h.find(')', s0);
    std::stringstream ss(h.substr(s0 + 1, s1 - s0 - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_not_of(" ") == std::string::npos) continue;
        a.shape.push_back(std::stoll(tok));
    }
    a.data.assign(p + off + hlen, p + n);
    return a;
}

// ---------------------------------------------------------------------------
// ZIP (stored entries only)
// ---------------------------------------------------------------------------

// CRC-32 (zip), eight bytes per step (slicing-by-8; the value is the same as the byte-wise table).
uint32_t crc32(const uint8_t* d, size_t n) {
    static const std::array<std::array<uint32_t, 256>, 8> table = [] {   // thread-safe one-time init
        std::array<std::array<uint32_t, 256>, 8> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; ++i)
            for (int s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFF];
        return t;
    }();
    uint32_t c = 0xFFFFFFFFu;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {   // little-endian assembly of the two words, independent of the host byte order
        const uint32_t lo = (c ^ (static_cast<uint32_t>(d[i]) | static_cast<uint32_t>(d[i + 1]) << 8 |
                                  static_cast<uint32_t>(d[i + 2]) << 16 | static_cast<uint32_t>(d[i + 3]) << 24));
        const uint32_t hi = static_cast<uint32_t>(d[i + 4]) | static_cast<uint32_t>(d[i + 5]) << 8 |
                            static_cast<uint32_t>(d[i + 6]) << 16 | static_cast<uint32_t>(d[i + 7]) << 24;
        c = table[7][lo & 0xFF] ^ table[6][(lo >> 8) & 0xFF] ^ table[5][(lo >> 16) & 0xFF] ^ table[4][lo >> 24] ^
            table[3][hi & 0xFF] ^ table[2][(hi >> 8) & 0xFF] ^ table[1][(hi >> 16) & 0xFF] ^ table[0][hi >> 24];
    }
    for (; i < n; ++i) c = table[0][(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void put16(std::string& s, uint32_t v) {
    s += static_cast<char>(v & 0xFF);
    s += static_cast<char>((v >> 8) & 0xFF);
}
void put32(std::string& s, uint32_t v) {
    put16(s, v & 0xFFFF);
    put16(s, v >> 16);
}

uint64_t get(const std::vector<uint8_t>& b, size_t at, int bytes) {
    if (at + bytes > b.size()) throw std::runtime_error("Truncated zip archive");
    uint64_t v = 0;
    for (int k = bytes - 1; k >= 0; --k) v = (v << 8) | b[at + k];
    return v;
}

}   // namespace

void write_npz(const std::string& path, const NpzArchive& arrays) {
    // Entry after entry goes straight to the file (the archive is never held whole in memory): only the central
    // directory is collected.
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write " + path);
    std::string central;
    uint64_t written = 0;
    uint32_t count = 0;
    for (const auto& kv : arrays) {
        const std::string name = kv.first + ".npy";
        const std::string body = npy_bytes(kv.second);
        if (body.size() >= 0xFFFFFFFFull || written >= 0xFFFFFFFFull)
            throw std::runtime_error("Checkpoint exceeds 4 GB (zip64 writing not supported)");
        const uint32_t crc = crc32(reinterpret_cast<const uint8_t*>(body.data()), body.size());
        const uint32_t offset = static_cast<uint32_t>(written);

        std::string zip;
        zip += "PK\x03\x04";
        put16(zip, 20);
        put16(zip, 0);
        put16(zip, 0);   // version, flags, stored
        put16(zip, 0);
        put16(zip, 0x21);   // time, date (1980-01-01)
        put32(zip, crc);
        put32(zip, static_cast<uint32_t>(body.size()));
        put32(zip, static_cast<uint32_t>(body.size()));
        put16(zip, static_cast<uint32_t>(name.size()));
        put16(zip, 0);
        zip += name;
        out.write(zip.data(), static_cast<std::streamsize>(zip.size()));
        out.write(body.data(), static_cast<std::streamsize>(body.size()));
        written += zip.size() + body.size();

        central += "PK\x01\x02";
        put16(central, 20);
        put16(central, 20);
        put16(central, 0);
        put16(central, 0);
        put16(central, 0);
        put16(central, 0x21);
        put32(central, crc);
        put32(central, static_cast<uint32_t>(body.size()));
        put32(central, static_cast<uint32_t>(body.size()));
        put16(central, static_cast<uint32_t>(name.size()));
        put16(central, 0);
        put16(central, 0);
        put16(central, 0);
        put16(central, 0);
        put32(central, 0);
        put32(central, offset);
        central += name;
        ++count;
    }
    const uint32_t cd_offset = static_cast<uint32_t>(written);
    std::string zip = central;
    zip += "PK\x05\x06";
    put16(zip, 0);
    put16(zip, 0);
    put16(zip, count);
    put16(zip, count);
    put32(zip, static_cast<uint32_t>(central.size()));
    put32(zip, cd_offset);
    put16(zip, 0);
    out.write(zip.data(), static_cast<std::streamsize>(zip.size()));
}

NpzArchive read_npz(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> b(file_size > 0 ? static_cast<size_t>(file_size) : 0);
    if (!b.empty()) in.read(reinterpret_cast<char*>(b.data()), file_size);
    if (b.size() < 22) throw std::runtime_error("Not a zip archive: " + path);

    // End of central directory (scan backwards over a possible comment)
    size_t eocd = std::string::npos;
    const size_t lo = b.size() > 65557 ? b.size() - 65557 : 0;
    for (size_t k = b.size() - 22 + 1; k-- > lo;)
        if (get(b, k, 4) == 0x06054b50) {
            eocd = k;
            break;
        }
    if (eocd == std::string::npos) throw std::runtime_error("No end-of-central-directory in " + path);
    uint64_t entries = get(b, eocd + 10, 2);
    uint64_t cd_off = get(b, eocd + 16, 4);
    if ((entries == 0xFFFF || cd_off == 0xFFFFFFFF) && eocd >= 20 && get(b, eocd - 20, 4) == 0x07064b50) {
        const size_t z = static_cast<size_t>(get(b, eocd - 20 + 8, 8));
        entries = get(b, z + 32, 8);
        cd_off = get(b, z + 48, 8);
    }

    NpzArchive out;
    size_t p = static_cast<size_t>(cd_off);
    for (uint64_t e = 0; e < entries; ++e) {
        if (get(b, p, 4) != 0x02014b50) throw std::runtime_error("Corrupt central directory in " + path);
        const uint64_t method = get(b, p + 10, 2);
        uint64_t csize = get(b, p + 20, 4), usize = get(b, p + 24, 4), loff = get(b, p + 42, 4);
        const size_t nlen = static_cast<size_t>(get(b, p + 28, 2));
        const size_t xlen = static_cast<size_t>(get(b, p + 30, 2));
        const size_t clen = static_cast<size_t>(get(b, p + 32, 2));
        const std::string name(reinterpret_cast<const char*>(&b[p + 46]), nlen);
        // zip64 extra field
        size_t x = p + 46 + nlen;
        const size_t xend = x + xlen;
        while (x + 4 <= xend) {
            const uint64_t id = get(b, x, 2), sz = get(b, x + 2, 2);
            if (id == 0x0001) {
                size_t y = x + 4;
                if (usize == 0xFFFFFFFF) {
                    usize = get(b, y, 8);
                    y += 8;
                }
                if (csize == 0xFFFFFFFF) {
                    csize = get(b, y, 8);
                    y += 8;
                }
                if (loff == 0xFFFFFFFF) {
                    loff = get(b, y, 8);
                    y += 8;
                }
            }
            x += 4 + static_cast<size_t>(sz);
        }
        if (method != 0)
            throw std::runtime_error("Compressed npz entries (np.savez_compressed) are not supported: " + name);
        const size_t lh = static_cast<size_t>(loff);
        const size_t data = lh + 30 + static_cast<size_t>(get(b, lh + 26, 2)) + static_cast<size_t>(get(b, lh + 28, 2));
        if (data + csize > b.size()) throw std::runtime_error("Truncated npz entry " + name);
        std::string key = name;
        if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0) key.resize(key.size() - 4);
        out[key] = parse_npy(&b[data], static_cast<size_t>(csize));
        p += 46 + nlen + xlen + clen;
    }
    return out;
}

// ---------------------------------------------------------------------------
// NpzFile: partial reads of an uncompressed archive
// ---------------------------------------------------------------------------

namespace {
std::vector<uint8_t> read_at(std::ifstream& in, uint64_t off, uint64_t n, const std::string& path) {
    std::vector<uint8_t> b(static_cast<size_t>(n));
    in.clear();
    in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    if (n > 0) in.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("Truncated npz file " + path);
    return b;
}
}   // namespace

NpzFile::NpzFile(const std::string& path) : path_(path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);
    in.seekg(0, std::ios::end);
    const uint64_t fsize = static_cast<uint64_t>(in.tellg());
    if (fsize < 22) throw std::runtime_error("Not a zip archive: " + path);
    const uint64_t tail = std::min<uint64_t>(fsize, 65557 + 22);
    const std::vector<uint8_t> t = read_at(in, fsize - tail, tail, path);
    size_t eocd = std::string::npos;
    for (size_t k = t.size() - 22 + 1; k-- > 0;)
        if (get(t, k, 4) == 0x06054b50) {
            eocd = k;
            break;
        }
    if (eocd == std::string::npos) throw std::runtime_error("No end-of-central-directory in " + path);
    uint64_t entries = get(t, eocd + 10, 2), cd_size = get(t, eocd + 12, 4), cd_off = get(t, eocd + 16, 4);
    if ((entries == 0xFFFF || cd_off == 0xFFFFFFFF) && eocd >= 20 && get(t, eocd - 20, 4) == 0x07064b50) {
        const uint64_t z = get(t, eocd - 20 + 8, 8);
        const std::vector<uint8_t> r = read_at(in, z, 56, path);
        entries = get(r, 32, 8);
        cd_size = get(r, 40, 8);
        cd_off = get(r, 48, 8);
    }
    const std::vector<uint8_t> b = read_at(in, cd_off, cd_size, path);
    size_t p = 0;
    for (uint64_t e = 0; e < entries; ++e) {
        if (get(b, p, 4) != 0x02014b50) throw std::runtime_error("Corrupt central directory in " + path);
        const uint64_t method = get(b, p + 10, 2);
        uint64_t csize = get(b, p + 20, 4), usize = get(b, p + 24, 4), loff = get(b, p + 42, 4);
        const size_t nlen = static_cast<size_t>(get(b, p + 28, 2));
        const size_t xlen = static_cast<size_t>(get(b, p + 30, 2));
        const size_t clen = static_cast<size_t>(get(b, p + 32, 2));
        const std::string name(reinterpret_cast<const char*>(&b[p + 46]), nlen);
        size_t x = p + 46 + nlen;
        const size_t xend = x + xlen;
        while (x + 4 <= xend) {
            const uint64_t id = get(b, x, 2), sz = get(b, x + 2, 2);
            if (id == 0x0001) {
                size_t y = x + 4;
                if (usize == 0xFFFFFFFF) {
                    usize = get(b, y, 8);
                    y += 8;
                }
                if (csize == 0xFFFFFFFF) {
                    csize = get(b, y, 8);
                    y += 8;
                }
                if (loff == 0xFFFFFFFF) {
                    loff = get(b, y, 8);
                    y += 8;
                }
            }
            x += 4 + static_cast<size_t>(sz);
        }
        if (method != 0) throw std::runtime_error("Compressed npz entries (np.savez_compressed) are not supported: " + name);
        const std::vector<uint8_t> lh = read_at(in, loff, 30, path);
        Entry en;
        en.data = loff + 30 + get(lh, 26, 2) + get(lh, 28, 2);
        en.size = csize;
        if (en.data + en.size > fsize) throw std::runtime_error("Truncated npz entry " + name);
        std::string key = name;
        if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0) key.resize(key.size() - 4);
        entries_[key] = en;
        p += 46 + nlen + xlen + clen;
    }
}

const NpzFile::Entry& NpzFile::entry(const std::string& key) const {
    const auto it = entries_.find(key);
    if (it == entries_.end()) throw std::out_of_range("No entry '" + key + "' in " + path_);
    return it->second;
}

NpyArray NpzFile::read(const std::string& key) const {
    const Entry& e = entry(key);
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path_);
    const std::vector<uint8_t> b = read_at(in, e.data, e.size, path_);
    return parse_npy(b.data(), b.size());
}

namespace {
// The .npy header of an entry: the array without data (descr, shape) and the offset of the data inside the entry.
NpyArray npy_header(std::ifstream& in, uint64_t at, const std::string& path, uint64_t& data_off) {
    const std::vector<uint8_t> h = read_at(in, at, 12, path);
    if (std::memcmp(h.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error("Not a .npy entry in " + path);
    const uint64_t hlen = h[6] == 1 ? (h[8] | (h[9] << 8)) : (h[8] | (h[9] << 8) | (static_cast<uint64_t>(h[10]) << 16) | (static_cast<uint64_t>(h[11]) << 24));
    const uint64_t off = h[6] == 1 ? 10 : 12;
    data_off = off + hlen;
    const std::vector<uint8_t> all = read_at(in, at, data_off, path);
    return parse_npy(all.data(), all.size());
}
}   // namespace

int64_t NpzFile::length(const std::string& key) const {
    const Entry& e = entry(key);
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path_);
    uint64_t off = 0;
    const NpyArray h = npy_header(in, e.data, path_, off);
    return h.shape.empty() ? 1 : h.shape[0];
}

std::vector<double> NpzFile::read_rows(const std::string& key, const std::vector<int64_t>& rows, int& ncol) const {
    const Entry& e = entry(key);
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path_);
    uint64_t off = 0;
    const NpyArray h = npy_header(in, e.data, path_, off);
    if (h.descr != "<f8" || h.shape.size() != 2) throw std::runtime_error("read_rows: '" + key + "' is not a 2-D float64 array");
    ncol = static_cast<int>(h.shape[1]);
    std::vector<double> out(rows.size() * static_cast<size_t>(ncol));
    const uint64_t row_bytes = static_cast<uint64_t>(ncol) * 8;
    size_t k = 0;
    while (k < rows.size()) {
        // a run of consecutive rows is one read
        size_t m = k + 1;
        while (m < rows.size() && rows[m] == rows[m - 1] + 1) ++m;
        if (rows[k] < 0 || rows[m - 1] >= h.shape[0]) throw std::out_of_range("read_rows: a row outside '" + key + "'");
        in.clear();
        in.seekg(static_cast<std::streamoff>(e.data + off + static_cast<uint64_t>(rows[k]) * row_bytes), std::ios::beg);
        in.read(reinterpret_cast<char*>(out.data() + k * static_cast<size_t>(ncol)), static_cast<std::streamsize>((m - k) * row_bytes));
        if (!in) throw std::runtime_error("Truncated npz file " + path_);
        k = m;
    }
    return out;
}

}   // namespace fv
