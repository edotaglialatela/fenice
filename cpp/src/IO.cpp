#include "fv/IO.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "fv/AMR.hpp"
#include "fv/SolutionMapping.hpp"

namespace fs = std::filesystem;

namespace fv {

namespace {

void write_file_atomic(const fs::path& path, const std::string& content) {
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write " + tmp.string());
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    fs::rename(tmp, path);   // atomic replace: readers never see partial files
}

// Replace NaN/Inf/subnormal values so ParaView's float32 parser accepts them.
inline double sanitize(double x) {
    if (std::isnan(x)) return 0.0;
    if (std::isinf(x)) return x > 0 ? 1e30 : -1e30;
    if (std::abs(x) < 1e-30) return 0.0;
    return x;
}

inline void append_fmt(std::string& s, const char* fmt, double v) {
    char buf[64];
    const int n = std::snprintf(buf, sizeof(buf), fmt, v);
    s.append(buf, static_cast<size_t>(n));
}

// One RectilinearGrid piece; `level` < 0 means no AMR_Level field.
std::string build_vtr(const EOS& eos, const Field& U, int nx, int ny, double x0, double x1,
                      double y0, double y1, int level, int nz = 1, double z0 = 0.0, double z1 = 0.0) {
    const int n = nx * ny * nz;
    const bool three_d = nz > 1;
    std::vector<double> rho(n), u(n), v(n), w(n), p(n), T(n), mach(n);
    for (int c = 0; c < n; ++c) {
        const State W = eos.cons_to_prim(U[c]);
        rho[c] = sanitize(W[0]); u[c] = sanitize(W[1]); v[c] = sanitize(W[2]); p[c] = sanitize(W[3]);
        w[c]   = sanitize(W[4]);
        T[c]   = sanitize(p[c] / (rho[c] + 1e-12));
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
        std::snprintf(head, sizeof(head), "        <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\">\n", c.first);
        s += head;
        for (double x : *c.second) { append_fmt(s, "%.6e", x); s += '\n'; }
        s += "        </DataArray>\n";
    }
    s += "      </Coordinates>\n";

    s += "      <CellData Scalars=\"Density\" Vectors=\"Velocity\">\n";
    auto scalar = [&](const char* name, const std::vector<double>& d) {
        std::snprintf(head, sizeof(head),
                      "        <DataArray type=\"Float32\" Name=\"%s\" NumberOfComponents=\"1\" format=\"ascii\">\n", name);
        s += head;
        for (double x : d) { append_fmt(s, "%.5e", x); s += '\n'; }
        s += "        </DataArray>\n";
    };
    scalar("Density", rho);
    scalar("Pressure", p);
    scalar("Temperature_nondim", T);
    scalar("Mach", mach);
    s += "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c = 0; c < n; ++c) {
        append_fmt(s, "%.5e", u[c]); s += ' ';
        append_fmt(s, "%.5e", v[c]); s += ' ';
        append_fmt(s, "%.5e", three_d ? w[c] : 0.0);  s += '\n';
    }
    s += "        </DataArray>\n";
    if (level >= 0) scalar("AMR_Level", std::vector<double>(n, static_cast<double>(level)));
    s += "      </CellData>\n";
    s += "    </Piece>\n";
    s += "  </RectilinearGrid>\n";
    s += "</VTKFile>\n";
    return s;
}

std::string step_name(const std::string& prefix, int step, const char* ext) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%05d", step);
    return prefix + buf + ext;
}

}  // namespace

// ---------------------------------------------------------------------------
// VTKExporter
// ---------------------------------------------------------------------------

VTKExporter::VTKExporter(std::string output_dir, bool verbose) : dir_(std::move(output_dir)) {
    fs::create_directories(dir_);
    if (verbose) std::printf("[EXPORTER] Output directory: '%s'\n", dir_.c_str());
}

void VTKExporter::export_vtr(const std::string& prefix, const EOS& eos, const Field& U, const Mesh& mesh,
                             double time, int step) {
    const std::string name = step_name(prefix, step, ".vtr");
    write_file_atomic(fs::path(dir_) / name, build_vtr(eos, U, mesh.nx, mesh.ny, 0.0, mesh.Lx, 0.0, mesh.Ly, -1,
                                                       mesh.nz, 0.0, mesh.Lz));
    collection.push_back({time, name});
}

void VTKExporter::export_amr(const std::string& prefix, const EOS& eos, const AMRTree& tree, double time, int step) {
    const std::string block_dir = step_name(prefix, step, "");
    fs::create_directories(fs::path(dir_) / block_dir);

    std::vector<std::string> leaf_files;
    int idx = 0;
    for (const AMRPatch* patch : tree.leaves()) {
        char base[32];
        std::snprintf(base, sizeof(base), "p%04d.vtr", idx++);
        write_file_atomic(fs::path(dir_) / block_dir / base,
                          build_vtr(eos, patch->U, patch->nx(), patch->ny(), patch->xmin, patch->xmax,
                                    patch->ymin, patch->ymax, patch->level));
        leaf_files.push_back(block_dir + "/" + base);
    }

    // Group this snapshot's leaves into one multi-block file; only the .vtm is
    // registered in the .pvd (ParaView's PVD reader is one-file-per-timestep).
    const std::string vtm = step_name(prefix, step, ".vtm");
    std::string s = "<?xml version=\"1.0\"?>\n"
                    "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
                    "  <vtkMultiBlockDataSet>\n";
    char line[512];
    for (size_t k = 0; k < leaf_files.size(); ++k) {
        std::snprintf(line, sizeof(line), "    <DataSet index=\"%zu\" name=\"patch_%04zu\" file=\"%s\"/>\n",
                      k, k, leaf_files[k].c_str());
        s += line;
    }
    s += "  </vtkMultiBlockDataSet>\n</VTKFile>\n";
    write_file_atomic(fs::path(dir_) / vtm, s);
    collection.push_back({time, vtm});
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
    if (t == "f8") get(double{});
    else if (t == "f4") get(float{});
    else if (t == "i8") get(int64_t{});
    else if (t == "i4") get(int32_t{});
    else if (t == "u8") get(uint64_t{});
    else if (t == "u4") get(uint32_t{});
    else if (t == "b1") get(uint8_t{});
    else throw std::runtime_error("Unsupported .npy dtype '" + descr + "'");
    return out;
}

std::vector<std::string> NpyArray::as_strings() const {
    if (descr.size() < 3 || descr[1] != 'U') throw std::runtime_error("Not a unicode .npy array: " + descr);
    const int64_t width = std::stoll(descr.substr(2));
    std::vector<std::string> out;
    for (int64_t k = 0; k < size(); ++k) {
        std::string s;
        for (int64_t c = 0; c < width; ++c) {
            uint32_t cp;
            std::memcpy(&cp, data.data() + (k * width + c) * 4, 4);
            if (cp == 0) break;
            if (cp < 0x80) s += static_cast<char>(cp);
            else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
            else if (cp < 0x10000) { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
            else { s += static_cast<char>(0xF0 | (cp >> 18)); s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
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
    out += '\x01'; out += '\x00';
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
    if (major == 1) { hlen = p[8] | (p[9] << 8); off = 10; }
    else { hlen = p[8] | (p[9] << 8) | (static_cast<size_t>(p[10]) << 16) | (static_cast<size_t>(p[11]) << 24); off = 12; }
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

uint32_t crc32(const uint8_t* d, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void put16(std::string& s, uint32_t v) { s += static_cast<char>(v & 0xFF); s += static_cast<char>((v >> 8) & 0xFF); }
void put32(std::string& s, uint32_t v) { put16(s, v & 0xFFFF); put16(s, v >> 16); }

uint64_t get(const std::vector<uint8_t>& b, size_t at, int bytes) {
    if (at + bytes > b.size()) throw std::runtime_error("Truncated zip archive");
    uint64_t v = 0;
    for (int k = bytes - 1; k >= 0; --k) v = (v << 8) | b[at + k];
    return v;
}

}  // namespace

void write_npz(const std::string& path, const NpzArchive& arrays) {
    std::string zip, central;
    uint32_t count = 0;
    for (const auto& kv : arrays) {
        const std::string name = kv.first + ".npy";
        const std::string body = npy_bytes(kv.second);
        if (body.size() >= 0xFFFFFFFFull || zip.size() >= 0xFFFFFFFFull)
            throw std::runtime_error("Checkpoint exceeds 4 GB (zip64 writing not supported)");
        const uint32_t crc = crc32(reinterpret_cast<const uint8_t*>(body.data()), body.size());
        const uint32_t offset = static_cast<uint32_t>(zip.size());

        zip += "PK\x03\x04";
        put16(zip, 20); put16(zip, 0); put16(zip, 0);       // version, flags, stored
        put16(zip, 0); put16(zip, 0x21);                    // time, date (1980-01-01)
        put32(zip, crc); put32(zip, static_cast<uint32_t>(body.size())); put32(zip, static_cast<uint32_t>(body.size()));
        put16(zip, static_cast<uint32_t>(name.size())); put16(zip, 0);
        zip += name;
        zip += body;

        central += "PK\x01\x02";
        put16(central, 20); put16(central, 20); put16(central, 0); put16(central, 0);
        put16(central, 0); put16(central, 0x21);
        put32(central, crc); put32(central, static_cast<uint32_t>(body.size())); put32(central, static_cast<uint32_t>(body.size()));
        put16(central, static_cast<uint32_t>(name.size())); put16(central, 0); put16(central, 0);
        put16(central, 0); put16(central, 0); put32(central, 0);
        put32(central, offset);
        central += name;
        ++count;
    }
    const uint32_t cd_offset = static_cast<uint32_t>(zip.size());
    zip += central;
    zip += "PK\x05\x06";
    put16(zip, 0); put16(zip, 0); put16(zip, count); put16(zip, count);
    put32(zip, static_cast<uint32_t>(central.size())); put32(zip, cd_offset); put16(zip, 0);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write " + path);
    out.write(zip.data(), static_cast<std::streamsize>(zip.size()));
}

NpzArchive read_npz(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (b.size() < 22) throw std::runtime_error("Not a zip archive: " + path);

    // End of central directory (scan backwards over a possible comment)
    size_t eocd = std::string::npos;
    const size_t lo = b.size() > 65557 ? b.size() - 65557 : 0;
    for (size_t k = b.size() - 22 + 1; k-- > lo;)
        if (get(b, k, 4) == 0x06054b50) { eocd = k; break; }
    if (eocd == std::string::npos) throw std::runtime_error("No end-of-central-directory in " + path);
    uint64_t entries = get(b, eocd + 10, 2);
    uint64_t cd_off  = get(b, eocd + 16, 4);
    if ((entries == 0xFFFF || cd_off == 0xFFFFFFFF) && eocd >= 20 && get(b, eocd - 20, 4) == 0x07064b50) {
        const size_t z = static_cast<size_t>(get(b, eocd - 20 + 8, 8));
        entries = get(b, z + 32, 8);
        cd_off  = get(b, z + 48, 8);
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
                if (usize == 0xFFFFFFFF) { usize = get(b, y, 8); y += 8; }
                if (csize == 0xFFFFFFFF) { csize = get(b, y, 8); y += 8; }
                if (loff  == 0xFFFFFFFF) { loff  = get(b, y, 8); y += 8; }
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

}  // namespace fv
