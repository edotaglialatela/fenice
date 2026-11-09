#include "fv/Profile.hpp"
#include "fv/Switches.hpp"
#include "fv/Simulation.hpp"
#include "fv/ViscousBound.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <map>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fv/Geometry.hpp"
#include "fv/SolutionMapping.hpp"
#ifdef FV_USE_MPI
#include "fv/MPIDecomposition.hpp"
#endif

#ifndef FV_PROJECT_ROOT
#define FV_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

#include "SimulationInternal.hpp"
#include "SimulationPostprocess.hpp"

namespace fv {

using namespace sim_detail;

std::string Simulation::checkpoint_path() const {
    return (fs::path(exporter.output_dir()) / (file_prefix + "_restart.npz")).string();
}

namespace {
// Version of the checkpoint layout (1 = the keys written below). A file without the key is older than the key and is read as
// version 1 (missing keys have defaults); a file with a higher number is from a newer solver and is refused.
constexpr int kCheckpointFormat = 1;

void require_known_format(const std::string& path, bool present, double version) {
    if (present && version > kCheckpointFormat)
        throw std::runtime_error("The checkpoint " + path + " has format_version " + std::to_string(static_cast<int>(version)) +
                                 ", newer than this solver reads (" + std::to_string(kCheckpointFormat) + "): use a newer fenice");
}
}  // namespace

void Simulation::save_checkpoint(int step, const LeafArrays* leaves) {
    const std::string path = checkpoint_path();
    const std::string tmp = path.substr(0, path.size() - 4) + "_tmp.npz";
    std::vector<double> times;
    std::vector<std::string> files;
    for (const auto& e : exporter.collection) {
        times.push_back(e.time);
        files.push_back(e.filename);
    }
    NpzArchive a;
    if (cells) {
        // an adaptive run: the leaves (level and integer coordinates) and their state, and the wave-speed bound the next
        // time step is taken from
        // (an octree also stores k and the fifth state component rho w; the 2-D file is unchanged)
        const bool octree = cells->dim == 3;
        const LeafArrays lf = leaves ? *leaves : LeafArrays{&cells->level, &cells->ci, &cells->cj, &cells->U, octree ? &cells->ck : nullptr};
        const size_t n = lf.level->size();
        std::vector<double> lev(n), li(n), lj(n), lk(octree ? n : 0);
        for (size_t c = 0; c < n; ++c) {
            lev[c] = (*lf.level)[c];
            li[c] = (*lf.ci)[c];
            lj[c] = (*lf.cj)[c];
            if (octree) lk[c] = (*lf.ck)[c];
        }
        a["U"] = field_to_npy(*lf.U, octree ? kNVar : 4);
        a["leaf_level"] = NpyArray::from_doubles(lev, {static_cast<int64_t>(n)});
        a["leaf_i"] = NpyArray::from_doubles(li, {static_cast<int64_t>(n)});
        a["leaf_j"] = NpyArray::from_doubles(lj, {static_cast<int64_t>(n)});
        if (octree) a["leaf_k"] = NpyArray::from_doubles(lk, {static_cast<int64_t>(n)});
        a["S"] = NpyArray::from_double(cell_speed_);
    } else {
        a["U"] = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    }
    a["format_version"] = NpyArray::from_int(kCheckpointFormat);
    a["t"] = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    a["nx"] = NpyArray::from_int(mesh.nx);
    a["ny"] = NpyArray::from_int(mesh.ny);
    a["Lx"] = NpyArray::from_double(mesh.Lx);
    a["Ly"] = NpyArray::from_double(mesh.Ly);
    a["nz"] = NpyArray::from_int(mesh.nz);
    a["Lz"] = NpyArray::from_double(mesh.Lz);
    a["collection_time"] = NpyArray::from_doubles(times, {static_cast<int64_t>(times.size())});
    a["collection_file"] = NpyArray::from_strings(files);
    write_npz(tmp, a);
    rename_replace(tmp, path);
}

int Simulation::load_checkpoint() {
    const std::string path = checkpoint_path();
    if (!fs::exists(path)) {
        std::printf("\n[WARNING] No checkpoint found in %s. Starting from zero.\n", path.c_str());
        return 0;
    }
    const NpzFile file(path);
    require_known_format(path, file.has("format_version"), file.has("format_version") ? file.read("format_version").scalar() : 0);
    const bool adaptive_file = file.has("leaf_level");
    // Several ranks keeping only their shares: the leaves (integers) give the topology every rank cuts its share from, and each rank
    // reads only the state rows of its own leaves and their halo; the whole state and the whole mesh are never held
    const bool share = adaptive_file && cells && cell_par && !replicated_storage();
    NpzArchive a;
    if (share) {
        for (const char* key : {"t", "step", "S", "collection_time", "collection_file", "collection"})
            if (file.has(key)) a[key] = file.read(key);
    } else {
        a = read_npz(path);
    }
    if (adaptive_file != (cells != nullptr))
        throw std::runtime_error(std::string("The checkpoint ") + path + " is from " +
                                 (adaptive_file ? "an adaptive (AMR)" : "a uniform-mesh") + " run: resume it with use_amr " +
                                 (adaptive_file ? "on" : "off") + " (a uniform checkpoint can seed an AMR run with a hot start)");
    if (adaptive_file) {
        // a 2-D checkpoint has no leaf_k; an octree checkpoint is resumed on the mesh it was written for
        const bool file_3d = file.has("leaf_k");
        if (file_3d != (cells->dim == 3))
            throw std::runtime_error(std::string("The checkpoint ") + path + " is from a " + (file_3d ? "3-D" : "2-D") +
                                     " adaptive run: resume it with nz " + (file_3d ? "> 1" : "= 1"));
        if (file_3d) {
            const auto num = [&](const char* key) { return file.read(key).scalar(); };
            if (static_cast<int>(num("nx")) != mesh.nx || static_cast<int>(num("ny")) != mesh.ny || static_cast<int>(num("nz")) != mesh.nz ||
                std::abs(num("Lx") - mesh.Lx) > 1e-10 || std::abs(num("Ly") - mesh.Ly) > 1e-10 || std::abs(num("Lz") - mesh.Lz) > 1e-10)
                throw std::runtime_error("The checkpoint " + path + " is from a base mesh of another size or domain: use_hot_start maps it onto this one");
        }
    }
    if (share) {
        const auto ints = [&](const char* key) {
            const std::vector<double> d = file.read(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        {
            const bool octree = cells->dim == 3;
            std::vector<int> lv = ints("leaf_level"), li = ints("leaf_i"), lj = ints("leaf_j"), lk = octree ? ints("leaf_k") : std::vector<int>();
            CellMesh topo(*cells);
            topo.U = Field();
            topo.set_faceless(true);
            if (octree) topo.assign_leaves(lv, li, lj, lk, nullptr);
            else topo.assign_leaves(lv, li, lj, nullptr);   // sorts by (level, Morton) and checks that the leaves tile the domain
            std::vector<int> file_row(static_cast<size_t>(topo.num_leaves()), -1);   // row of the checkpoint of each leaf of the topology
            for (size_t r = 0; r < lv.size(); ++r)
                file_row[static_cast<size_t>(topo.find(lv[r], li[r], lj[r], octree ? lk[r] : 0))] = static_cast<int>(r);
            lv = li = lj = lk = std::vector<int>();
            cell_par->resume_share(topo, [&](CellMesh& m) {
                const size_t n = static_cast<size_t>(m.num_leaves());
                std::vector<int64_t> rows(n);
                for (size_t c = 0; c < n; ++c)
                    rows[c] = file_row[static_cast<size_t>(topo.find(m.level[c], m.ci[c], m.cj[c], octree ? m.ck[c] : 0))];
                int ncol = 0;
                const std::vector<double> d = file.read_rows("U", rows, ncol);
                if (ncol != 4 && ncol != kNVar) throw std::runtime_error("Expected an (n, 4) or (n, 5) state array");
                if (octree && ncol != kNVar) throw std::runtime_error("The checkpoint of a 3-D adaptive run needs an (n, 5) state array");
                m.U.assign(n, State{});
                for (size_t c = 0; c < n; ++c)
                    for (int q = 0; q < ncol; ++q) m.U[c][static_cast<size_t>(q)] = d[c * static_cast<size_t>(ncol) + static_cast<size_t>(q)];
            });
        }
        share_resumed_ = true;
        cell_speed_ = a.count("S") ? a.at("S").scalar() : 0.0;
    } else if (cells) {
        const auto ints = [&](const char* key) {
            const std::vector<double> d = a.at(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        Field Uc = npy_to_field(a.at("U"));
        if (cells->dim == 3) {
            if (a.at("U").shape.size() != 2 || a.at("U").shape[1] != kNVar)
                throw std::runtime_error("The checkpoint of a 3-D adaptive run needs an (n, 5) state array");
            cells->assign_leaves(ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), ints("leaf_k"), &Uc);
        } else {
            cells->assign_leaves(ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), &Uc);
        }
        cell_engine->build(*cells);
        cell_speed_ = a.count("S") ? a.at("S").scalar() : 0.0;
    } else {
        U = npy_to_field(a.at("U"));
    }
    t = a.at("t").scalar();
    const int step = static_cast<int>(a.at("step").scalar());
    exporter.collection.clear();
    if (a.count("collection_time") && a.count("collection_file")) {
        const auto times = a.at("collection_time").as_doubles();
        const auto files = a.at("collection_file").as_strings();
        for (size_t k = 0; k < std::min(times.size(), files.size()); ++k)
            exporter.collection.push_back({times[k], files[k]});
    } else if (a.count("collection")) {
        std::printf("[RESTART] Legacy checkpoint (pickled index): .pvd index restarts from this point\n");
    }
    std::printf("\n[RESTART] Simulation resumed from t = %.5f s (Step %d)\n", t, step);
    return step;
}

Field Simulation::load_hot_start(const std::string& path) {
    // If the source is this run's own checkpoint, back it up before the first save overwrites it.
    const std::string ck = checkpoint_path();
    std::error_code ec;
    if (fs::exists(path) && fs::exists(ck) && fs::equivalent(path, ck, ec)) {
        std::string backup = ck;
        backup.replace(backup.size() - std::string("_restart.npz").size(), std::string::npos, "_hotstart_source.npz");
        fs::copy_file(path, backup, fs::copy_options::overwrite_existing);
        std::printf("[Hot Start] Source checkpoint backed up to: %s\n", backup.c_str());
    }
    const NpzArchive a = read_npz(path);
    require_known_format(path, a.count("format_version") > 0, a.count("format_version") ? a.at("format_version").scalar() : 0);
    int nx_c, ny_c;
    double Lx_c, Ly_c;
    if (a.count("nx")) {
        nx_c = static_cast<int>(a.at("nx").scalar());
        ny_c = static_cast<int>(a.at("ny").scalar());
        Lx_c = a.at("Lx").scalar();
        Ly_c = a.at("Ly").scalar();
    } else {
        nx_c = p.hot_start_src_nx;
        ny_c = p.hot_start_src_ny;
        if (nx_c <= 0 || ny_c <= 0)
            throw std::runtime_error("Hot-start file '" + path +
                                     "' has no grid metadata and no manual Nx / Ny were provided "
                                     "(set hot_start_src_nx / hot_start_src_ny).");
        Lx_c = mesh.Lx;
        Ly_c = mesh.Ly;
        std::printf("[Hot Start] No metadata in file - using manual dims: Nx=%d, Ny=%d, Lx=%g, Ly=%g\n", nx_c, ny_c,
                    Lx_c, Ly_c);
    }
    if (std::abs(Lx_c - mesh.Lx) > 1e-10 || std::abs(Ly_c - mesh.Ly) > 1e-10)
        std::printf("[WARNING] Hot-start domain (%g x %g) differs from target (%g x %g). "
                    "Proceeding with coordinate rescaling.\n",
                    Lx_c, Ly_c, mesh.Lx, mesh.Ly);
    int nz_c = a.count("nz") ? static_cast<int>(a.at("nz").scalar()) : 1;
    if (nz_c > 1 && !mesh.is3d()) throw std::runtime_error("Hot start from a 3-D checkpoint needs a 3-D run (nz > 1)");
    std::printf("[Hot Start] Mapping %d x %d x %d -> %d x %d x %d ...\n", nx_c, ny_c, nz_c, mesh.nx, mesh.ny, mesh.nz);
    Field Uc = npy_to_field(a.at("U"));
    hot_src_.clear();
    hot_src_nx_ = hot_src_ny_ = 0;
    hot_src_nz_ = 1;
    if (a.count("leaf_level")) {
        // the checkpoint of an adaptive run: its leaves on the uniform mesh of their finest level (nx, ny, nz are the base mesh)
        const bool src_3d = a.count("leaf_k") > 0;
        if (src_3d != (nz_c > 1)) throw std::runtime_error("The adaptive checkpoint has inconsistent leaves (leaf_k and nz disagree)");
        if (mesh.is3d() != src_3d)
            throw std::runtime_error(std::string("Hot start from a ") + (src_3d ? "3-D" : "2-D") + " adaptive (AMR) checkpoint needs a " +
                                     (src_3d ? "3-D" : "2-D") + " run");
        const auto ints = [&](const char* key) {
            const std::vector<double> d = a.at(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        int nxf = 0, nyf = 0, nzf = 1;
        if (src_3d) {
            Uc = leaves_to_uniform_3d(Uc, ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), ints("leaf_k"), nx_c, ny_c, nz_c, nxf, nyf, nzf);
            std::printf("[Hot Start] Adaptive checkpoint: %d x %d x %d base mesh, finest level as %d x %d x %d cells\n", nx_c, ny_c, nz_c,
                        nxf, nyf, nzf);
        } else {
            Uc = leaves_to_uniform(Uc, ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), nx_c, ny_c, nxf, nyf);
            std::printf("[Hot Start] Adaptive checkpoint: %d x %d base mesh, finest level as %d x %d cells\n", nx_c, ny_c, nxf, nyf);
        }
        nx_c = nxf;
        ny_c = nyf;
        nz_c = nzf;
    }
    if (cells && a.count("leaf_level") && (nx_c != mesh.nx || ny_c != mesh.ny || nz_c != mesh.nz)) {
        // an adaptive target samples the finest-level field of an adaptive source at the leaf centres: keep it at its own
        // resolution, not on the base mesh (a uniform source is mapped onto the base mesh first)
        hot_src_ = Uc;
        hot_src_nx_ = nx_c;
        hot_src_ny_ = ny_c;
        hot_src_nz_ = nz_c;
    }
    if (static_cast<int>(Uc.size()) != nx_c * ny_c * nz_c)
        throw std::runtime_error("Hot-start state has " + std::to_string(Uc.size()) + " cells, expected " +
                                 std::to_string(nx_c * ny_c * nz_c));
    if (nz_c > 1) return map_solution_3d(Uc, nx_c, ny_c, nz_c, mesh.nx, mesh.ny, mesh.nz, mesh.Lx, mesh.Ly, mesh.Lz);
    Field U2 = map_solution(Uc, nx_c, ny_c, mesh.nx, mesh.ny, mesh.Lx, mesh.Ly);
    if (!mesh.is3d()) return U2;
    Field U3(static_cast<size_t>(mesh.num_cells()));   // 2-D source on a 3-D mesh: extrude along z
    for (int k = 0; k < mesh.nz; ++k)
        std::copy(U2.begin(), U2.end(), U3.begin() + static_cast<std::ptrdiff_t>(k) * U2.size());
    return U3;
}

void Simulation::dump_final_state(int step, const LeafArrays* leaves) const {
    if (p.dump_state.empty()) return;
    NpzArchive a;
    a["t"] = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    if (cells) {
        const bool octree = cells->dim == 3;
        const LeafArrays lf = leaves ? *leaves : LeafArrays{&cells->level, &cells->ci, &cells->cj, &cells->U, octree ? &cells->ck : nullptr};
        const size_t n = lf.level->size(), cols = octree ? 7 : 5;   // x0, x1, y0, y1[, z0, z1], level
        std::vector<double> meta;
        meta.reserve(n * cols);
        for (size_t k = 0; k < n; ++k) {
            const int l = (*lf.level)[k], i = (*lf.ci)[k], j = (*lf.cj)[k];
            meta.insert(meta.end(), {cells->x0 + i * cells->dx(l), cells->x0 + (i + 1) * cells->dx(l),
                                     cells->y0 + j * cells->dy(l), cells->y0 + (j + 1) * cells->dy(l)});
            if (octree) {
                const int z = (*lf.ck)[k];
                meta.insert(meta.end(), {cells->z0 + z * cells->dz(l), cells->z0 + (z + 1) * cells->dz(l)});
            }
            meta.push_back(static_cast<double>(l));
        }
        a["U"] = field_to_npy(*lf.U, octree ? kNVar : 4);   // (3-D: rho w too)
        a["leaf_meta"] = NpyArray::from_doubles(meta, {static_cast<int64_t>(n), static_cast<int64_t>(cols)});
    } else {
        a["U"] = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    }
    write_npz(p.dump_state, a);
}

}   // namespace fv
