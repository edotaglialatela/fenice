#pragma once
// The post-processing writer of a run (probes, line samples, volume means, force rows) and its due-tests; internal to Simulation*.cpp.
#include "fv/Simulation.hpp"
#include "SimulationInternal.hpp"

namespace fv {
namespace sim_detail {


inline std::string output_token(const std::string& name) {
    std::string out;
    for (unsigned char c : name) {
        if (std::isalnum(c) || c == '-' || c == '_') out.push_back(static_cast<char>(c));
        else if (!out.empty() && out.back() != '_') out.push_back('_');
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "output" : out;
}

// A resumed run appends to the file of the stopped one. Rows past the restart step (written after the last checkpoint of a run that
// did not stop gracefully) are dropped first, so that no step appears twice.
inline void drop_rows_after(const fs::path& path, int step) {
    std::ifstream in(path, std::ios::binary);   // binary: the line ends (CR LF on Windows) are kept as written
    if (!in) return;
    std::string line, kept;
    bool first = true, dropped = false;
    while (std::getline(in, line)) {
        if (!first && std::atoi(line.c_str()) > step) {
            dropped = true;
            continue;
        }
        first = false;
        kept += line;
        kept.push_back(static_cast<char>(10));   // LF
    }
    in.close();
    if (!dropped) return;
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out << kept;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("Cannot rewrite post-processing output " + path.string() + ": " + ec.message());
}

inline std::FILE* open_post_csv(const fs::path& path, int start_step, const std::string& header) {
    std::error_code ec;
    const bool existed = start_step > 0 && fs::exists(path, ec);
    if (existed) drop_rows_after(path, start_step);
    std::FILE* f = std::fopen(path.string().c_str(), start_step > 0 ? "a" : "w");
    if (!f) throw std::runtime_error("Cannot open post-processing output " + path.string());
    if (!existed) std::fprintf(f, "%s\n", header.c_str());
    return f;
}

inline int bounded_cell(double x, double origin, double spacing, int count) {
    int i = static_cast<int>(std::floor((x - origin) / spacing));
    return std::max(0, std::min(count - 1, i));
}

inline int leaf_at_point(const CellMesh& cm, const std::vector<int>& levels, const std::vector<int>& ci,
                  const std::vector<int>& cj, double x, double y, const std::vector<int>* ck = nullptr, double z = 0.0) {
    for (int l = cm.max_level; l >= 0; --l) {
        const int i = bounded_cell(x, cm.x0, cm.dx(l), cm.nx_at(l));
        const int j = bounded_cell(y, cm.y0, cm.dy(l), cm.ny_at(l));
        const int k = ck ? bounded_cell(z, cm.z0, cm.dz(l), cm.nz_at(l)) : 0;
        const uint64_t key = ck ? cm.key_of(i, j, k) : morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
        auto first = std::lower_bound(levels.begin(), levels.end(), l);
        auto last = std::upper_bound(first, levels.end(), l);
        size_t lo = static_cast<size_t>(first - levels.begin());
        size_t hi = static_cast<size_t>(last - levels.begin());
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const uint64_t mid_key = ck ? cm.key_of(ci[mid], cj[mid], (*ck)[mid])
                                        : morton_key(static_cast<uint32_t>(ci[mid]), static_cast<uint32_t>(cj[mid]));
            if (mid_key < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < static_cast<size_t>(last - levels.begin()) && ci[lo] == i && cj[lo] == j && (!ck || (*ck)[lo] == k))
            return static_cast<int>(lo);
    }
    return -1;
}

class PostprocessWriter {
public:
    PostprocessWriter(const TestCase& tc, const std::string& output_dir, const std::string& prefix, int start_step)
        : probes_(tc.probes), samples_(tc.samples), forces_(tc.forces), averages_(tc.averages),
          avg_state_(tc.averages.size()), last_probe_(tc.probes.size(), -1), last_sample_(tc.samples.size(), -1),
          last_force_(tc.forces.size(), -1), dir_(output_dir), prefix_(prefix), start_step_(start_step) {
        try {
            for (const ProbeSpec& s : probes_)
                probe_files_.push_back(open_csv(s.name + "_probe", "step,t,x,y,z,cell_x,cell_y,cell_z,rho,u,v,p,w"));
            for (const SampleSpec& s : samples_)
                sample_files_.push_back(open_csv(s.name + "_sample", "step,t,point,x,y,z,cell_x,cell_y,cell_z,rho,u,v,p,w"));
            for (const ForceSpec& s : forces_)
                force_files_.push_back(open_csv(s.name + "_force", "step,t,Fx,Fy,Mz,Cd,Cl,Cm"));
            for (const AverageSpec& s : averages_) {
                std::string header = "step,t";
                for (const std::string& var : s.variables) header += "," + var + "_time_average";
                average_files_.push_back(open_csv(s.name + "_average", header));
            }
        } catch (...) {
            close_all();
            throw;
        }
    }

    ~PostprocessWriter() { close_all(); }
    PostprocessWriter(const PostprocessWriter&) = delete;
    PostprocessWriter& operator=(const PostprocessWriter&) = delete;

    bool empty() const { return probes_.empty() && samples_.empty() && forces_.empty() && averages_.empty(); }
    bool has_averages() const { return !averages_.empty(); }
    bool has_force_due(int step, bool force = false) const {
        for (size_t i = 0; i < forces_.size(); ++i)
            if ((force || step % forces_[i].interval == 0) && last_force_[i] != step) return true;
        return false;
    }

    void skip_step(int step) {
        std::fill(last_probe_.begin(), last_probe_.end(), step);
        std::fill(last_sample_.begin(), last_sample_.end(), step);
        std::fill(last_force_.begin(), last_force_.end(), step);
        for (AverageState& state : avg_state_) state.last_written_step = step;
    }

    void write_uniform(const Mesh& mesh, const EOS& eos, const Field& U, int step, double time, bool force = false) {
        for (size_t n = 0; n < probes_.size(); ++n) {
            if (!due(step, probes_[n].interval, force, last_probe_[n])) continue;
            const ProbeSpec& p = probes_[n];
            const int i = bounded_cell(p.x, mesh.x0, mesh.dx, mesh.nx);
            const int j = bounded_cell(p.y, mesh.y0, mesh.dy, mesh.ny);
            const int k = mesh.is3d() ? bounded_cell(p.z, mesh.z0, mesh.dz, mesh.nz) : 0;
            const State w = eos.cons_to_prim(U[static_cast<size_t>(mesh.cell(i, j, k))]);
            write_point(probe_files_[n], step, time, -1, p.x, p.y, p.z, mesh.xc(i), mesh.yc(j),
                        mesh.is3d() ? mesh.zc(k) : 0.0, w);
            last_probe_[n] = step;
        }
        for (size_t n = 0; n < samples_.size(); ++n) {
            if (!due(step, samples_[n].interval, force, last_sample_[n])) continue;
            const SampleSpec& s = samples_[n];
            for (int point = 0; point < s.points; ++point) {
                const double f = static_cast<double>(point) / static_cast<double>(s.points - 1);
                const double x = s.from[0] + f * (s.to[0] - s.from[0]);
                const double y = s.from[1] + f * (s.to[1] - s.from[1]);
                const double z = s.from.size() == 3 ? s.from[2] + f * (s.to[2] - s.from[2]) : 0.0;
                const int i = bounded_cell(x, mesh.x0, mesh.dx, mesh.nx);
                const int j = bounded_cell(y, mesh.y0, mesh.dy, mesh.ny);
                const int k = mesh.is3d() ? bounded_cell(z, mesh.z0, mesh.dz, mesh.nz) : 0;
                const State w = eos.cons_to_prim(U[static_cast<size_t>(mesh.cell(i, j, k))]);
                write_point(sample_files_[n], step, time, point, x, y, z, mesh.xc(i), mesh.yc(j),
                            mesh.is3d() ? mesh.zc(k) : 0.0, w);
            }
            last_sample_[n] = step;
        }
    }

    // `ck`: the k of the leaves of an octree (null in 2-D)
    void write_amr(const CellMesh& mesh, const EOS& eos, const std::vector<int>& levels, const std::vector<int>& ci,
                   const std::vector<int>& cj, const Field& U, int step, double time, bool force = false,
                   const std::vector<int>* ck = nullptr) {
        for (size_t n = 0; n < probes_.size(); ++n) {
            if (!due(step, probes_[n].interval, force, last_probe_[n])) continue;
            const ProbeSpec& p = probes_[n];
            const int c = leaf_at_point(mesh, levels, ci, cj, p.x, p.y, ck, p.z);
            if (c < 0) throw std::logic_error("post-processing probe did not map to an AMR leaf");
            const State w = eos.cons_to_prim(U[static_cast<size_t>(c)]);
            const int l = levels[static_cast<size_t>(c)];
            const double cx = mesh.x0 + (ci[static_cast<size_t>(c)] + 0.5) * mesh.dx(l);
            const double cy = mesh.y0 + (cj[static_cast<size_t>(c)] + 0.5) * mesh.dy(l);
            const double cz = ck ? mesh.z0 + ((*ck)[static_cast<size_t>(c)] + 0.5) * mesh.dz(l) : 0.0;
            write_point(probe_files_[n], step, time, -1, p.x, p.y, ck ? p.z : 0.0, cx, cy, cz, w);
            last_probe_[n] = step;
        }
        for (size_t n = 0; n < samples_.size(); ++n) {
            if (!due(step, samples_[n].interval, force, last_sample_[n])) continue;
            const SampleSpec& s = samples_[n];
            for (int point = 0; point < s.points; ++point) {
                const double f = static_cast<double>(point) / static_cast<double>(s.points - 1);
                const double x = s.from[0] + f * (s.to[0] - s.from[0]);
                const double y = s.from[1] + f * (s.to[1] - s.from[1]);
                const double z = ck && s.from.size() == 3 ? s.from[2] + f * (s.to[2] - s.from[2]) : 0.0;
                const int c = leaf_at_point(mesh, levels, ci, cj, x, y, ck, z);
                if (c < 0) throw std::logic_error("post-processing sample did not map to an AMR leaf");
                const State w = eos.cons_to_prim(U[static_cast<size_t>(c)]);
                const int l = levels[static_cast<size_t>(c)];
                const double cx = mesh.x0 + (ci[static_cast<size_t>(c)] + 0.5) * mesh.dx(l);
                const double cy = mesh.y0 + (cj[static_cast<size_t>(c)] + 0.5) * mesh.dy(l);
                const double cz = ck ? mesh.z0 + ((*ck)[static_cast<size_t>(c)] + 0.5) * mesh.dz(l) : 0.0;
                write_point(sample_files_[n], step, time, point, x, y, z, cx, cy, cz, w);
            }
            last_sample_[n] = step;
        }
    }

    void write_forces(int step, double time, const std::array<double, 3>& loads, const Mesh& mesh, bool force = false) {
        for (size_t n = 0; n < forces_.size(); ++n) {
            if (!due(step, forces_[n].interval, force, last_force_[n])) continue;
            const ForceSpec& s = forces_[n];
            const double q = 0.5 * s.reference_density * s.reference_velocity * s.reference_velocity;
            const double area = s.reference_length * (mesh.is3d() ? mesh.Lz : 1.0);
            const double scale = q * area;
            std::fprintf(force_files_[n], "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", step, time, loads[0],
                         loads[1], loads[2], scale > 0.0 ? loads[0] / scale : 0.0,
                         scale > 0.0 ? loads[1] / scale : 0.0,
                         scale > 0.0 ? -loads[2] / (scale * s.reference_length) : 0.0);
            std::fflush(force_files_[n]);
            last_force_[n] = step;
        }
    }

    void observe_averages(int step, double time, const std::array<double, 5>& means, bool force = false, bool emit = true) {
        for (size_t n = 0; n < averages_.size(); ++n) {
            const AverageSpec& spec = averages_[n];
            AverageState& state = avg_state_[n];
            std::vector<double> current;
            current.reserve(spec.variables.size());
            for (const std::string& name : spec.variables) current.push_back(means[primitive_index(name)]);
            if (!state.initialized) {
                state.start = state.previous_time = time;
                state.previous = current;
                state.integral.assign(current.size(), 0.0);
                state.initialized = true;
            } else if (time > state.previous_time) {
                const double dt = time - state.previous_time;
                for (size_t q = 0; q < current.size(); ++q)
                    state.integral[q] += 0.5 * (state.previous[q] + current[q]) * dt;
                state.previous_time = time;
                state.previous = current;
            }
            if (!emit) {
                state.last_written_step = step;
            } else if ((force || step % spec.interval == 0) && state.last_written_step != step) {
                std::fprintf(average_files_[n], "%d,%.17e", step, time);
                const double elapsed = time - state.start;
                for (size_t q = 0; q < current.size(); ++q)
                    std::fprintf(average_files_[n], ",%.17e", elapsed > 0.0 ? state.integral[q] / elapsed : current[q]);
                std::fprintf(average_files_[n], "\n");
                std::fflush(average_files_[n]);
                state.last_written_step = step;
            }
        }
    }

    void reset_averages(double time, const std::array<double, 5>& means) {
        for (size_t n = 0; n < averages_.size(); ++n) {
            const AverageSpec& spec = averages_[n];
            AverageState& state = avg_state_[n];
            state = AverageState{};
            for (const std::string& name : spec.variables) state.previous.push_back(means[primitive_index(name)]);
            state.integral.assign(state.previous.size(), 0.0);
            state.start = state.previous_time = time;
            state.initialized = true;
        }
    }

private:
    struct AverageState {
        bool initialized = false;
        double start = 0.0, previous_time = 0.0;
        int last_written_step = -1;
        std::vector<double> previous, integral;
    };

    std::FILE* open_csv(const std::string& suffix, const std::string& header) {
        const fs::path path = fs::path(dir_) / (prefix_ + "_" + output_token(suffix) + ".csv");
        std::FILE* f = open_post_csv(path, start_step_, header);
        files_.push_back(f);
        return f;
    }

    static bool due(int step, int interval, bool force, int last) { return (force || step % interval == 0) && last != step; }

    static int primitive_index(const std::string& name) {
        if (name == "rho") return 0;
        if (name == "u") return 1;
        if (name == "v") return 2;
        if (name == "p") return 3;
        return 4;
    }

    static void write_point(std::FILE* f, int step, double time, int point, double x, double y, double z, double cx,
                            double cy, double cz, const State& w) {
        if (point < 0)
            std::fprintf(f, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", step,
                         time, x, y, z, cx, cy, cz, w[0], w[1], w[2], w[3], w[4]);
        else
            std::fprintf(f, "%d,%.17e,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", step,
                         time, point, x, y, z, cx, cy, cz, w[0], w[1], w[2], w[3], w[4]);
        std::fflush(f);
    }

    void close_all() {
        for (std::FILE* f : files_)
            if (f) std::fclose(f);
        files_.clear();
    }

    std::vector<ProbeSpec> probes_;
    std::vector<SampleSpec> samples_;
    std::vector<ForceSpec> forces_;
    std::vector<AverageSpec> averages_;
    std::vector<AverageState> avg_state_;
    std::vector<int> last_probe_, last_sample_, last_force_;
    std::vector<std::FILE*> files_, probe_files_, sample_files_, force_files_, average_files_;
    std::string dir_, prefix_;
    int start_step_;
};

inline std::array<double, 5> volume_primitive_means(const EOS& eos, const Mesh& mesh, const Field& U) {
    std::array<double, 5> sums{};
    const double volume = mesh.cell_volume();
    for (const State& u : U) {
        const State w = eos.cons_to_prim(u);
        for (int q = 0; q < 5; ++q) sums[static_cast<size_t>(q)] += w[static_cast<size_t>(q)] * volume;
    }
    const double total = static_cast<double>(U.size()) * volume;
    for (double& x : sums) x /= total;
    return sums;
}

inline std::array<double, 5> volume_primitive_means(const EOS& eos, const CellMesh& mesh, const std::vector<int>& levels,
                                             const Field& U) {
    std::array<double, 5> sums{};
    double total = 0.0;
    for (size_t c = 0; c < levels.size(); ++c) {
        const double area = mesh.dim == 3 ? mesh.dx(levels[c]) * mesh.dy(levels[c]) * mesh.dz(levels[c])   // (the volume)
                                          : mesh.dx(levels[c]) * mesh.dy(levels[c]);
        const State w = eos.cons_to_prim(U[c]);
        for (int q = 0; q < 5; ++q) sums[static_cast<size_t>(q)] += w[static_cast<size_t>(q)] * area;
        total += area;
    }
    for (double& x : sums) x /= total;
    return sums;
}

inline bool has_postprocess(const TestCase& tc) {
    return !tc.probes.empty() || !tc.samples.empty() || !tc.forces.empty() || !tc.averages.empty();
}

inline bool postprocess_state_due(const TestCase& tc, int step) {
    for (const ProbeSpec& p : tc.probes) if (step % p.interval == 0) return true;
    for (const SampleSpec& s : tc.samples) if (step % s.interval == 0) return true;
    for (const ForceSpec& f : tc.forces) if (step % f.interval == 0) return true;
    return !tc.averages.empty();   // a time average is integrated at every completed step
}

inline bool amr_postprocess_state_due(const TestCase& tc, int step) {
    for (const ProbeSpec& p : tc.probes) if (step % p.interval == 0) return true;
    for (const SampleSpec& s : tc.samples) if (step % s.interval == 0) return true;
    return !tc.averages.empty();   // force outputs are computed from the adaptive residual separately
}

inline bool postprocess_force_due(const TestCase& tc, int step) {
    for (const ForceSpec& f : tc.forces) if (step % f.interval == 0) return true;
    return false;
}


}  // namespace sim_detail
}  // namespace fv
