#include "fv/CellAdvance.hpp"

#include "fv/Profile.hpp"

#include <algorithm>
#include <stdexcept>

namespace fv {

namespace {

// dst[0, n) = src[0, n), on the threads of the face loops (a level's state is a memory-bound copy)
void copy_states(const State* src, int n, State* dst) {
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) dst[c] = src[c];
}

}   // namespace

void CellAdvance::enable_positivity_fallback(std::function<bool(const State&)> admissible) {
    admissible_ = std::move(admissible);
    guard_.mark = [this](const Field& cand) {
        std::vector<char>& fl = engine_.fallback_flags;
        if (static_cast<int>(fl.size()) != mesh_.num_leaves()) fl.assign(static_cast<size_t>(mesh_.num_leaves()), 0);
        const int n = static_cast<int>(cand.size());
        bool grew = false;
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int c = 0; c < n; ++c) {
            const size_t leaf = static_cast<size_t>(cur_b_ + c);
            if (!fl[leaf] && !admissible_(cand[static_cast<size_t>(c)])) {
                fl[leaf] = 1;
#pragma omp critical(fenice_amr_fallback_grew)
                grew = true;
            }
        }
        // every rank takes the same decision: the stage has collectives (exchange_stage), so the ranks repeat it together
        if (par) grew = par->reduce_max(grew ? 1.0 : 0.0) > 0.0;
        if (grew) {
            if (par) par->exchange_flags(cur_level_, fl);   // the flags of the halo leaves of a face this rank computes too
            engine_.fallback_on = true;
            repeat_ = true;
            ++repairs_;
        }
        return grew;
    };
    guard_.clear = [this]() {
        repeat_ = false;
        if (!engine_.fallback_on) return;
        std::vector<char>& fl = engine_.fallback_flags;
        for (int c = engine_.owned_begin(cur_level_); c < engine_.owned_end(cur_level_); ++c) flagged_ += fl[static_cast<size_t>(c)] ? 1 : 0;
        std::fill(fl.begin(), fl.end(), char(0));
        engine_.fallback_on = false;
    };
    scheme_.set_guard(&guard_);
}

CellAdvance::~CellAdvance() {
    if (admissible_) scheme_.set_guard(nullptr);
}

double CellAdvance::step(double dt, double t) {
    if (static_cast<int>(mesh_.U.size()) != mesh_.num_leaves())
        throw std::logic_error("CellAdvance::step: the mesh has no state");
    {
        ProfileScope ps(Section::AmrForce);
        engine_.force_state(mesh_, mesh_.U);   // body forcing of every level before the first pass
    }
    const double S = advance(0, dt, t);
    return par ? par->reduce_max(S) : S;
}

double CellAdvance::advance(int l, double dt, double t) {
    const int max_l = mesh_.max_level;
    double S = 0.0;
    // a level that exists on any rank is advanced by every rank (the exchanges are collective): the whole mesh decides, not the
    // leaves a rank keeps
    const auto present = [&](int k) { return par ? par->level_present(k) : mesh_.level_begin[k + 1] > mesh_.level_begin[k]; };
    if (hancock_ && present(l)) {
        // MUSCL-Hancock: this level's leaves are predicted by half of their own step from the state of the start of the step, before the finer
        // levels move (they read the coarse side of their jump from here, in both of their sub-steps)
        ProfileScope ps(Section::AmrFluxes);
        scheme_.stage_time = t + 0.5 * dt;
        engine_.stage_time = scheme_.stage_time;
        if (stage_hook) stage_hook(l, scheme_.stage_time);
        if (par) {
            // ranks: the limited coarse side of a coarse-fine face comes from the fine leaves' owner, and the predicted side states go
            // to the ranks that compute a face without owning the leaf
            engine_.predict_limit(mesh_, mesh_.U, l);
            par->exchange_limited(l);
            engine_.predict_cells(mesh_, l, 0.5 * dt);
            par->exchange_predicted(l);
        } else {
            engine_.predict_level(mesh_, mesh_.U, l, 0.5 * dt);
        }
    }
    if (l < max_l && present(l + 1)) {
        {
            ProfileScope ps(Section::AmrRegisters);
            engine_.clear_registers(mesh_, l + 1);
        }
        for (int sub = 0; sub < 2; ++sub) S = std::max(S, advance(l + 1, dt / 2, t + sub * (dt / 2)));
        if (par) {
            ProfileScope ps(Section::AmrExchange);
            par->exchange_registers(l + 1);   // the coarse owners get the fine fluxes of their coarse-fine faces
        }
    }
    if (present(l)) S = std::max(S, advance_own(l, dt, t));
    return S;
}

double CellAdvance::advance_own(int l, double dt, double t) {
    const int b = engine_.owned_begin(l), e = engine_.owned_end(l);   // the leaves of this level this rank advances
    const std::vector<double> weights = scheme_.stage_weights(), offsets = scheme_.stage_offsets();
    if (weights.size() != offsets.size()) throw std::logic_error("CellAdvance: inconsistent stage tables");
    size_t stage = 0;
    if (ul_.size() != static_cast<size_t>(mesh_.max_level) + 1) {
        ul_.assign(static_cast<size_t>(mesh_.max_level) + 1, {});
        work_.assign(static_cast<size_t>(mesh_.max_level) + 1, {});
    }
    cur_level_ = l;
    cur_b_ = b;
    Field& Ul = ul_[static_cast<size_t>(l)];
    Ul.resize(static_cast<size_t>(e - b));
    copy_states(mesh_.U.data() + b, e - b, Ul.data());
    const ResidualFn res = [&](const Field& Uin, Field& R) {
        // a stage the guard repeats: the same stage again (same time, same weight), its registers replace the first pass's
        const bool again = repeat_;
        repeat_ = false;
        if (again && stage == 0) throw std::logic_error("CellAdvance: a repeat before the first stage");
        const size_t st = again ? stage - 1 : stage;
        if (st >= weights.size()) throw std::logic_error("CellAdvance: the scheme made more stages than it declares");
        ProfileScope ps_stage(Section::AmrStage);
        // this level's stage state, seen by its neighbours (the first stage's is the state the mesh holds already)
        if (&Uin != &Ul) copy_states(Uin.data(), e - b, mesh_.U.data() + b);
        ps_stage.stop();
        if (par) {                                                // ... including the ones on other ranks
            ProfileScope ps(Section::AmrExchange);
            par->exchange_stage(l);
        }
        ProfileScope ps_time(Section::AmrStage);
        // stage time as TimeScheme::step() sets it
        const double o = offsets[st];
        scheme_.stage_time = o == 0.0 ? t : (o == 1.0 ? t + dt : t + o * dt);
        engine_.stage_time = scheme_.stage_time;
        if (stage_hook) stage_hook(l, scheme_.stage_time);
        ps_time.stop();
        double s;
        {
            ProfileScope ps(Section::AmrFluxes);
            s = engine_.compute_level_fluxes(mesh_, mesh_.U, l, hancock_ != nullptr);
        }
        if (l > 0) {
            ProfileScope ps(Section::AmrRegisters);
            if (admissible_) {
                if (again) engine_.restore_registers(l, reg_conv_, reg_visc_);
                else engine_.save_registers(l, reg_conv_, reg_visc_);
            }
            engine_.accumulate_registers(mesh_, l, 0.5 * weights[st]);
        }
        {
            ProfileScope ps(Section::AmrGather);
            engine_.gather_level(mesh_, l, flux_replacement, R, hancock_ != nullptr);
        }
        if (!again) ++stage;
        return s;
    };
    scheme_.swap_work(work_[static_cast<size_t>(l)]);   // the work fields of this level's size
    const double S = scheme_.advance_fixed_dt(Ul, res, dt);
    scheme_.swap_work(work_[static_cast<size_t>(l)]);
    if (stage != weights.size()) throw std::logic_error("CellAdvance: the scheme made fewer stages than it declares");
    copy_states(Ul.data(), e - b, mesh_.U.data() + b);   // the result, not the last stage's input
    if (par) {
        ProfileScope ps(Section::AmrExchange);
        par->exchange_final(l);
    }
    {
        ProfileScope ps(Section::AmrForce);
        engine_.force_level(mesh_, mesh_.U, l);
    }
    return S;
}

}   // namespace fv
