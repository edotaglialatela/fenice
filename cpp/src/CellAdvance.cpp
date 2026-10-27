#include "fv/CellAdvance.hpp"

#include <algorithm>
#include <stdexcept>

namespace fv {

double CellAdvance::step(double dt, double t) {
    if (static_cast<int>(mesh_.U.size()) != mesh_.num_leaves())
        throw std::logic_error("CellAdvance::step: the mesh has no state");
    engine_.force_state(mesh_, mesh_.U);   // body forcing of every level before the first pass
    const double S = advance(0, dt, t);
    return par ? par->reduce_max(S) : S;
}

double CellAdvance::advance(int l, double dt, double t) {
    const int max_l = mesh_.max_level;
    double S = 0.0;
    // a level that exists on any rank is advanced by every rank (the exchanges are collective): the whole mesh decides, not the
    // leaves a rank keeps
    const auto present = [&](int k) { return par ? par->level_present(k) : mesh_.level_begin[k + 1] > mesh_.level_begin[k]; };
    if (l < max_l && present(l + 1)) {
        engine_.clear_registers(mesh_, l + 1);
        for (int sub = 0; sub < 2; ++sub) S = std::max(S, advance(l + 1, dt / 2, t + sub * (dt / 2)));
        if (par) par->exchange_registers(l + 1);   // the coarse owners get the fine fluxes of their coarse-fine faces
    }
    if (present(l)) S = std::max(S, advance_own(l, dt, t));
    return S;
}

double CellAdvance::advance_own(int l, double dt, double t) {
    const int b = engine_.owned_begin(l), e = engine_.owned_end(l);   // the leaves of this level this rank advances
    const std::vector<double> weights = scheme_.stage_weights(), offsets = scheme_.stage_offsets();
    if (weights.size() != offsets.size()) throw std::logic_error("CellAdvance: inconsistent stage tables");
    size_t stage = 0;
    Field Ul(mesh_.U.begin() + b, mesh_.U.begin() + e);
    const ResidualFn res = [&](const Field& Uin, Field& R) {
        if (stage >= weights.size()) throw std::logic_error("CellAdvance: the scheme made more stages than it declares");
        std::copy(Uin.begin(), Uin.end(), mesh_.U.begin() + b);   // this level's stage state, seen by its neighbours
        if (par) par->exchange_stage(l);                          // ... including the ones on other ranks
        // stage time as TimeScheme::step() sets it
        const double o = offsets[stage];
        scheme_.stage_time = o == 0.0 ? t : (o == 1.0 ? t + dt : t + o * dt);
        engine_.stage_time = scheme_.stage_time;
        if (stage_hook) stage_hook(l, scheme_.stage_time);
        const double s = engine_.compute_level_fluxes(mesh_, mesh_.U, l);
        if (l > 0) engine_.accumulate_registers(mesh_, l, 0.5 * weights[stage]);
        engine_.gather_level(mesh_, l, flux_replacement, R);
        ++stage;
        return s;
    };
    const double S = scheme_.advance_fixed_dt(Ul, res, dt);
    if (stage != weights.size()) throw std::logic_error("CellAdvance: the scheme made fewer stages than it declares");
    std::copy(Ul.begin(), Ul.end(), mesh_.U.begin() + b);   // the result, not the last stage's input
    if (par) par->exchange_final(l);
    engine_.force_level(mesh_, mesh_.U, l);
    return S;
}

}   // namespace fv
