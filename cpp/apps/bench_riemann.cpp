// Cost and bit-level fingerprint of the face-flux solvers: ns per call for each Riemann solver and EOS on a fixed
// set of state pairs (smooth flow with small jumps, plus strong shocks and vacuum-like states), and an order
// dependent hash of the bits of every flux and wave speed.  A change of a kernel that must not change the
// numerics keeps the hashes.
//   bench_riemann [calls]
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "fv/Schemes.hpp"

using namespace fv;
using Clock = std::chrono::steady_clock;

namespace {

struct Lcg {
    uint64_t s = 88172645463325252ull;
    double next() {   // uniform in [0, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    }
};

uint64_t bits(double x) {
    uint64_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

}   // namespace

int main(int argc, char** argv) {
    const int calls = argc > 1 ? std::atoi(argv[1]) : 4000000;
    const int pool = 1 << 16;
    for (int eos_kind = 0; eos_kind < 2; ++eos_kind) {
        const EOS eos = eos_kind == 0 ? EOS::ideal(1.4) : EOS::stiffened(4.4, 6.0e8);
        Lcg rng;
        std::vector<State> UL(pool), UR(pool);
        std::vector<int> axis(pool);
        for (int k = 0; k < pool; ++k) {
            const bool strong = rng.next() < 0.1;
            const double p0 = eos_kind == 0 ? 1.0 : 1.0e8;
            State a = {1.0 + 0.2 * rng.next(), 2.0 * rng.next() - 1.0, 2.0 * rng.next() - 1.0, p0 * (1.0 + 0.2 * rng.next()), 2.0 * rng.next() - 1.0};
            State b = a;
            const double jump = strong ? 4.0 : 0.02;
            b[0] *= 1.0 + jump * rng.next();
            b[1] += jump * (rng.next() - 0.5);
            b[2] += jump * (rng.next() - 0.5);
            b[3] *= 1.0 + jump * rng.next();
            b[4] += 0.1 * jump * (rng.next() - 0.5);
            UL[k] = eos.prim_to_cons(a);
            UR[k] = eos.prim_to_cons(b);
            axis[k] = static_cast<int>(rng.next() * 3.0) % 3;
        }
        // Edge cases overwrite the first entries: densities and pressures below the floor, a negative internal energy,
        // identical and mirrored states (S_M exactly 0 or nearly), the sound speed equal to the normal velocity to the last
        // bits (a wave speed that is 0 or one ulp off), signed zeros, a stationary contact, and the same with one ulp noise.
        {
            int k = 0;
            const double p0 = eos_kind == 0 ? 1.0 : 1.0e8;
            const auto put = [&](State l, State r, int ax) {
                UL[k] = l;
                UR[k] = r;
                axis[k] = ax;
                ++k;
            };
            for (int ax = 0; ax < 3; ++ax) {
                const State a = eos.prim_to_cons({1.0, 0.3, -0.2, p0, 0.1});
                put(a, a, ax);                                                               // identical states
                put(eos.prim_to_cons({1.0, 0.3, -0.2, p0, 0.1}), eos.prim_to_cons({1.0, -0.3, -0.2, p0, 0.1}), ax);   // mirrored normal velocity
                put(eos.prim_to_cons({1.0, 0.0, 0.0, p0, 0.0}), eos.prim_to_cons({0.125, -0.0, 0.0, 0.1 * p0, -0.0}), ax);
                put(eos.prim_to_cons({1.0, 0.0, 0.0, p0, 0.0}), eos.prim_to_cons({1.0, 0.0, 0.0, 2.0 * p0, 0.0}), ax);   // stationary pressure jump
                put({1e-20, 0.0, 0.0, 1e-20, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);          // density below the floor
                put({1.0, 5.0, 0.0, 0.1, 0.0}, {1.0, 5.0, 0.0, 0.1, 0.0}, ax);              // pressure floored (negative internal energy)
                put({-1.0, 1.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);              // negative density
                for (int ulp = -2; ulp <= 2; ++ulp) {                                         // normal velocity = sound speed, to the ulp
                    const double c = std::sqrt(eos.gamma * (p0 + (eos_kind == 0 ? 0.0 : eos.p_inf)) / 1.0);
                    double u = c;
                    for (int q = 0; q < std::abs(ulp); ++q) u = std::nextafter(u, ulp < 0 ? 0.0 : 1e300);
                    State W = {1.0, 0.0, 0.0, p0, 0.0};
                    W[ax == 0 ? 1 : (ax == 1 ? 2 : 4)] = u;
                    State Wr = W;
                    Wr[3] *= 1.0 + 1e-3;
                    put(eos.prim_to_cons(W), eos.prim_to_cons(Wr), ax);
                    put(eos.prim_to_cons(Wr), eos.prim_to_cons(W), ax);
                }
            }
        }
        for (const RiemannType type : {RiemannType::HLLC, RiemannType::LowMachHLLC, RiemannType::PreconditionedHLLC}) {
            const RiemannSolver rs(eos, type);
            uint64_t h = 1469598103934665603ull;
            double sink = 0.0;
            const bool batch = std::getenv("BENCH_BATCH") != nullptr;
            std::vector<State> F(pool);
            std::vector<double> SM(pool);
            const auto t0 = Clock::now();
            for (int n = 0; n < calls; n += pool) {
                if (batch) {
                    rs.face_flux_batch(UL.data(), UR.data(), axis.data(), pool, F.data(), SM.data());
                } else {
                    for (int k = 0; k < pool; ++k) F[k] = rs.face_flux(UL[k], UR[k], axis[k], SM[k]);
                }
                for (int k = 0; k < pool; ++k) {
                    for (int q = 0; q < kNVar; ++q) h = (h ^ bits(F[k][q])) * 1099511628211ull;
                    h = (h ^ bits(SM[k])) * 1099511628211ull;
                    sink += SM[k];
                }
            }
            const double s = std::chrono::duration<double>(Clock::now() - t0).count();
            std::printf("%s %-22s %7.1f ns/call  hash %016llx  (%g)\n", eos_kind == 0 ? "ideal    " : "stiffened",
                        type == RiemannType::HLLC ? "HLLC" : (type == RiemannType::LowMachHLLC ? "Low-Mach" : "Preconditioned"),
                        1e9 * s / calls, static_cast<unsigned long long>(h), sink);
        }
    }
    return 0;
}
