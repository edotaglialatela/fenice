#include "fv/Geometry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fv {

namespace {

constexpr double kPi = 3.14159265358979323846;

// "x y" on one line (more columns are ignored); false for any other line.
bool parse_pair(const std::string& line, double& x, double& y) {
    const char* s = line.c_str();
    char* end = nullptr;
    x = std::strtod(s, &end);
    if (end == s) return false;
    const char* s2 = end;
    y = std::strtod(s2, &end);
    return end != s2;
}

}  // namespace

bool Polygon2D::contains(double x, double y) const {
    bool in = false;
    const size_t n = pts.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const double xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
        if ((yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) in = !in;
    }
    return in;
}

Polygon2D load_airfoil(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open airfoil file '" + path + "'");
    std::vector<std::pair<int, std::string>> lines;   // (line number, text), blank lines dropped
    std::string text;
    for (int no = 1; std::getline(in, text); ++no)
        if (text.find_first_not_of(" \t\r") != std::string::npos) lines.push_back({no, text});

    std::vector<std::array<double, 2>> data;
    std::vector<int> data_line;
    size_t k = 0;
    double a, b;
    if (!lines.empty() && !parse_pair(lines[0].second, a, b)) k = 1;   // name line
    // Lednicer: the first numeric line holds the two point counts (both > 1)
    bool lednicer = false;
    int n_up = 0, n_lo = 0;
    if (k < lines.size() && parse_pair(lines[k].second, a, b) && a > 1.0 && b > 1.0) {
        lednicer = true;
        n_up = static_cast<int>(a);
        n_lo = static_cast<int>(b);
        ++k;
    }
    for (; k < lines.size(); ++k) {
        if (!parse_pair(lines[k].second, a, b))
            throw std::runtime_error(path + ":" + std::to_string(lines[k].first) + ": expected 'x y', got '" +
                                     lines[k].second + "'");
        data.push_back({a, b});
        data_line.push_back(lines[k].first);
    }

    Polygon2D poly;
    if (lednicer) {
        if (static_cast<int>(data.size()) != n_up + n_lo)
            throw std::runtime_error(path + ": Lednicer header announces " + std::to_string(n_up) + " + " +
                                     std::to_string(n_lo) + " points, the file has " + std::to_string(data.size()));
        // Upper surface reversed (trailing edge -> leading edge), then the lower surface
        for (int i = n_up - 1; i >= 0; --i) poly.pts.push_back(data[i]);
        for (int i = n_up; i < n_up + n_lo; ++i)
            if (i > n_up || data[i] != data[0]) poly.pts.push_back(data[i]);   // skip the repeated leading edge
    } else {
        poly.pts = data;
    }
    // A closed file repeats the first point at the end
    if (poly.pts.size() > 1 && poly.pts.front() == poly.pts.back()) poly.pts.pop_back();
    if (poly.pts.size() < 3) throw std::runtime_error(path + ": an airfoil needs at least 3 points");

    // Normalise: leading edge (smallest x) at the origin, chord 1 along x
    const auto le = std::min_element(poly.pts.begin(), poly.pts.end(),
                                     [](const auto& p, const auto& q) { return p[0] < q[0]; });
    const double x0 = (*le)[0], y0 = (*le)[1];
    double xmax = x0;
    for (const auto& p : poly.pts) xmax = std::max(xmax, p[0]);
    const double c = xmax - x0;
    if (!(c > 0.0)) throw std::runtime_error(path + ": the coordinates have zero chord");
    for (auto& p : poly.pts) p = {(p[0] - x0) / c, (p[1] - y0) / c};
    return poly;
}

std::string naca_digits(const std::string& shape) {
    std::string s;
    for (const char ch : shape)
        if (!std::isspace(static_cast<unsigned char>(ch))) s += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (s.size() != 8 || s.compare(0, 4, "naca") != 0) return "";
    for (size_t i = 4; i < 8; ++i)
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return "";
    return s.substr(4);
}

Polygon2D naca4(const std::string& digits, int n) {
    if (digits.size() != 4 || !std::all_of(digits.begin(), digits.end(), [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); }))
        throw std::invalid_argument("NACA 4-digit code must be 4 digits, got '" + digits + "'");
    const double m = (digits[0] - '0') / 100.0, p = (digits[1] - '0') / 10.0;
    const double t = std::stoi(digits.substr(2)) / 100.0;
    if (!(t > 0.0)) throw std::invalid_argument("NACA " + digits + ": zero thickness");
    if (m > 0.0 && !(p > 0.0)) throw std::invalid_argument("NACA " + digits + ": camber needs a camber position");
    std::vector<std::array<double, 2>> up(n + 1), lo(n + 1);
    for (int i = 0; i <= n; ++i) {
        const double x = 0.5 * (1.0 - std::cos(kPi * i / n));   // cosine spacing: dense at both edges
        const double yt = 5.0 * t * (0.2969 * std::sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x -
                                     0.1036 * x * x * x * x);   // closed trailing edge
        double yc = 0.0, dyc = 0.0;
        if (m > 0.0) {
            if (x < p) { yc = m / (p * p) * (2.0 * p * x - x * x);  dyc = 2.0 * m / (p * p) * (p - x); }
            else       { yc = m / ((1 - p) * (1 - p)) * (1.0 - 2.0 * p + 2.0 * p * x - x * x);
                         dyc = 2.0 * m / ((1 - p) * (1 - p)) * (p - x); }
        }
        const double th = std::atan(dyc);
        up[i] = {x - yt * std::sin(th), yc + yt * std::cos(th)};
        lo[i] = {x + yt * std::sin(th), yc - yt * std::cos(th)};
    }
    Polygon2D poly;
    for (int i = n; i >= 0; --i) poly.pts.push_back(up[i]);   // TE -> LE over the upper surface
    for (int i = 1; i < n; ++i) poly.pts.push_back(lo[i]);    // LE -> TE (the TE point is shared)
    return poly;
}

Polygon2D place_airfoil(const Polygon2D& unit, double chord, double aoa_deg, double xq, double yq) {
    const double a = aoa_deg * kPi / 180.0, ca = std::cos(a), sa = std::sin(a);
    Polygon2D out;
    out.pts.reserve(unit.pts.size());
    for (const auto& p : unit.pts) {
        const double x = (p[0] - 0.25) * chord, y = p[1] * chord;   // quarter chord at the origin
        out.pts.push_back({xq + x * ca + y * sa, yq - x * sa + y * ca});   // clockwise by aoa: nose up
    }
    return out;
}

}  // namespace fv
