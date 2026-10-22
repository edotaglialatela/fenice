// Immersed-body geometry: closed 2-D polygons for airfoil sections, read from coordinate
// files (Selig or Lednicer format, as in the UIUC airfoil database) or generated from the
// NACA 4-digit formulas, then placed by chord, angle of attack and quarter-chord position.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace fv {

// A closed polygon: the last vertex connects back to the first.
struct Polygon2D {
    std::vector<std::array<double, 2>> pts;
    // Point-in-polygon by the even-odd rule.
    bool contains(double x, double y) const;
};

// Unit-chord airfoil from a coordinate file, leading edge at (0, 0) and chord along +x:
//   Selig:    optional name line, then "x y" around the profile (trailing edge, upper
//             surface, leading edge, lower surface, trailing edge);
//   Lednicer: name line, a line with the two point counts, then the upper and the lower
//             surface, each from the leading to the trailing edge.
// Throws std::runtime_error naming the file and line on a malformed file.
Polygon2D load_airfoil(const std::string& path);

// Unit-chord NACA 4-digit section ("2412": 2 % camber at 40 % chord, 12 % thick) with a
// closed trailing edge, `n` cosine-spaced points per surface.
Polygon2D naca4(const std::string& digits, int n = 200);

// The four digits of a "NACA dddd" shape name (case and spaces ignored), or "" for any other name.
std::string naca_digits(const std::string& shape);

// `unit` scaled to `chord`, rotated by `aoa_deg` about its quarter chord (positive: nose up
// for a flow along +x) and moved so the quarter chord sits at (xq, yq).
Polygon2D place_airfoil(const Polygon2D& unit, double chord, double aoa_deg, double xq, double yq);

}   // namespace fv
