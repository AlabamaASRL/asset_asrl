#include "PinesGravityModel.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace ASSET {
namespace {
constexpr double sqrt_two = 1.4142135623730950488;
}

PinesGravityModel::PinesGravityModel(const std::string& coefficient_file, int degree,
                                   double mu, double reference_radius,
                                   const std::string& normalization)
    : degree_(degree), mu_(mu), reference_radius_(reference_radius),
      coefficient_file_(coefficient_file) {
  // Initial supported range covers the full EGM96 release. It also bounds
  // allocations from configuration errors before reading the input file.
  if (degree < 0 || degree > 360)
    throw std::invalid_argument("PinesGravity degree must be in [0, 360]");
  if (!std::isfinite(mu) || mu <= 0 || !std::isfinite(reference_radius) || reference_radius <= 0)
    throw std::invalid_argument("PinesGravity mu and reference_radius must be finite and positive");
  if (normalization != "fully_normalized")
    throw std::invalid_argument("PinesGravity supports only fully_normalized coefficients");

  const auto count = index(degree + 1, 0);
  c_.assign(count, 0.0);
  s_.assign(count, 0.0);
  c_[0] = 1.0;  // EGM96 .nor files start at n=2; degree one defaults to zero.
  std::vector<bool> seen(count, false);
  std::ifstream input(coefficient_file);
  if (!input) throw std::invalid_argument("Cannot open Pines coefficient file: " + coefficient_file);
  std::string line;
  int line_number = 0, maximum_degree = -1;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment = line.find_first_of("#!");
    if (comment != std::string::npos) line.erase(comment);
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    // Accept Fortran D exponents as well as E exponents.
    std::replace(line.begin(), line.end(), 'D', 'E');
    std::replace(line.begin(), line.end(), 'd', 'e');
    std::istringstream row(line);
    row.imbue(std::locale::classic());
    int n, m;
    double c, s;
    if (!(row >> n >> m >> c >> s) || n < 0 || m < 0 || m > n ||
        !std::isfinite(c) || !std::isfinite(s))
      throw std::invalid_argument("Invalid Pines coefficient record at line " + std::to_string(line_number));
    // Additional columns (e.g. EGM96 coefficient uncertainties) are ignored.
    maximum_degree = std::max(maximum_degree, n);
    if (n > degree) continue;
    const auto k = index(n, m);
    if (seen[k])
      throw std::invalid_argument("Duplicate Pines coefficient at line " + std::to_string(line_number));
    seen[k] = true;
    c_[k] = c;
    s_[k] = s;
  }
  if (input.bad()) throw std::runtime_error("Error reading Pines coefficient file: " + coefficient_file);
  if (maximum_degree < degree)
    throw std::invalid_argument("Pines coefficient file does not reach the requested degree");

  g_.assign(index(degree + 2, 0), 0.0);
  h_.assign(g_.size(), 0.0);
  alpha_.assign(count, 0.0);
  beta_.assign(count, 0.0);
  diagonal_.assign(degree + 2, 0.0);
  superdiagonal_.resize(degree + 1);
  for (int n = 1; n <= degree + 1; ++n) {
    diagonal_[n] = std::sqrt((2.0*n + 1.0) / (2.0*n));
    for (int m = 0; m < n; ++m) {
      const auto k = index(n, m);
      g_[k] = std::sqrt((2.0*n + 1.0) * (2.0*n - 1.0) / ((n - m) * (n + m)));
      if (n >= 2 && n - m - 1 > 0)
        h_[k] = std::sqrt((n + m - 1.0) * (2.0*n + 1.0) * (n - m - 1.0)
                         / ((n + m) * (n - m) * (2.0*n - 3.0)));
    }
  }
  for (int n = 0; n <= degree; ++n) {
    superdiagonal_[n] = std::sqrt(2.0*n + 3.0);
    for (int m = 0; m <= n; ++m) {
      const auto k = index(n, m);
      alpha_[k] = std::sqrt(double(n - m) * (n + m + 1.0));
      beta_[k] = std::sqrt((2.0*n + 1.0) * (n + m + 2.0) * (n + m + 1.0) / (2.0*n + 3.0));
      if (m == 0) {
        alpha_[k] /= sqrt_two;
        beta_[k] /= sqrt_two;
      }
    }
  }
}

std::array<double, 3> PinesGravityModel::acceleration(const std::array<double, 3>& position) const {
  for (double x : position)
    if (!std::isfinite(x)) throw std::invalid_argument("Pines position must be finite");
  const double radius = std::sqrt(position[0]*position[0] + position[1]*position[1] + position[2]*position[2]);
  if (!(radius > 0.0) || !std::isfinite(radius))
    throw std::invalid_argument("Pines position must have a finite, nonzero radius");
  const auto result = acceleration_scalar<double>(position);
  for (double x : result)
    if (!std::isfinite(x)) throw std::overflow_error("Pines acceleration overflow; check radius and model constants");
  return result;
}
}  // namespace ASSET
