#include "PinesGravityModel.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ASSET {
namespace {

constexpr double sqrt_two = 1.4142135623730950488;

[[noreturn]] void invalid_coefficient_line(
    int line_number,
    const std::string& reason) {

  throw std::invalid_argument(
      "Invalid Pines coefficient record at line " +
      std::to_string(line_number) + ": " + reason);
}

}  // namespace

PinesGravityModel::PinesGravityModel(
    const std::string& coefficient_file,
    int degree,
    double mu,
    double reference_radius,
    const std::string& normalization)
    : degree_(degree),
      mu_(mu),
      reference_radius_(reference_radius),
      coefficient_file_(coefficient_file) {

  if (degree_ < 0 || degree_ > 360) {
    throw std::invalid_argument(
        "PinesGravity degree must be in [0, 360]");
  }

  if (!std::isfinite(mu_) || mu_ <= 0.0 ||
      !std::isfinite(reference_radius_) ||
      reference_radius_ <= 0.0) {

    throw std::invalid_argument(
        "PinesGravity mu and reference_radius must be finite and positive");
  }

  if (normalization != "fully_normalized") {
    throw std::invalid_argument(
        "PinesGravity supports only fully_normalized coefficients");
  }

  const std::size_t coefficient_count =
      index(degree_ + 1, 0);

  c_.assign(coefficient_count, 0.0);
  s_.assign(coefficient_count, 0.0);

  // Fully normalized gravity-field convention.
  c_[index(0, 0)] = 1.0;

  std::vector<bool> seen(coefficient_count, false);

  std::ifstream input(coefficient_file_);
  if (!input) {
    throw std::invalid_argument(
        "Cannot open Pines coefficient file: " + coefficient_file_);
  }

  input.imbue(std::locale::classic());

  std::string line;
  int line_number = 0;
  int maximum_degree_in_file = -1;

  while (std::getline(input, line)) {
    ++line_number;

    const std::size_t comment = line.find_first_of("#!");
    if (comment != std::string::npos) {
      line.erase(comment);
    }

    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    // EGM96 files may use Fortran D exponents.
    std::replace(line.begin(), line.end(), 'D', 'E');
    std::replace(line.begin(), line.end(), 'd', 'e');

    std::istringstream row(line);
    row.imbue(std::locale::classic());

    int n = -1;
    int m = -1;
    double c = 0.0;
    double s = 0.0;

    if (!(row >> n >> m >> c >> s)) {
      invalid_coefficient_line(
          line_number, "expected degree, order, C, and S");
    }

    if (n < 0 || m < 0 || m > n) {
      invalid_coefficient_line(
          line_number, "require degree >= 0 and 0 <= order <= degree");
    }

    if (!std::isfinite(c) || !std::isfinite(s)) {
      invalid_coefficient_line(
          line_number, "coefficients must be finite");
    }

    maximum_degree_in_file =
        std::max(maximum_degree_in_file, n);

    // Parse and validate all records, but store only requested degrees.
    if (n > degree_) {
      continue;
    }

    const std::size_t k = index(n, m);

    if (seen[k]) {
      throw std::invalid_argument(
          "Duplicate Pines coefficient at line " +
          std::to_string(line_number));
    }

    seen[k] = true;
    c_[k] = c;
    s_[k] = s;
  }

  if (input.bad()) {
    throw std::runtime_error(
        "Error reading Pines coefficient file: " + coefficient_file_);
  }

  if (maximum_degree_in_file < degree_) {
    throw std::invalid_argument(
        "Pines coefficient file does not reach the requested degree");
  }

  // Recurrence coefficients require rows through degree + 1.
  const std::size_t recurrence_count =
      index(degree_ + 2, 0);

  g_.assign(recurrence_count, 0.0);
  h_.assign(recurrence_count, 0.0);

  alpha_.assign(coefficient_count, 0.0);
  beta_.assign(coefficient_count, 0.0);

  diagonal_.assign(
      static_cast<std::size_t>(degree_) + 2, 0.0);

  superdiagonal_.resize(
      static_cast<std::size_t>(degree_) + 1);

  for (int n = 1; n <= degree_ + 1; ++n) {
    const double nd = static_cast<double>(n);

    diagonal_[n] =
        std::sqrt((2.0 * nd + 1.0) / (2.0 * nd));

    for (int m = 0; m < n; ++m) {
      const double md = static_cast<double>(m);
      const std::size_t k = index(n, m);

      g_[k] = std::sqrt(
          (2.0 * nd + 1.0) * (2.0 * nd - 1.0) /
          ((nd - md) * (nd + md)));

      if (n >= 2 && n - m - 1 > 0) {
        h_[k] = std::sqrt(
            (nd + md - 1.0) * (2.0 * nd + 1.0) *
            (nd - md - 1.0) /
            ((nd + md) * (nd - md) * (2.0 * nd - 3.0)));
      }
    }
  }

  for (int n = 0; n <= degree_; ++n) {
    const double nd = static_cast<double>(n);

    superdiagonal_[n] =
        std::sqrt(2.0 * nd + 3.0);

    for (int m = 0; m <= n; ++m) {
      const double md = static_cast<double>(m);
      const std::size_t k = index(n, m);

      alpha_[k] =
          std::sqrt((nd - md) * (nd + md + 1.0));

      beta_[k] = std::sqrt(
          (2.0 * nd + 1.0) *
          (nd + md + 2.0) *
          (nd + md + 1.0) /
          (2.0 * nd + 3.0));

      if (m == 0) {
        alpha_[k] /= sqrt_two;
        beta_[k] /= sqrt_two;
      }
    }
  }
}

std::array<double, 3> PinesGravityModel::acceleration(
    const std::array<double, 3>& position) const {

  for (double coordinate : position) {
    if (!std::isfinite(coordinate)) {
      throw std::invalid_argument(
          "Pines position must be finite");
    }
  }

  const double x = position[0];
  const double y = position[1];
  const double z = position[2];

  const double radius = std::sqrt(x * x + y * y + z * z);

  if (!(radius > 0.0) || !std::isfinite(radius)) {
    throw std::invalid_argument(
        "Pines position must have a finite, nonzero radius");
  }

  const auto result = acceleration_scalar<double>(position);

  for (double component : result) {
    if (!std::isfinite(component)) {
      throw std::overflow_error(
          "Pines acceleration overflow; check radius and model constants");
    }
  }

  return result;
}

}  // namespace ASSET
