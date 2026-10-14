
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

// Report malformed coefficient records with their source line number.
// This makes coefficient-file errors easier to diagnose.
[[noreturn]] void invalid_coefficient_line(
    int line_number,
    const std::string& reason) {

  throw std::invalid_argument(
      "Invalid Pines coefficient record at line " +
      std::to_string(line_number) + ": " + reason);
}

}  // namespace

// Construct a Pines spherical-harmonic gravity model.
//
// References:
//
//     S. Pines, "Uniform Representation of the Gravitational Potential
//     and its Derivatives," AIAA Journal, Vol. 11, No. 11, 1973,
//     pp. 1508-1511.
//     DOI: https://doi.org/10.2514/3.50619
//
// The coefficient file must use the normalization and conventions
// expected by this implementation.
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

  // Limit the model to the supported spherical-harmonic degree.
  if (degree_ < 0 || degree_ > 360) {
    throw std::invalid_argument(
        "PinesGravity degree must be in [0, 360]");
  }

  // The gravitational parameter and reference radius must be positive
  // finite values to prevent invalid scaling in the gravity evaluation.
  if (!std::isfinite(mu_) || mu_ <= 0.0 ||
      !std::isfinite(reference_radius_) ||
      reference_radius_ <= 0.0) {

    throw std::invalid_argument(
        "PinesGravity mu and reference_radius must be finite and positive");
  }

  // This implementation uses fully normalized spherical-harmonic
  // coefficients. Other coefficient normalizations are not supported.
  if (normalization != "fully_normalized") {
    throw std::invalid_argument(
        "PinesGravity supports only fully_normalized coefficients");
  }

  // Triangular storage:
  // index(n, m) = n*(n+1)/2 + m, for 0 <= m <= n.
  //
  // Allocate coefficients through the requested degree. Each entry
  // represents one cosine coefficient C_nm and one sine coefficient S_nm.
  const std::size_t coefficient_count =
      index(degree_ + 1, 0);

  c_.assign(coefficient_count, 0.0);
  s_.assign(coefficient_count, 0.0);

  // The fully normalized spherical-harmonic expansion uses C_00 = 1.
  // Other coefficients default to zero until loaded from the file.
  c_[index(0, 0)] = 1.0;

  // Track coefficient records that have already been read so duplicate
  // degree/order pairs can be rejected rather than silently overwritten.
  std::vector<bool> seen(coefficient_count, false);

  // Open the gravity coefficient file.
  std::ifstream input(coefficient_file_);
  if (!input) {
    throw std::invalid_argument(
        "Cannot open Pines coefficient file: " + coefficient_file_);
  }

  // Use locale-independent decimal parsing for reproducible file reading.
  input.imbue(std::locale::classic());

  std::string line;
  int line_number = 0;
  int maximum_degree_in_file = -1;

  while (std::getline(input, line)) {
    ++line_number;

    // Accept '#' and '!' as comment markers. Ignore the comment suffix.
    const std::size_t comment = line.find_first_of("#!");
    if (comment != std::string::npos) {
      line.erase(comment);
    }

    // Skip blank lines and lines containing only whitespace.
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    // EGM96-style files may contain Fortran D exponents, such as
    // 1.234D-06. Convert these to E exponents for standard C++ parsing.
    std::replace(line.begin(), line.end(), 'D', 'E');
    std::replace(line.begin(), line.end(), 'd', 'e');

    std::istringstream row(line);
    row.imbue(std::locale::classic());

    int n = -1;
    int m = -1;
    double c = 0.0;
    double s = 0.0;

    // Each coefficient record must provide degree, order, C_nm, and S_nm.
    if (!(row >> n >> m >> c >> s)) {
      invalid_coefficient_line(
          line_number, "expected degree, order, C, and S");
    }

    // A spherical-harmonic order cannot exceed its degree.
    if (n < 0 || m < 0 || m > n) {
      invalid_coefficient_line(
          line_number, "require degree >= 0 and 0 <= order <= degree");
    }

    // Reject NaN and infinity before they enter the gravity model.
    if (!std::isfinite(c) || !std::isfinite(s)) {
      invalid_coefficient_line(
          line_number, "coefficients must be finite");
    }

    maximum_degree_in_file =
        std::max(maximum_degree_in_file, n);

    // Validate records across the file, but store only the coefficients
    // needed for the requested truncation degree.
    if (n > degree_) {
      continue;
    }

    const std::size_t k = index(n, m);

    // Duplicate records can indicate a malformed or concatenated file.
    if (seen[k]) {
      throw std::invalid_argument(
          "Duplicate Pines coefficient at line " +
          std::to_string(line_number));
    }

    seen[k] = true;
    c_[k] = c;
    s_[k] = s;
  }

  // Distinguish a read failure from normal end-of-file.
  if (input.bad()) {
    throw std::runtime_error(
        "Error reading Pines coefficient file: " + coefficient_file_);
  }

  // Ensure that the file contains records reaching at least the requested
  // degree. This check does not require every degree/order pair to exist;
  // omitted coefficients remain zero.
  if (maximum_degree_in_file < degree_) {
    throw std::invalid_argument(
        "Pines coefficient file does not reach the requested degree");
  }

  // Precompute recurrence coefficients used during gravity evaluation.
  //
  // The acceleration recurrence requires Legendre-function values
  // through degree + 1, so g_ and h_ need one additional row beyond
  // the requested gravity-field degree.
  const std::size_t recurrence_count =
      index(degree_ + 2, 0);

  g_.assign(recurrence_count, 0.0);
  h_.assign(recurrence_count, 0.0);

  // alpha_ and beta_ are used in the horizontal and radial recurrence
  // contributions. They use the same triangular indexing as C_nm/S_nm.
  alpha_.assign(coefficient_count, 0.0);
  beta_.assign(coefficient_count, 0.0);

  // Diagonal and superdiagonal factors are stored separately because
  // they are reused repeatedly during evaluation.
  diagonal_.assign(
      static_cast<std::size_t>(degree_) + 2, 0.0);

  superdiagonal_.resize(
      static_cast<std::size_t>(degree_) + 1);

  // Precompute normalized Legendre recurrence factors.
  //
  // g_[index(n,m)] and h_[index(n,m)] scale the previous one or two
  // degree rows when constructing the next recurrence row.
  // See reference [1] for the underlying Pines recurrence formulation.
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

      // h_ is needed for the two-row recurrence when n >= 2 and
      // the order is below the subdiagonal boundary.
      if (n >= 2 && n - m - 1 > 0) {
        h_[k] = std::sqrt(
            (nd + md - 1.0) * (2.0 * nd + 1.0) *
            (nd - md - 1.0) /
            ((nd + md) * (nd - md) * (2.0 * nd - 3.0)));
      }
    }
  }

  // Precompute alpha and beta factors for each coefficient.
  // These factors reduce repeated square-root calculations during
  // acceleration evaluations.
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

      // The order-zero terms use a sqrt(2) normalization adjustment
      // consistent with the recurrence convention used by the model.
      if (m == 0) {
        alpha_[k] /= sqrt_two;
        beta_[k] /= sqrt_two;
      }
    }
  }
}

// Evaluate gravitational acceleration at a Cartesian position.
//
// The input position and returned acceleration use the unit system
// associated with mu_ and reference_radius_. For example, if distances
// are in kilometres and mu_ is in km^3/s^2, acceleration is in km/s^2.
//
std::array<double, 3> PinesGravityModel::acceleration(
    const std::array<double, 3>& position) const {

  // Reject non-finite coordinates before computing the radius.
  for (double coordinate : position) {
    if (!std::isfinite(coordinate)) {
      throw std::invalid_argument(
          "Pines position must be finite");
    }
  }

  const double x = position[0];
  const double y = position[1];
  const double z = position[2];

  // Cartesian distance from the gravity-field origin.
  const double radius = std::sqrt(x * x + y * y + z * z);

  // The gravity evaluation is undefined at the origin. Also reject
  // overflow in the radius calculation.
  if (!(radius > 0.0) || !std::isfinite(radius)) {
    throw std::invalid_argument(
        "Pines position must have a finite, nonzero radius");
  }

  // Evaluate the Pines spherical-harmonic acceleration recurrence.
  const auto result = acceleration_scalar<double>(position);

  // Guard against overflow or non-finite results from extreme inputs
  // or unsuitable model parameters.
  for (double component : result) {
    if (!std::isfinite(component)) {
      throw std::overflow_error(
          "Pines acceleration overflow; check radius and model constants");
    }
  }

  return result;
}

}  // namespace ASSET
