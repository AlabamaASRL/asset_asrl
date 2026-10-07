#include "PinesGravityModel.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace ASSET {
namespace {

constexpr double sqrt_two =
    1.4142135623730950488;

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
      inverse_reference_radius_(1.0 / reference_radius),
      coefficient_file_(coefficient_file) {

  // ==========================================================================
  // Validate inputs
  // ==========================================================================

  if (degree < 0 || degree > 360)
    throw std::invalid_argument(
        "PinesGravity degree must be in [0, 360]");

  if (!std::isfinite(mu) ||
      mu <= 0.0 ||
      !std::isfinite(reference_radius) ||
      reference_radius <= 0.0)
    throw std::invalid_argument(
        "PinesGravity mu and reference_radius must be finite and positive");

  if (normalization != "fully_normalized")
    throw std::invalid_argument(
        "PinesGravity supports only fully_normalized coefficients");


  // ==========================================================================
  // Coefficient storage
  // ==========================================================================

  const auto count =
      index(degree, degree) + 1;

  c_.assign(
      count,
      0.0);

  s_.assign(
      count,
      0.0);

  c_[0] = 1.0;

  // EGM96 .nor files start at n=2; degree-one terms default to zero.
  std::vector<bool> seen(
      count,
      false);


  // ==========================================================================
  // Read coefficient file
  // ==========================================================================

  std::ifstream input(
      coefficient_file);

  if (!input)
    throw std::invalid_argument(
        "Cannot open Pines coefficient file: " +
        coefficient_file);

  std::string line;

  int line_number = 0;
  int maximum_degree = -1;

  while (std::getline(input, line)) {

    ++line_number;

    // Remove comments.
    const auto comment =
        line.find_first_of("#!");

    if (comment != std::string::npos)
      line.erase(comment);

    // Ignore blank lines.
    if (line.find_first_not_of(" \t\r") ==
        std::string::npos)
      continue;

    // Accept Fortran D/d exponents as well as E/e exponents.
    std::replace(
        line.begin(),
        line.end(),
        'D',
        'E');

    std::replace(
        line.begin(),
        line.end(),
        'd',
        'e');

    std::istringstream row(line);

    row.imbue(
        std::locale::classic());

    int n;
    int m;
    double c;
    double s;

    if (!(row >> n >> m >> c >> s) ||
        n < 0 ||
        m < 0 ||
        m > n ||
        !std::isfinite(c) ||
        !std::isfinite(s)) {

      throw std::invalid_argument(
          "Invalid Pines coefficient record at line " +
          std::to_string(line_number));
    }

    // Additional columns, such as EGM96 coefficient uncertainties,
    // are intentionally ignored.

    maximum_degree =
        std::max(
            maximum_degree,
            n);

    if (n > degree)
      continue;

    const auto k =
        index(n, m);

    if (seen[k])
      throw std::invalid_argument(
          "Duplicate Pines coefficient at line " +
          std::to_string(line_number));

    seen[k] = true;

    c_[k] = c;
    s_[k] = s;
  }

  if (input.bad())
    throw std::runtime_error(
        "Error reading Pines coefficient file: " +
        coefficient_file);

  if (maximum_degree < degree)
    throw std::invalid_argument(
        "Pines coefficient file does not reach the requested degree");


  // ==========================================================================
  // Legendre recurrence coefficients
  // ==========================================================================

  // Need rows through degree + 1 for the recurrence.
  g_.assign(
      index(degree + 1, degree + 1) + 1,
      0.0);

  h_.assign(
      g_.size(),
      0.0);

  alpha_.assign(
      count,
      0.0);

  beta_.assign(
      count,
      0.0);

  diagonal_.assign(
      degree + 2,
      0.0);

  superdiagonal_.resize(
      degree + 1);


  // ==========================================================================
  // Precompute triangular row offsets
  // ==========================================================================

  row_offset_.resize(
      degree + 3);

  for (int n = 0;
       n <= degree + 2;
       ++n) {

    row_offset_[n] =
        static_cast<std::size_t>(n) *
        static_cast<std::size_t>(n + 1) /
        2;
  }


  // ==========================================================================
  // Legendre recurrence coefficients
  // ==========================================================================

  for (int n = 1;
       n <= degree + 1;
       ++n) {

    diagonal_[n] =
        std::sqrt(
            (2.0 * n + 1.0) /
            (2.0 * n));

    for (int m = 0;
         m < n;
         ++m) {

      const auto k =
          index(n, m);

      g_[k] =
          std::sqrt(
              (2.0 * n + 1.0) *
              (2.0 * n - 1.0) /
              ((n - m) *
               (n + m)));

      if (n >= 2 &&
          n - m - 1 > 0) {

        h_[k] =
            std::sqrt(
                (n + m - 1.0) *
                (2.0 * n + 1.0) *
                (n - m - 1.0) /
                ((n + m) *
                 (n - m) *
                 (2.0 * n - 3.0)));
      }
    }
  }


  // ==========================================================================
  // Derivative recurrence coefficients
  // ==========================================================================

  for (int n = 0;
       n <= degree;
       ++n) {

    superdiagonal_[n] =
        std::sqrt(
            2.0 * n + 3.0);

    for (int m = 0;
         m <= n;
         ++m) {

      const auto k =
          index(n, m);

      alpha_[k] =
          std::sqrt(
              static_cast<double>(n - m) *
              (n + m + 1.0));

      beta_[k] =
          std::sqrt(
              (2.0 * n + 1.0) *
              (n + m + 2.0) *
              (n + m + 1.0) /
              (2.0 * n + 3.0));

      if (m == 0) {
        alpha_[k] /= sqrt_two;
        beta_[k] /= sqrt_two;
      }
    }
  }


  // ==========================================================================
  // Precompute coefficient factors
  // ==========================================================================

  c_sqrt2_.resize(
      count);

  s_sqrt2_.resize(
      count);

  for (std::size_t i = 0;
       i < count;
       ++i) {

    c_sqrt2_[i] =
        sqrt_two * c_[i];

    s_sqrt2_[i] =
        sqrt_two * s_[i];
  }


  // ==========================================================================
  // Precompute m as double
  // ==========================================================================

  m_double_.resize(
      degree + 1);

  for (int m = 0;
       m <= degree;
       ++m) {

    m_double_[m] =
        static_cast<double>(m);
  }


  // ==========================================================================
  // Allocate reusable double-evaluation workspace
  // ==========================================================================

  const std::size_t a_count =
      row_offset_[degree + 2];

  const std::size_t width =
      static_cast<std::size_t>(degree) + 2;

  scratch_double_.resize(
      a_count +
      3 * width);
}


// ============================================================================
// Optimized double-precision gravity evaluation
// ============================================================================

std::array<double, 3>
PinesGravityModel::acceleration(
    const std::array<double, 3>& position) const {

  // ==========================================================================
  // Input validation
  // ==========================================================================

  for (double x : position) {

    if (!std::isfinite(x))
      throw std::invalid_argument(
          "Pines position must be finite");
  }


  // ==========================================================================
  // Position / radius
  // ==========================================================================

  const double x =
      position[0];

  const double y =
      position[1];

  const double z =
      position[2];

  const double radius =
      std::sqrt(
          x * x +
          y * y +
          z * z);

  if (!(radius > 0.0) ||
      !std::isfinite(radius)) {

    throw std::invalid_argument(
        "Pines position must have a finite, nonzero radius");
  }

  // Compute reciprocal once and reuse it.
  const double inverse_radius =
      1.0 / radius;


  // ==========================================================================
  // Normalized position
  // ==========================================================================

  const double xhat =
      x * inverse_radius;

  const double yhat =
      y * inverse_radius;

  const double zhat =
      z * inverse_radius;


  // ==========================================================================
  // Radius ratio
  // ==========================================================================

  const double ratio =
      reference_radius_ *
      inverse_radius;


  // ==========================================================================
  // Reusable workspace
  // ==========================================================================

  double* const a =
      scratch_double_.data();

  const std::size_t a_count =
      row_offset_[degree_ + 2];

  const std::size_t width =
      static_cast<std::size_t>(degree_) + 2;

  double* const rho =
      a + a_count;

  double* const re =
      rho + width;

  double* const im =
      re + width;


  // ==========================================================================
  // Initialize
  // ==========================================================================

  a[0] = 1.0;

  rho[0] =
      mu_ *
      inverse_radius;

  re[0] = 1.0;
  im[0] = 0.0;


  // ==========================================================================
  // Longitude recurrence
  // ==========================================================================

  for (int m = 1;
       m <= degree_ + 1;
       ++m) {

    re[m] =
        xhat * re[m - 1] -
        yhat * im[m - 1];

    im[m] =
        xhat * im[m - 1] +
        yhat * re[m - 1];

    rho[m] =
        ratio * rho[m - 1];
  }


  // ==========================================================================
  // Accumulate gravity terms
  // ==========================================================================

  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  double sum4 = 0.0;


  // ==========================================================================
  // Degree loop
  // ==========================================================================

  for (int n = 0;
       n <= degree_;
       ++n) {

    const int k =
        n + 1;


    // ------------------------------------------------------------------------
    // Cache triangular row offsets
    // ------------------------------------------------------------------------

    const std::size_t row_n =
        row_offset_[n];

    const std::size_t row_k =
        row_offset_[k];

    const std::size_t row_km1 =
        row_offset_[k - 1];

    const std::size_t row_km2 =
        row_offset_[k - 2];

    const std::size_t row_np1 =
        row_offset_[n + 1];


    // ------------------------------------------------------------------------
    // Diagonal recurrence
    // ------------------------------------------------------------------------

    a[row_k + k] =
        diagonal_[k] *
        a[row_km1 + k - 1];


    // ------------------------------------------------------------------------
    // Superdiagonal recurrence
    // ------------------------------------------------------------------------

    a[row_k + k - 1] =
        zhat *
        superdiagonal_[k - 1] *
        a[row_km1 + k - 1];


    // ------------------------------------------------------------------------
    // Interior Legendre recurrence
    // ------------------------------------------------------------------------

    for (int m = 0;
         m <= k - 2;
         ++m) {

      const std::size_t j =
          row_k + m;

      a[j] =
          g_[j] *
          zhat *
          a[row_km1 + m]
          -
          h_[j] *
          a[row_km2 + m];
    }


    // ------------------------------------------------------------------------
    // Harmonic accumulation
    // ------------------------------------------------------------------------

    double term1 = 0.0;
    double term2 = 0.0;
    double term3 = 0.0;
    double term4 = 0.0;

    for (int m = 0;
         m <= n;
         ++m) {

      const std::size_t j =
          row_n + m;


      // ----------------------------------------------------------------------
      // Potential contribution
      //
      // sqrt(2)*C and sqrt(2)*S have already been precomputed.
      // ----------------------------------------------------------------------

      const double d =
          c_sqrt2_[j] * re[m] +
          s_sqrt2_[j] * im[m];


      // ----------------------------------------------------------------------
      // Longitude derivative terms
      // ----------------------------------------------------------------------

      if (m > 0) {

        const double e =
            c_sqrt2_[j] * re[m - 1] +
            s_sqrt2_[j] * im[m - 1];

        const double f =
            c_sqrt2_[j] * re[m - 1] -
            s_sqrt2_[j] * im[m - 1];

        const double am =
            a[j] *
            m_double_[m];

        term1 += am * e;
        term2 += am * f;
      }


      // ----------------------------------------------------------------------
      // Latitude/radial derivative terms
      // ----------------------------------------------------------------------

      if (m < n) {

        term3 +=
            alpha_[j] *
            a[row_n + m + 1] *
            d;

        term4 +=
            beta_[j] *
            a[row_np1 + m + 1] *
            d;
      }
    }


    // ------------------------------------------------------------------------
    // Radial scaling
    // ------------------------------------------------------------------------

    const double radial =
        rho[n + 1] *
        inverse_reference_radius_;


    // ------------------------------------------------------------------------
    // Accumulate
    // ------------------------------------------------------------------------

    sum1 += radial * term1;
    sum2 += radial * term2;
    sum3 += radial * term3;
    sum4 -= radial * term4;
  }


  // ==========================================================================
  // Cartesian acceleration
  // ==========================================================================

  const std::array<double, 3> result = {

      sum1 + sum4 * xhat,
      sum2 + sum4 * yhat,
      sum3 + sum4 * zhat
  };


  // ==========================================================================
  // Output validation
  // ==========================================================================

  for (double value : result) {

    if (!std::isfinite(value))
      throw std::overflow_error(
          "Pines acceleration overflow; "
          "check radius and model constants");
  }

  return result;
}

}  // namespace ASSET
