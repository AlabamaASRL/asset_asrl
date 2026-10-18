
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ASSET {

// References:
// [1] S. Pines, "Uniform Representation of the Gravitational Potential
//     and its Derivatives," AIAA Journal, Vol. 11, No. 11, 1973,
//     pp. 1508-1511.
//     DOI: https://doi.org/10.2514/3.50619
//
// The implementation uses a Pines-style recurrence to evaluate
// spherical-harmonic gravitational acceleration. The recurrence and
// coefficient normalization must be consistent with the supplied
// gravity coefficient file.

/**
 * @brief Evaluates gravitational acceleration using the Pines spherical
 *        harmonic formulation.
 *
 * Coefficients are loaded from an external gravity-field file during
 * construction. Recurrence factors are precomputed and reused during
 * subsequent acceleration evaluations.
 *
 * Units must be consistent across all inputs:
 *   - Position: distance units consistent with reference_radius.
 *   - mu: gravitational parameter, distance^3 / time^2.
 *   - reference_radius: distance.
 *   - Acceleration: distance / time^2.
 *
 * The coefficient file must use the fully normalized convention
 * expected by this implementation, including its C00 convention.
 */
class PinesGravityModel {
 public:
  /**
   * @brief Loads gravity coefficients and precomputes recurrence factors.
   *
   * @param coefficient_file Path to the spherical harmonic coefficient file.
   * @param degree Maximum spherical harmonic degree included in evaluation.
   * @param mu Gravitational parameter in consistent distance^3/time^2 units.
   * @param reference_radius Reference radius used to scale harmonic terms.
   * @param normalization Must be "fully_normalized".
   */
  PinesGravityModel(
      const std::string& coefficient_file,
      int degree,
      double mu,
      double reference_radius,
      const std::string& normalization);

  /**
   * @brief Computes checked double-precision gravitational acceleration.
   *
   * @param position Cartesian position [x, y, z].
   * @return Cartesian acceleration [ax, ay, az].
   */
  std::array<double, 3> acceleration(
      const std::array<double, 3>& position) const;

  /**
   * @brief Computes gravitational acceleration for a supported scalar type.
   *
   * Uses triangular recurrence storage and per-thread scratch memory.
   * The scratch vector is reused after its first allocation for a given
   * scalar type and sufficient degree.
   *
   * Precondition: position must be finite and have nonzero radius.
   * This templated method does not independently validate that precondition.
   */
  template <class Scalar>
  std::array<Scalar, 3> acceleration_scalar(
      const std::array<Scalar, 3>& position) const {

    using std::sqrt;

    constexpr double sqrt_two = 1.4142135623730950488;

    const Scalar x = position[0];
    const Scalar y = position[1];
    const Scalar z = position[2];

    const Scalar radius = sqrt(x * x + y * y + z * z);

    // Unit direction vector.
    const Scalar xhat = x / radius;
    const Scalar yhat = y / radius;
    const Scalar zhat = z / radius;

    // Radial attenuation ratio and constant inverse reference radius.
    const Scalar ratio = Scalar(reference_radius_) / radius;
    const Scalar inv_reference_radius =
        Scalar(1.0 / reference_radius_);

    // Scratch layout:
    //   a   : triangular Legendre recurrence
    //   rho : radial scaling by degree
    //   re  : real parts of longitude powers
    //   im  : imaginary parts of longitude powers
    //
    // Thread-local storage prevents cross-thread scratch-buffer sharing.
    thread_local std::vector<Scalar> scratch;

    const std::size_t a_count = index(degree_ + 2, 0);
    const std::size_t width =
        static_cast<std::size_t>(degree_) + 2;
    const std::size_t required = a_count + 3 * width;

    if (scratch.size() < required) {
      scratch.resize(required);
    }

    Scalar* const a = scratch.data();
    Scalar* const rho = a + a_count;
    Scalar* const re = rho + width;
    Scalar* const im = re + width;

    a[0] = Scalar(1.0);

    // rho[n] = (mu/r) * (R/r)^n.
    rho[0] = Scalar(mu_) / radius;

    // Powers of (xhat + i*yhat)^m.
    re[0] = Scalar(1.0);
    im[0] = Scalar(0.0);

    for (int m = 1; m <= degree_; ++m) {
      const std::size_t j = static_cast<std::size_t>(m);

      const Scalar previous_re = re[j - 1];
      const Scalar previous_im = im[j - 1];

      re[j] = xhat * previous_re - yhat * previous_im;
      im[j] = xhat * previous_im + yhat * previous_re;
    }

    // The radial recurrence must reach degree + 1.
    for (int n = 1; n <= degree_ + 1; ++n) {
      const std::size_t j = static_cast<std::size_t>(n);
      rho[j] = ratio * rho[j - 1];
    }

    // Accumulate Cartesian acceleration contributions.
    Scalar sum1 = Scalar(0.0);
    Scalar sum2 = Scalar(0.0);
    Scalar sum3 = Scalar(0.0);
    Scalar sum4 = Scalar(0.0);

    for (int n = 0; n <= degree_; ++n) {
      const int k = n + 1;

      // Recurrence row starts.
      const std::size_t current_start = index(k, 0);
      const std::size_t previous_start = index(k - 1, 0);

      Scalar* const current_row = a + current_start;
      const Scalar* const previous_row = a + previous_start;

      // Diagonal recurrence.
      current_row[k] =
          Scalar(diagonal_[k]) * previous_row[k - 1];

      // Adjacent-order recurrence.
      current_row[k - 1] =
          zhat * Scalar(superdiagonal_[k - 1]) *
          previous_row[k - 1];

      // Two-row recurrence for the remaining orders.
      if (k >= 2) {
        const Scalar* const previous2_row =
            a + index(k - 2, 0);

        const double* const g_row = g_.data() + current_start;
        const double* const h_row = h_.data() + current_start;

        for (int m = 0; m <= k - 2; ++m) {
          current_row[m] =
              Scalar(g_row[m]) * zhat * previous_row[m] -
              Scalar(h_row[m]) * previous2_row[m];
        }
      }

      // Contiguous coefficient and recurrence-factor rows.
      const std::size_t row_start = index(n, 0);
      const std::size_t next_row_start = index(n + 1, 0);

      const double* const c_row = c_.data() + row_start;
      const double* const s_row = s_.data() + row_start;
      const double* const alpha_row = alpha_.data() + row_start;
      const double* const beta_row = beta_.data() + row_start;

      const Scalar* const a_row = a + row_start;
      const Scalar* const a_next_row = a + next_row_start;

      Scalar term1 = Scalar(0.0);
      Scalar term2 = Scalar(0.0);
      Scalar term3 = Scalar(0.0);
      Scalar term4 = Scalar(0.0);

      for (int m = 0; m <= n; ++m) {
        const std::size_t j = static_cast<std::size_t>(m);

        const Scalar cm = Scalar(c_row[m]);
        const Scalar sm = Scalar(s_row[m]);

        // Harmonic coefficient contribution.
        const Scalar d =
            Scalar(sqrt_two) *
            (cm * re[m] + sm * im[m]);

        if (m > 0) {
          const Scalar previous_re = re[m - 1];
          const Scalar previous_im = im[m - 1];

          const Scalar e =
              Scalar(sqrt_two) *
              (cm * previous_re + sm * previous_im);

          const Scalar f =
              Scalar(sqrt_two) *
              (sm * previous_re - cm * previous_im);

          const Scalar weighted = a_row[m] * Scalar(m);

          term1 += weighted * e;
          term2 += weighted * f;
        }

        // Coupling to order m + 1 within the current degree.
        if (m < n) {
          term3 +=
              Scalar(alpha_row[m]) * a_row[m + 1] * d;
        }

        // Coupling to order m + 1 in the next degree row.
        term4 +=
            Scalar(beta_row[m]) * a_next_row[m + 1] * d;
      }

      // Radial scaling. Precompute the inverse reference radius above
      // to avoid a division for every degree.
      const Scalar radial =
          rho[n + 1] * inv_reference_radius;

      sum1 += radial * term1;
      sum2 += radial * term2;
      sum3 += radial * term3;
      sum4 -= radial * term4;
    }

    // Assemble Cartesian acceleration.
    return {
        sum1 + sum4 * xhat,
        sum2 + sum4 * yhat,
        sum3 + sum4 * zhat
    };
  }

  /// @return Maximum spherical harmonic degree configured for this model.
  int degree() const {
    return degree_;
  }

  /// @return Gravitational parameter used by the model.
  double mu() const {
    return mu_;
  }

  /// @return Reference radius used by the model.
  double reference_radius() const {
    return reference_radius_;
  }

  /// @return Path to the coefficient file used by the model.
  const std::string& coefficient_file() const {
    return coefficient_file_;
  }

 private:
  /**
   * @brief Maps (n,m) to triangular-array storage.
   *
   * index(n,m) = n*(n+1)/2 + m.
   */
  static std::size_t index(int n, int m) {
    return static_cast<std::size_t>(n) *
               (static_cast<std::size_t>(n) + 1) / 2 +
           static_cast<std::size_t>(m);
  }

  int degree_;
  double mu_;
  double reference_radius_;
  std::string coefficient_file_;

  // Fully normalized coefficients in triangular degree/order layout.
  std::vector<double> c_;
  std::vector<double> s_;

  // Precomputed normalized Legendre recurrence factors.
  std::vector<double> g_;
  std::vector<double> h_;

  // Precomputed acceleration recurrence factors.
  std::vector<double> alpha_;
  std::vector<double> beta_;

  // Degree-dependent diagonal and adjacent-order factors.
  std::vector<double> diagonal_;
  std::vector<double> superdiagonal_;
};

}  // namespace ASSET
