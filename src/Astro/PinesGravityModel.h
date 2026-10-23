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
//     pp. 1508–1511.
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
 * The model evaluates a gravitational field from fully normalized
 * spherical harmonic coefficients C_nm and S_nm up to a configured
 * maximum degree.
 *
 * Coefficients are loaded from an external gravity-field file during
 * construction. Recurrence factors are precomputed and stored for reuse
 * during subsequent acceleration evaluations.
 *
 * Units must be consistent across all inputs:
 *   - Position: distance units consistent with reference_radius.
 *   - mu: gravitational parameter, distance^3 / time^2.
 *   - reference_radius: distance.
 *   - Acceleration: distance / time^2.
 *
 * The coefficient file must use the fully normalized convention
 * expected by this implementation, including its C00 convention.
 *
 */
//Delcare for build
void BuildPinesGravity(FunctionRegistry& reg, py::module& m);

class PinesGravityModel {
 public:
  /**
   * @brief Loads gravity coefficients and precomputes recurrence factors.
   *
   * @param coefficient_file Path to the spherical harmonic coefficient file.
   * @param degree Maximum spherical harmonic degree included in evaluation.
   * @param mu Gravitational parameter in consistent distance^3/time^2 units.
   * @param reference_radius Reference radius used to scale harmonic terms.
   * @param normalization Coefficient normalization convention. This
   *        implementation requires "fully_normalized".
   *
   * The coefficient arrays and recurrence factors are initialized once
   * so that repeated field evaluations can reuse the prepared model.
   *
   * @throws std::invalid_argument If the inputs or coefficient file are
   *         invalid or incompatible with the requested degree.
   */
  PinesGravityModel(
      const std::string& coefficient_file,
      int degree,
      double mu,
      double reference_radius,
      const std::string& normalization);

  /**
   * @brief Computes gravitational acceleration for a Cartesian position.
   *
   * @param position Cartesian position [x, y, z], expressed in the same
   *        distance units as reference_radius.
   * @return Cartesian acceleration [ax, ay, az].
   *
   * This non-templated interface accepts double-precision coordinates
   * and provides the checked public evaluation path.
   */
  std::array<double, 3> acceleration(
      const std::array<double, 3>& position) const;

  /**
   * @brief Evaluates gravitational acceleration for a supported scalar type.
   *
   * The calculation is templated so that the recurrence can operate on
   * double precision or compatible automatic-differentiation scalar types.
   *
   * Scratch storage is reused per thread and scalar type to reduce
   * repeated memory allocations. Gravity coefficients and recurrence
   * factors remain owned by this model.
   *
   * @param position Cartesian position [x, y, z].
   * @return Cartesian acceleration [ax, ay, az].
   *
   * Precondition: position must be finite and have a nonzero radius.
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

    // Normalize the Cartesian position to obtain direction cosines.
    // The dimensionless ratio R/r controls radial attenuation with degree.
    const Scalar xhat = x / radius;
    const Scalar yhat = y / radius;
    const Scalar zhat = z / radius;

    const Scalar ratio = Scalar(reference_radius_) / radius;

    // Triangular storage for the normalized Legendre recurrence:
    //   index(n,m) = n*(n+1)/2 + m, for 0 <= m <= n.
    //
    // Each degree occupies a contiguous row. The recurrence depends on
    // preceding degree rows, so rows are generated in ascending order.
    //
    // Reuse per-thread scratch storage to avoid repeated heap allocations.
    // Each thread has its own buffer.
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

    // Radial scaling for the potential-gradient terms:
    //   rho[n] = (mu/r) * (R/r)^n.
    //
    // The recurrence avoids evaluating a separate power for each degree.
    rho[0] = Scalar(mu_) / radius;

    // Real and imaginary parts of (xhat + i*yhat)^m.
    // These powers encode the longitude/order dependence without
    // evaluating trigonometric functions inside the harmonic loops.
    // One of the true benefits of the Pines implementation.
    re[0] = Scalar(1.0);
    im[0] = Scalar(0.0);

    for (int m = 1; m <= degree_ + 1; ++m) {
      const std::size_t j = static_cast<std::size_t>(m);
      const std::size_t jm1 = j - 1;

      const Scalar previous_re = re[jm1];
      const Scalar previous_im = im[jm1];

      re[j] = xhat * previous_re - yhat * previous_im;
      im[j] = xhat * previous_im + yhat * previous_re;

      rho[j] = ratio * rho[jm1];
    }

    // Accumulate separate contributions used to assemble the final
    // Cartesian acceleration. Preserve the recurrence factors and signs.
    Scalar sum1 = Scalar(0.0);
    Scalar sum2 = Scalar(0.0);
    Scalar sum3 = Scalar(0.0);
    Scalar sum4 = Scalar(0.0);

    for (int n = 0; n <= degree_; ++n) {
      const int k = n + 1;

      // Construct the normalized Legendre recurrence row required for
      // degree n's acceleration terms.
      const std::size_t kk = index(k, k);
      const std::size_t prev_diag = index(k - 1, k - 1);

      // Diagonal term.
      a[kk] = Scalar(diagonal_[k]) * a[prev_diag];

      // Adjacent-order term.
      a[index(k, k - 1)] =
          zhat * Scalar(superdiagonal_[k - 1]) * a[prev_diag];

      // Remaining orders depend on the preceding two degree rows.
      for (int m = 0; m <= k - 2; ++m) {
        const std::size_t current = index(k, m);
        const std::size_t previous = index(k - 1, m);
        const std::size_t previous2 = index(k - 2, m);

        a[current] =
            Scalar(g_[current]) * zhat * a[previous] -
            Scalar(h_[current]) * a[previous2];
      }

      Scalar term1 = Scalar(0.0);
      Scalar term2 = Scalar(0.0);
      Scalar term3 = Scalar(0.0);
      Scalar term4 = Scalar(0.0);

      const std::size_t row_start = index(n, 0);

      // Cache coefficient row bases. The coefficient arrays share the
      // same triangular indexing convention.
      const double* const c_row = c_.data() + row_start;
      const double* const s_row = s_.data() + row_start;
      const double* const alpha_row = alpha_.data() + row_start;
      const double* const beta_row = beta_.data() + row_start;

      for (int m = 0; m <= n; ++m) {
        const std::size_t j =
            row_start + static_cast<std::size_t>(m);

        const Scalar cm = Scalar(c_[j]);
        const Scalar sm = Scalar(s_[j]);

        // Harmonic coefficient contribution for order m.
        const Scalar d =
            Scalar(sqrt_two) * (cm * re[m] + sm * im[m]);

        if (m > 0) {
          // Order-dependent contributions for the first two sums.
          const Scalar e =
              Scalar(sqrt_two) *
              (cm * re[m - 1] + sm * im[m - 1]);

          const Scalar f =
              Scalar(sqrt_two) *
              (sm * re[m - 1] - cm * im[m - 1]);

          const Scalar weighted =
              a[j] * Scalar(m);

          term1 += weighted * e;
          term2 += weighted * f;
        }

        if (m < n) {
          // Contribution coupling the current order to order m + 1.
          term3 +=
              Scalar(alpha_[j]) *
              a[index(n, m + 1)] * d;
        }

        // Contribution coupling to the next degree row.
        term4 +=
            Scalar(beta_[j]) *
            a[index(n + 1, m + 1)] * d;
      }

      // Apply radial scaling to the accumulated degree contributions.
      const Scalar radial =
          rho[n + 1] / Scalar(reference_radius_);

      sum1 += radial * term1;
      sum2 += radial * term2;
      sum3 += radial * term3;
      sum4 -= radial * term4;
    }

    // Assemble Cartesian acceleration from the accumulated terms.
    //
    // The degree-zero contribution is intended to recover central
    // gravity when C00 = 1 and all other coefficients are zero.
    // Do not add a separate central-gravity term here.
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

  /// @return Reference radius used to scale harmonic terms.
  double reference_radius() const {
    return reference_radius_;
  }

  /// @return Path to the coefficient file used to construct the model.
  const std::string& coefficient_file() const {
    return coefficient_file_;
  }

 private:
  /**
   * @brief Maps a degree/order pair to triangular-array storage.
   *
   * @param n Spherical harmonic degree.
   * @param m Spherical harmonic order, where 0 <= m <= n.
   * @return Offset of coefficient (n,m) in a triangular array.
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

  // Fully normalized spherical harmonic coefficients in triangular
  // degree/order layout.
  std::vector<double> c_;
  std::vector<double> s_;

  // Precomputed factors for the normalized Legendre recurrence.
  std::vector<double> g_;
  std::vector<double> h_;

  // Precomputed factors used in the acceleration recurrence.
  std::vector<double> alpha_;
  std::vector<double> beta_;

  // Degree-dependent factors for diagonal and adjacent-order terms.
  std::vector<double> diagonal_;
  std::vector<double> superdiagonal_;
};

}  // namespace ASSET
