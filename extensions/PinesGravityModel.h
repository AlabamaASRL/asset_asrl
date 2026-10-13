```cpp
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ASSET {

class PinesGravityModel {
 public:
  PinesGravityModel(
      const std::string& coefficient_file,
      int degree,
      double mu,
      double reference_radius,
      const std::string& normalization);

  std::array<double, 3> acceleration(
      const std::array<double, 3>& position) const;

  template <class Scalar>
  std::array<Scalar, 3> acceleration_scalar(
      const std::array<Scalar, 3>& position) const {

    using std::sqrt;

    constexpr double sqrt_two = 1.4142135623730950488;

    const Scalar x = position[0];
    const Scalar y = position[1];
    const Scalar z = position[2];

    const Scalar radius = sqrt(x * x + y * y + z * z);

    const Scalar xhat = x / radius;
    const Scalar yhat = y / radius;
    const Scalar zhat = z / radius;

    const Scalar ratio = Scalar(reference_radius_) / radius;

    // Triangular recurrence storage:
    // index(n,m) = n*(n+1)/2 + m.
    //
    // The thread-local buffer avoids a fresh heap allocation on every
    // evaluation when the same thread evaluates the same or lower degree.
    // Each thread has independent scratch storage.
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

    // rho[n] = mu/r * (R/r)^n.
    rho[0] = Scalar(mu_) / radius;

    // Real and imaginary parts of (xhat + i*yhat)^m.
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

    Scalar sum1 = Scalar(0.0);
    Scalar sum2 = Scalar(0.0);
    Scalar sum3 = Scalar(0.0);
    Scalar sum4 = Scalar(0.0);

    for (int n = 0; n <= degree_; ++n) {
      const int k = n + 1;

      // Build the normalized Legendre recurrence row needed by
      // this degree's acceleration terms.
      const std::size_t kk = index(k, k);
      const std::size_t prev_diag = index(k - 1, k - 1);

      a[kk] = Scalar(diagonal_[k]) * a[prev_diag];

      a[index(k, k - 1)] =
          zhat * Scalar(superdiagonal_[k - 1]) * a[prev_diag];

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

      for (int m = 0; m <= n; ++m) {
        const std::size_t j =
            row_start + static_cast<std::size_t>(m);

        const Scalar cm = Scalar(c_[j]);
        const Scalar sm = Scalar(s_[j]);

        const Scalar d =
            Scalar(sqrt_two) * (cm * re[m] + sm * im[m]);

        if (m > 0) {
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
          term3 +=
              Scalar(alpha_[j]) *
              a[index(n, m + 1)] * d;
        }

        term4 +=
            Scalar(beta_[j]) *
            a[index(n + 1, m + 1)] * d;
      }

      const Scalar radial =
          rho[n + 1] / Scalar(reference_radius_);

      sum1 += radial * term1;
      sum2 += radial * term2;
      sum3 += radial * term3;
      sum4 -= radial * term4;
    }

    // The degree-zero term must produce central gravity:
    // a = -mu*r_vec/|r_vec|^3 when C00=1 and all other coefficients
    // are zero. Do not add another central term here: it is intended
    // to be supplied by the n=0 recurrence contribution.
    return {
        sum1 + sum4 * xhat,
        sum2 + sum4 * yhat,
        sum3 + sum4 * zhat
    };
  }

  int degree() const {
    return degree_;
  }

  double mu() const {
    return mu_;
  }

  double reference_radius() const {
    return reference_radius_;
  }

  const std::string& coefficient_file() const {
    return coefficient_file_;
  }

 private:
  static std::size_t index(int n, int m) {
    return static_cast<std::size_t>(n) *
               (static_cast<std::size_t>(n) + 1) / 2 +
           static_cast<std::size_t>(m);
  }

  int degree_;
  double mu_;
  double reference_radius_;
  std::string coefficient_file_;

  std::vector<double> c_;
  std::vector<double> s_;

  std::vector<double> g_;
  std::vector<double> h_;

  std::vector<double> alpha_;
  std::vector<double> beta_;

  std::vector<double> diagonal_;
  std::vector<double> superdiagonal_;
};

}  // namespace ASSET
```
