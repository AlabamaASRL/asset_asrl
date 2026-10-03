#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ASSET {

// Double-precision, fully normalized Pines gravity. Position/radius in km,
// mu in km^3/s^2; the result includes degree zero and is in km/s^2.
// Initialized data is never modified during evaluation.
class PinesGravityModel {
 public:
  PinesGravityModel(const std::string& coefficient_file, int degree,
                    double mu, double reference_radius,
                    const std::string& normalization);
  std::array<double, 3> acceleration(const std::array<double, 3>& position) const;

  // Scalar-templated recurrence used by ASSET automatic differentiation.
  // Coefficients are immutable doubles; every operation depending on position
  // remains in Scalar so values, Jacobians, and Hessians follow the same Pines
  // equations rather than a reduced gravity approximation.
  template<class Scalar>
  std::array<Scalar, 3> acceleration_scalar(
      const std::array<Scalar, 3>& position) const {
    using std::sqrt;
    constexpr double sqrt_two = 1.4142135623730950488;
    const Scalar radius = sqrt(position[0]*position[0] +
                               position[1]*position[1] +
                               position[2]*position[2]);
    const Scalar xhat = position[0] / radius;
    const Scalar yhat = position[1] / radius;
    const Scalar zhat = position[2] / radius;
    const Scalar ratio = Scalar(reference_radius_) / radius;

    // Each Scalar specialization has independent per-thread storage. All
    // entries read by the recurrence are overwritten during every call.
    thread_local std::vector<Scalar> scratch;
    const auto a_count = index(degree_ + 2, 0);
    const auto width = std::size_t(degree_) + 2;
    scratch.resize(a_count + 3 * width);
    Scalar* a = scratch.data();
    Scalar* rho = a + a_count;
    Scalar* re = rho + width;
    Scalar* im = re + width;
    a[0] = Scalar(1.0);
    rho[0] = Scalar(mu_) / radius;
    re[0] = Scalar(1.0);
    im[0] = Scalar(0.0);
    for (int m = 1; m <= degree_ + 1; ++m) {
      re[m] = xhat * re[m-1] - yhat * im[m-1];
      im[m] = xhat * im[m-1] + yhat * re[m-1];
      rho[m] = ratio * rho[m-1];
    }

    Scalar sum1 = Scalar(0.0), sum2 = Scalar(0.0);
    Scalar sum3 = Scalar(0.0), sum4 = Scalar(0.0);
    for (int n = 0; n <= degree_; ++n) {
      const int k = n + 1;
      a[index(k, k)] = Scalar(diagonal_[k]) * a[index(k-1, k-1)];
      a[index(k, k-1)] = zhat * Scalar(superdiagonal_[k-1]) *
                          a[index(k-1, k-1)];
      for (int m = 0; m <= k - 2; ++m)
        a[index(k, m)] = Scalar(g_[index(k, m)]) * zhat *
                           a[index(k-1, m)] -
                         Scalar(h_[index(k, m)]) * a[index(k-2, m)];

      Scalar term1 = Scalar(0.0), term2 = Scalar(0.0);
      Scalar term3 = Scalar(0.0), term4 = Scalar(0.0);
      for (int m = 0; m <= n; ++m) {
        const auto j = index(n, m);
        const Scalar d = Scalar(sqrt_two) *
                         (Scalar(c_[j]) * re[m] + Scalar(s_[j]) * im[m]);
        if (m > 0) {
          const Scalar e = Scalar(sqrt_two) *
                           (Scalar(c_[j]) * re[m-1] +
                            Scalar(s_[j]) * im[m-1]);
          const Scalar f = Scalar(sqrt_two) *
                           (Scalar(s_[j]) * re[m-1] -
                            Scalar(c_[j]) * im[m-1]);
          term1 += a[j] * Scalar(m) * e;
          term2 += a[j] * Scalar(m) * f;
        }
        if (m < n)
          term3 += Scalar(alpha_[j]) * a[index(n, m+1)] * d;
        term4 += Scalar(beta_[j]) * a[index(n+1, m+1)] * d;
      }
      const Scalar radial = rho[n+1] / Scalar(reference_radius_);
      sum1 += radial * term1;
      sum2 += radial * term2;
      sum3 += radial * term3;
      sum4 -= radial * term4;
    }
    return {sum1 + sum4*xhat, sum2 + sum4*yhat, sum3 + sum4*zhat};
  }

  int degree() const { return degree_; }
  double mu() const { return mu_; }
  double reference_radius() const { return reference_radius_; }
  const std::string& coefficient_file() const { return coefficient_file_; }

 private:
  static std::size_t index(int n, int m) {
    return std::size_t(n) * (std::size_t(n) + 1) / 2 + std::size_t(m);
  }
  int degree_;
  double mu_, reference_radius_;
  std::string coefficient_file_;
  std::vector<double> c_, s_, g_, h_, alpha_, beta_, diagonal_, superdiagonal_;
};
}  // namespace ASSET
