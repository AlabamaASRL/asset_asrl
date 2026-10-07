#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace ASSET {

class PinesGravityModel {
 public:
  PinesGravityModel(const std::string& coefficient_file,
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

    const Scalar radius =
        sqrt(position[0] * position[0] +
             position[1] * position[1] +
             position[2] * position[2]);

    const Scalar inverse_radius = Scalar(1.0) / radius;

    const Scalar xhat = position[0] * inverse_radius;
    const Scalar yhat = position[1] * inverse_radius;
    const Scalar zhat = position[2] * inverse_radius;

    const Scalar ratio =
        Scalar(reference_radius_) * inverse_radius;

    const std::size_t a_count =
        row_offset_[degree_ + 2];

    const std::size_t width =
        static_cast<std::size_t>(degree_) + 2;

    thread_local std::vector<Scalar> scratch;

    const std::size_t required =
        a_count + 2 * width;

    if (scratch.size() < required)
      scratch.resize(required);

    Scalar* const a = scratch.data();
    Scalar* const re = a + a_count;
    Scalar* const im = re + width;

    a[0] = Scalar(1.0);

    re[0] = Scalar(1.0);
    im[0] = Scalar(0.0);

    for (int m = 1; m <= degree_ + 1; ++m) {
      const Scalar prev_re = re[m - 1];
      const Scalar prev_im = im[m - 1];

      re[m] = xhat * prev_re - yhat * prev_im;
      im[m] = xhat * prev_im + yhat * prev_re;
    }

    Scalar radial =
        Scalar(mu_) *
        inverse_radius *
        inverse_radius;

    Scalar sum1 = Scalar(0.0);
    Scalar sum2 = Scalar(0.0);
    Scalar sum3 = Scalar(0.0);
    Scalar sum4 = Scalar(0.0);

    for (int n = 0; n <= degree_; ++n) {
      const int k = n + 1;

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

      a[row_k + k] =
          Scalar(diagonal_[k]) *
          a[row_km1 + k - 1];

      a[row_k + k - 1] =
          zhat *
          Scalar(superdiagonal_[k - 1]) *
          a[row_km1 + k - 1];

      for (int m = 0; m <= k - 2; ++m) {
        const std::size_t j =
            row_k + m;

        a[j] =
            Scalar(g_[j]) *
            zhat *
            a[row_km1 + m]
            -
            Scalar(h_[j]) *
            a[row_km2 + m];
      }

      Scalar term1 = Scalar(0.0);
      Scalar term2 = Scalar(0.0);
      Scalar term3 = Scalar(0.0);
      Scalar term4 = Scalar(0.0);

      for (int m = 0; m <= n; ++m) {
        const std::size_t j =
            row_n + m;

        const Scalar d =
            Scalar(sqrt_two) *
            (Scalar(c_[j]) * re[m] +
             Scalar(s_[j]) * im[m]);

        if (m > 0) {
          const Scalar e =
              Scalar(sqrt_two) *
              (Scalar(c_[j]) * re[m - 1] +
               Scalar(s_[j]) * im[m - 1]);

          const Scalar f =
              Scalar(sqrt_two) *
              (Scalar(c_[j]) * re[m - 1] -
               Scalar(s_[j]) * im[m - 1]);

          const Scalar am =
              a[j] * Scalar(m);

          term1 += am * e;
          term2 += am * f;
        }

        if (m < n) {
          term3 +=
              Scalar(alpha_[j]) *
              a[row_n + m + 1] *
              d;
        }

        term4 +=
            Scalar(beta_[j]) *
            a[row_np1 + m + 1] *
            d;
      }

      sum1 += radial * term1;
      sum2 += radial * term2;
      sum3 += radial * term3;
      sum4 -= radial * term4;

      radial *= ratio;
    }

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
  static constexpr std::size_t index(int n, int m) {
    return static_cast<std::size_t>(n) *
               static_cast<std::size_t>(n + 1) / 2 +
           static_cast<std::size_t>(m);
  }

  int degree_;
  double mu_;
  double reference_radius_;
  double inverse_reference_radius_;

  std::string coefficient_file_;

  std::vector<double> c_;
  std::vector<double> s_;

  std::vector<double> g_;
  std::vector<double> h_;

  std::vector<double> alpha_;
  std::vector<double> beta_;

  std::vector<double> diagonal_;
  std::vector<double> superdiagonal_;

  std::vector<std::size_t> row_offset_;

  std::vector<double> c_sqrt2_;
  std::vector<double> s_sqrt2_;

  std::vector<double> m_double_;

  mutable std::vector<double> scratch_double_;
};

}  // namespace ASSET
