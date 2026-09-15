#pragma once

#include "VectorFunctions/ASSET_VectorFunctions.h"

#include <stdexcept>
#include <vector>

namespace ASSET {

// Fully-normalized spherical-harmonic gravity, evaluated with the Pines
// recursion. Unlike a Python callback, this is a native ASSET function: its
// Jacobian and adjoint Hessian are obtained by ASSET forward autodiff while
// evaluating the same compiled recurrence used for values.
struct PinesGravityRHS : VectorFunction<PinesGravityRHS, 6, 6, AutodiffFwd, AutodiffFwd> {
  using Base = VectorFunction<PinesGravityRHS, 6, 6, AutodiffFwd, AutodiffFwd>;
  DENSE_FUNCTION_BASE_TYPES(Base);

  double mu;
  double reference_radius;
  int degree;
  std::vector<double> c;
  std::vector<double> s;
  std::vector<double> g;
  std::vector<double> h;
  std::vector<double> diagonal_factor;
  std::vector<double> off_diagonal_factor;
  std::vector<double> alpha;
  std::vector<double> beta;

  PinesGravityRHS(double mu_, double reference_radius_, int degree_, std::vector<double> c_, std::vector<double> s_)
      : mu(mu_), reference_radius(reference_radius_), degree(degree_), c(std::move(c_)), s(std::move(s_)) {
    if (mu <= 0.0 || reference_radius <= 0.0 || degree < 0) {
      throw std::invalid_argument("mu and reference_radius must be positive and degree non-negative");
    }
    const int side = degree + 1;
    if (static_cast<int>(c.size()) != side * side || static_cast<int>(s.size()) != side * side) {
      throw std::invalid_argument("coefficient storage does not match degree");
    }
    this->setIORows(6, 6);
    const int work_side = degree + 2;
    g.assign(work_side * work_side, 0.0);
    h.assign(work_side * work_side, 0.0);
    diagonal_factor.resize(work_side);
    off_diagonal_factor.resize(degree + 1);
    alpha.assign(side * side, 0.0);
    beta.assign(side * side, 0.0);
    for (int n = 1; n <= degree + 1; ++n) diagonal_factor[n] = std::sqrt((2.0 * n + 1.0) / (2.0 * n));
    for (int n = 1; n <= degree; ++n) off_diagonal_factor[n] = std::sqrt(2.0 * n + 3.0);
    for (int n = 1; n <= degree + 1; ++n) {
      for (int m = 0; m < n; ++m) {
        g_at(n, m) = std::sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / ((n - m) * (n + m)));
        if (n >= 2 && n - m - 1 > 0) {
          h_at(n, m) = std::sqrt((n + m - 1.0) * (2.0 * n + 1.0) * (n - m - 1.0) /
                                 ((n + m) * (n - m) * (2.0 * n - 3.0)));
        }
      }
    }
    for (int n = 0; n <= degree; ++n) {
      for (int m = 0; m <= n; ++m) {
        const double normalization = (m == 0) ? std::sqrt(2.0) : 1.0;
        alpha[n * side + m] = std::sqrt((n - m) * (n + m + 1.0)) / normalization;
        beta[n * side + m] = std::sqrt((2.0 * n + 1.0) * (n + m + 2.0) * (n + m + 1.0) /
                                           (2.0 * n + 3.0)) /
                               normalization;
      }
    }
  }

  double& g_at(int n, int m) { return g[n * (degree + 2) + m]; }
  double& h_at(int n, int m) { return h[n * (degree + 2) + m]; }
  double g_at(int n, int m) const { return g[n * (degree + 2) + m]; }
  double h_at(int n, int m) const { return h[n * (degree + 2) + m]; }
  double coefficient(const std::vector<double>& coefficients, int n, int m) const {
    return coefficients[n * (degree + 1) + m];
  }
  double harmonic_factor(const std::vector<double>& factors, int n, int m) const {
    return factors[n * (degree + 1) + m];
  }

  template<class InType, class OutType>
  inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
    using Scalar = typename InType::Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    const int work_side = degree + 2;
    const int coefficient_side = degree + 1;
    std::vector<Scalar> a(work_side * work_side, Scalar(0));
    std::vector<Scalar> rho(work_side, Scalar(0));
    std::vector<Scalar> real_part(work_side, Scalar(0));
    std::vector<Scalar> imag_part(work_side, Scalar(0));
    std::vector<Scalar> d(coefficient_side * coefficient_side, Scalar(0));
    std::vector<Scalar> e(coefficient_side * coefficient_side, Scalar(0));
    std::vector<Scalar> f(coefficient_side * coefficient_side, Scalar(0));
    auto work = [work_side](std::vector<Scalar>& values, int n, int m) -> Scalar& { return values[n * work_side + m]; };
    auto coeff = [coefficient_side](std::vector<Scalar>& values, int n, int m) -> Scalar& { return values[n * coefficient_side + m]; };

    const Scalar radius = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const Scalar xhat = x[0] / radius;
    const Scalar yhat = x[1] / radius;
    const Scalar zhat = x[2] / radius;
    const Scalar ratio = Scalar(reference_radius) / radius;
    const Scalar sqrt_two = sqrt(Scalar(2));

    work(a, 0, 0) = Scalar(1);
    if (degree >= 0) work(a, 1, 0) = zhat * sqrt(Scalar(3));
    for (int n = 1; n <= degree + 1; ++n)
      work(a, n, n) = Scalar(diagonal_factor[n]) * work(a, n - 1, n - 1);
    for (int n = 1; n <= degree; ++n)
      work(a, n + 1, n) = zhat * Scalar(off_diagonal_factor[n]) * work(a, n, n);
    for (int n = 2; n <= degree + 1; ++n)
      for (int m = 0; m < n - 1; ++m)
        work(a, n, m) = Scalar(g_at(n, m)) * zhat * work(a, n - 1, m) - Scalar(h_at(n, m)) * work(a, n - 2, m);

    real_part[0] = Scalar(1);
    if (degree >= 0) { real_part[1] = xhat; imag_part[1] = yhat; }
    for (int m = 2; m <= degree + 1; ++m) {
      real_part[m] = xhat * real_part[m - 1] - yhat * imag_part[m - 1];
      imag_part[m] = xhat * imag_part[m - 1] + yhat * real_part[m - 1];
    }
    rho[0] = Scalar(mu) / radius;
    for (int n = 1; n <= degree + 1; ++n) rho[n] = ratio * rho[n - 1];

    for (int n = 0; n <= degree; ++n) {
      for (int m = 0; m <= n; ++m) {
        coeff(d, n, m) = sqrt_two * (Scalar(coefficient(c, n, m)) * real_part[m] + Scalar(coefficient(s, n, m)) * imag_part[m]);
        if (m > 0) {
          coeff(e, n, m) = sqrt_two * (Scalar(coefficient(c, n, m)) * real_part[m - 1] + Scalar(coefficient(s, n, m)) * imag_part[m - 1]);
          coeff(f, n, m) = sqrt_two * (Scalar(coefficient(s, n, m)) * real_part[m - 1] - Scalar(coefficient(c, n, m)) * imag_part[m - 1]);
        }
      }
    }

    Scalar sum_1 = Scalar(0), sum_2 = Scalar(0), sum_3 = Scalar(0), sum_4 = Scalar(0);
    for (int n = 0; n <= degree; ++n) {
      Scalar term_1 = Scalar(0), term_2 = Scalar(0), term_3 = Scalar(0), term_4 = Scalar(0);
      for (int m = 0; m <= n; ++m) {
        const Scalar alpha_nm = Scalar(harmonic_factor(alpha, n, m));
        const Scalar beta_nm = Scalar(harmonic_factor(beta, n, m));
        term_1 += work(a, n, m) * Scalar(m) * coeff(e, n, m);
        term_2 += work(a, n, m) * Scalar(m) * coeff(f, n, m);
        term_3 += alpha_nm * work(a, n, m + 1) * coeff(d, n, m);
        term_4 += beta_nm * work(a, n + 1, m + 1) * coeff(d, n, m);
      }
      const Scalar radial = rho[n + 1] / Scalar(reference_radius);
      sum_1 += radial * term_1;
      sum_2 += radial * term_2;
      sum_3 += radial * term_3;
      sum_4 -= radial * term_4;
    }
    fx[0] = x[3]; fx[1] = x[4]; fx[2] = x[5];
    fx[3] = sum_1 + sum_4 * xhat;
    fx[4] = sum_2 + sum_4 * yhat;
    fx[5] = sum_3 + sum_4 * zhat;
  }
};

}  // namespace ASSET
