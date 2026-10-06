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

 private:
  static std::size_t index(int n, int m) {
    return static_cast<std::size_t>(n) *
               static_cast<std::size_t>(n + 1) /
               2 +
           static_cast<std::size_t>(m);
  }

  int degree_;
  double mu_;
  double reference_radius_;
  double inverse_reference_radius_;
  std::string coefficient_file_;

  // Gravity coefficients.
  std::vector<double> c_;
  std::vector<double> s_;

  // Legendre recurrence coefficients.
  std::vector<double> g_;
  std::vector<double> h_;
  std::vector<double> alpha_;
  std::vector<double> beta_;
  std::vector<double> diagonal_;
  std::vector<double> superdiagonal_;

  // --------------------------------------------------------------------------
  // Optimized evaluation data
  // --------------------------------------------------------------------------

  // Starting index of each triangular-array row.
  //
  // row_offset_[n] = n(n+1)/2
  //
  // This avoids repeatedly evaluating index(n,m) inside the hot loops.
  std::vector<std::size_t> row_offset_;

  // sqrt(2) multiplied into the coefficients once at construction.
  std::vector<double> c_sqrt2_;
  std::vector<double> s_sqrt2_;

  // m converted to double once.
  std::vector<double> m_double_;

  // Reusable workspace for the double evaluation path.
  mutable std::vector<double> scratch_double_;
};

}  // namespace ASSET
