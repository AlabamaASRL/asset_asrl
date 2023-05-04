#pragma once

#include "FDCoeffs.h"
#include "pch.h"

namespace ASSET {

  // Notes: "time axis" must be monotonic increasing

  /*!
   * @brief
   *
   * @tparam DType Data Type: An Eigen::Matrix type
   * @tparam Accuracy O(h^Accuracy)
   * @tparam Order Take the "Order-th" derivative
   */
  template<class DType, int Order, int Accuracy>
  struct FinDiffDerivUniform {
   public:
    template<int Dir, int Shift>
    using Coeffs = FDCoeffs<Order, Accuracy, Dir, Shift>;
    using Scalar = typename DType::Scalar;

    int axis;
    int length;
    std::vector<Eigen::MatrixBase<DType>> data;
    Scalar h;

    static constexpr int acc = 2 * ((Accuracy + 1) / 2);
    static constexpr int ord = Order;

    static constexpr int centStenSize = ((ord / 2) == ((ord + 1) / 2)) ? (ord - 1 + acc) : (ord + acc);
    static constexpr int fbStenSize = acc + ord;

    inline void setAxisID(int i) {
      this->axis = i;
    }

    inline void setData(std::vector<Eigen::MatrixBase<DType>> d) {
      if (d.size() < this->fbStenSize) {
        std::cout << "ERROR: Not enough data for desired derivative/accuracy" << std::endl;
        return;
      } else {
        this->data = d;
        this->length = d.size();
        this->h = d[1][axis] - d[0][axis];
      }
    }

    template<class DerivType>
    inline DType derivAt(const int i, Eigen::MatrixBase<DerivType> &dout) const {
      if (i < 0 || i > length - 1) {
        std::cout << "ERROR: Index out of bounds" << std::endl;
        return;
      }

      // Calc shift
      int dir;
      int shift;
      std::vector<int> stencil;
      if (i < fbStenSize / 2) {  // Forward / semi-forward
        dir = FDCoeffType::Forwards;
        shift = i;
        for (int j = 0; j < fbStenSize; j++) {
          stencil.push_back(i - j);
        }
      } else if (length - i < fbStenSize / 2) {  // Backward / semi-backward
        dir = FDCoeffType::Backwards;
        shift = (length - i);
        for (int j = length - fbStenSize; j < length; j++) {
          stencil.push_back(i - j);
        }
      } else {  // Centered
        dir = FDCoeffType::Central;
        shift = 0;
        int lb = centStenSize / 2;
        for (int j = 0; j < centStenSize; j++) {
          stencil.push_back(i + j - lb);
        }
      }
      int sz = stencil.size();

      // Get weights
      // using CF = Coeffs<dir, shift>;

      // Calc deriv
      dout.setZero();
      for (int j = 0; j < sz; j++) {
        // dout += CF::weights[j] * data[i + stencil[j]];
      }
    }
  };

}  // namespace ASSET
