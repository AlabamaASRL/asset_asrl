#pragma once

#include <Eigen/Cholesky>

#include "pch.h"

namespace ASSET {

  template<class VectorType>
  std::vector<VectorType> FDiffData(const std::vector<VectorType>& BL, int axis, bool inctime) {
    std::vector<VectorType> BLDot = BL;
    double h = BL[1][axis] - BL[0][axis];
    for (int i = 0; i < BL.size(); i++) {
      if (!inctime) {
        BLDot[i].resize(axis);
      }
      if (i < 2) {
        BLDot[i].head(axis) = ((1.0 / 3.0) * BL[i + 3].head(axis) - 1.5 * BL[i + 2].head(axis)
                               + 3.0 * BL[i + 1].head(axis) - (11.0 / 6) * BL[i].head(axis))
                              * (1.0 / h);
      } else if (i > (BL.size() - 3)) {
        BLDot[i].head(axis) = (BL[i].head(6) - BL[i - 1].head(axis)) * (1.0 / h);
      } else {
        BLDot[i].head(axis) = ((-1.0 / 12) * BL[i + 2].head(axis) + (2.0 / 3.0) * BL[i + 1].head(axis)
                               + (-2.0 / 3.0) * BL[i - 1].head(axis) + (1.0 / 12.0) * BL[i - 2].head(axis))
                              * (1.0 / h);
      }
    }
    return BLDot;
  }

  /*!
   * @brief
   *
   * @tparam DType Data Type: An Eigen::Matrix type
   */
  template<class DType>
  struct FDDerivArbitrary {
   public:
    using Scalar = typename DType::Scalar;

    int axis;
    int length;
    std::vector<DType> data;

    inline void setAxis(int i) {
      this->axis = i;
    }

    inline void setData(const std::vector<DType>& d) {
      this->data = d;
      this->length = d.size();
    }

    FDDerivArbitrary() {};
    FDDerivArbitrary(int i, const std::vector<DType>& d) {
      this->setAxis(i);
      this->setData(d);
    }

    /*!
     * @brief
     *
     * @tparam Order
     * @tparam Accuracy
     * @tparam DerivType
     * @param i
     * @param dout
     */
    template<class DerivType>
    inline void derivAt(const int i, DerivType& dout, const int Order, const int Accuracy) const {
      const int acc = 2 * ((Accuracy + 1) / 2);
      const int ord = Order;

      const int centStenSize = ((ord / 2) == ((ord + 1) / 2)) ? (ord - 1 + acc) : (ord + acc);
      const int fbStenSize = acc + ord;

      if (length < fbStenSize) {
        std::cout << "ERROR: Requested accuracy too high for given data" << std::endl;
      }

      if (i < 0 || i > length - 1) {
        std::cout << "ERROR: Index out of bounds" << std::endl;
        return;
      }

      Scalar t0 = data[i][axis];

      // Calc steps and stencil
      Eigen::Matrix<Scalar, -1, 1> steps;
      std::vector<int> stencil;
      if (i < fbStenSize / 2) {  // Forward / semi-forward
        steps.resize(fbStenSize);
        for (int j = 0; j < fbStenSize; j++) {
          steps[j] = data[j][axis] - t0;
          stencil.push_back(j - i);
        }
      } else if (length - 1 - i < fbStenSize / 2) {  // Backward / semi-backward
        steps.resize(fbStenSize);
        for (int j = length - fbStenSize; j < length; j++) {
          int k = j - length + fbStenSize;
          steps[k] = data[j][axis] - t0;
          stencil.push_back(j - i);
        }
      } else {  // Centered
        steps.resize(centStenSize);
        int lb = centStenSize / 2;
        for (int j = 0; j < centStenSize; j++) {
          steps[j] = data[i - lb + j][axis] - t0;
          stencil.push_back(j - lb);
        }
      }

      // Calc weights
      int sz = steps.size();

      Eigen::Matrix<Scalar, -1, -1> mat(sz, sz);
      for (int j = 0; j < sz; j++) {
        mat.row(j) = steps.array().pow(j);
      }

      Eigen::Matrix<Scalar, -1, 1> vec(sz);
      vec.setZero();
      vec[ord] = factorial(ord);

      Eigen::Matrix<Scalar, -1, 1> weights = mat.colPivHouseholderQr().solve(vec);

      // Calc deriv
      dout.setZero();
      for (int j = 0; j < sz; j++) {
        dout += weights[j] * data[i + stencil[j]];
      }
    }

    /*!
     * @brief
     *
     * @tparam Order
     * @tparam Accuracy
     * @tparam DerivType
     * @param i
     * @return DerivType
     */
    template<class DerivType>
    inline DerivType derivAt(const int i, const int Order, const int Accuracy) const {
      DerivType dout(this->data[0].size());
      this->derivAt<DType>(i, dout, Order, Accuracy);
      dout[axis] = this->data[i][axis];
      return dout;
    }

    /*!
     * @brief Calculate all derivatives at once
     *
     * @tparam Order
     * @tparam Accuracy
     * @tparam DerivType Output Eigen::Matrix type
     * @param bulk reference vector of DerivType
     */
    template<class DerivType>
    inline void deriv(std::vector<DerivType>& bulk, const int Order, const int Accuracy) const {
      bulk.resize(length);
      for (int i = 0; i < length; i++) {
        bulk[i] = derivAt<DerivType>(i, Order, Accuracy);
      }
    }

    /*!
     * @brief Calculate all derivatives at once
     *
     * @tparam Order
     * @tparam Accuracy
     * @tparam DerivType
     * @return std::vector<DerivType>
     */
    template<class DerivType>
    inline std::vector<DerivType> deriv(const int Order, const int Accuracy) const {
      std::vector<DerivType> bulk;
      deriv<DerivType>(bulk, Order, Accuracy);
      return bulk;
    }

    template<class DerivType>
    inline std::vector<DerivType> allderiv_python(const int Order, const int Accuracy) const {
      return deriv<DerivType>(Order, Accuracy);
    }
    template<class DerivType>
    inline DerivType ithderiv_python(const int i, const int Order, const int Accuracy) const {
      return derivAt<DerivType>(i, Order, Accuracy);
    }


    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<FDDerivArbitrary<DType>>(m, name);
      obj.def(py::init<int, const std::vector<DType>&>());
      obj.def("all_derivs", &FDDerivArbitrary<DType>::allderiv_python<DType>);
      obj.def("deriv", &FDDerivArbitrary<DType>::ithderiv_python<DType>);
    }
  };

}  // namespace ASSET
