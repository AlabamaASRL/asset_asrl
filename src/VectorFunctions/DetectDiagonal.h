#pragma once

#include "pch.h"

namespace ASSET {

  template<class T>
  struct Is_EigenDiagonalMatrix : std::false_type {};

  template<class Scalar, int Rows_Cols>
  struct Is_EigenDiagonalMatrix<Eigen::DiagonalMatrix<Scalar, Rows_Cols>> : std::true_type {};

  template<class T>
  struct Is_EigenDiagonalMatrix<Eigen::DiagonalWrapper<T>> : std::true_type {};
  template<class T>
  struct Is_EigenDiagonalMatrix<const Eigen::DiagonalWrapper<T>> : std::true_type {};

}  // namespace ASSET
