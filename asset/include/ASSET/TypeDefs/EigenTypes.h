#pragma once

#include <Eigen/Core>

namespace ASSET {

  template<class Scalar, int rows>
  using Vector = Eigen::Matrix<Scalar, rows, 1>;

  template<class Scalar>
  using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  template<class Scalar>
  using Vector1 = Eigen::Matrix<Scalar, 1, 1>;

  template<class Scalar>
  using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

  template<class Scalar>
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  template<class Scalar>
  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;

  template<class Scalar>
  using Vector5 = Eigen::Matrix<Scalar, 5, 1>;

  template<class Scalar>
  using Vector6 = Eigen::Matrix<Scalar, 6, 1>;

  template<class Scalar>
  using Vector7 = Eigen::Matrix<Scalar, 7, 1>;

  template<class Scalar>
  using Vector8 = Eigen::Matrix<Scalar, 8, 1>;

  template<class Scalar>
  using Vector9 = Eigen::Matrix<Scalar, 9, 1>;

  template<class Scalar>
  using Vector10 = Eigen::Matrix<Scalar, 10, 1>;

  template<class Scalar>
  using Vector11 = Eigen::Matrix<Scalar, 11, 1>;

  template<class Scalar>
  using Vector12 = Eigen::Matrix<Scalar, 12, 1>;

  template<class Scalar>
  using Vector13 = Eigen::Matrix<Scalar, 13, 1>;

  template<class Scalar>
  using Vector14 = Eigen::Matrix<Scalar, 14, 1>;

  template<class Type>
  using EigenRef = Eigen::Ref<Type>;

  template<class Type>
  using ConstEigenRef = const Eigen::Ref<const Type>&;

  template<class Scalar, int MaxSize>
  using MaxVector = Eigen::Matrix<Scalar, -1, 1, 0, MaxSize, 1>;

  template<class Scalar, int MaxSize>
  using MaxMatrix = Eigen::Matrix<Scalar, -1, -1, 0, MaxSize, MaxSize>;

  using IOint = int;

  using DomainMatrix = Eigen::Matrix<IOint, 2, -1>;

  template<class Scalar, int Sz>
  using SuperScalarType = Eigen::Array<Scalar, Sz, 1>;

  using DefaultSuperScalar = SuperScalarType<double, 4>;

}  // namespace ASSET
