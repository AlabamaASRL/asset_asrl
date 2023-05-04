#pragma once

#include "ComputableBase.h"

namespace ASSET {

  template<class Derived, int IR, int OR>
  struct SparseFunctionBase : ComputableBase<Derived, IR, OR> {
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Gradient = typename Base::template Gradient<Scalar>;

    template<class Scalar>
    using Jacobian = Eigen::SparseMatrix<Scalar>;
    template<class Scalar>
    using Hessian = Eigen::SparseMatrix<Scalar>;

    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using VectorBaseRef = Eigen::MatrixBase<Scalar>&;

    template<class Scalar>
    using ConstMatrixBaseRef = const Eigen::SparseMatrixBase<Scalar>&;
    template<class Scalar>
    using MatrixBaseRef = Eigen::SparseMatrixBase<Scalar>&;

   protected:
    mutable Jacobian<double> JacobianTemplate;
    mutable Hessian<double> HessianTemplate;
  };

}  // namespace ASSET
