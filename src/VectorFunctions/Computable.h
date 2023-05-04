#pragma once

#include "ComputableBase.h"

namespace ASSET {

  template<class Derived, int IR, int OR>
  struct Computable : ComputableBase<Derived, IR, OR> {
    using Base = ComputableBase<Derived, IR, OR>;
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Gradient = typename Base::template Gradient<Scalar>;

    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using VectorBaseRef = Eigen::MatrixBase<Scalar>&;
  };

  ///// Scalar Specialization
  template<class Derived, int IR>
  struct Computable<Derived, IR, 1> : ComputableBase<Derived, IR, 1> {
    using Base = ComputableBase<Derived, IR, 1>;

    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Gradient = typename Base::template Gradient<Scalar>;

    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;
    template<class Scalar>
    using VectorBaseRef = Eigen::MatrixBase<Scalar>&;
  };

}  // namespace ASSET
