#pragma once

#include "DenseFunctionBase.h"
#include "DenseScalarFunctionBase.h"

namespace ASSET {

  template<class Derived, int IR, int OR>
  struct DenseFunction : DenseFunctionBase<Derived, IR, OR> {
    using Base = DenseFunctionBase<Derived, IR, OR>;
  };

  template<class Derived, int IR>
  struct DenseFunction<Derived, IR, 1> : DenseScalarFunctionBase<Derived, IR> {
    using Base = DenseScalarFunctionBase<Derived, IR>;
  };
}  // namespace ASSET
