#pragma once

#include "DenseFunction.h"

namespace ASSET {

  enum DenseDerivativeModes {
    Analytic,
    FDiffFwd,
    FDiffCentArray,
    AutodiffFwd,
  };

  template<class Derived, int IR, int OR, int Jmode>
  struct DenseFirstDerivatives : DenseFunction<Derived, IR, OR> {
    using Base = DenseFunction<Derived, IR, OR>;
  };

  template<class Derived, int IR, int OR, int Jmode, int Hmode>
  struct DenseSecondDerivatives : DenseFirstDerivatives<Derived, IR, OR, Jmode> {
    using Base = DenseFirstDerivatives<Derived, IR, OR, Jmode>;
  };

  template<class Derived, int IR, int OR, int Jmode, int Hmode>
  struct DenseDerivatives : DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode> {
    using Base = DenseSecondDerivatives<Derived, IR, OR, Jmode, Hmode>;
    DENSE_FUNCTION_BASE_TYPES(Base)
  };

}  // namespace ASSET
