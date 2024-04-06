#pragma once

namespace ASSET {

  struct DirectAssignment {};
  struct PlusEqualsAssignment {};
  struct MinusEqualsAssignment {};
  struct AliasedDirectAssignment {};

  template<class Scalar>
  struct ScaledDirectAssignment {
    Scalar value;
    ScaledDirectAssignment(Scalar v) : value(v) {
    }
  };

  template<class Scalar>
  struct ScaledPlusEqualsAssignment {
    Scalar value;
    ScaledPlusEqualsAssignment(Scalar v) : value(v) {
    }
  };

}  // namespace ASSET