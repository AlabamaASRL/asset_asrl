#pragma once
#include "pch.h"
namespace ASSET {

  template<class T>
  struct Is_SuperScalar : std::false_type {};

  template<class Scalar, int Sz>
  struct Is_SuperScalar<SuperScalarType<Scalar, Sz>> : std::true_type {};

}  // namespace ASSET