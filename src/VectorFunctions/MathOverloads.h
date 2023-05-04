/*
File Name: MathOverloads.h

File Description: Overloads the common mathematical functions contained in cmath for ASSET scalar
functions.

////////////////////////////////////////////////////////////////////////////////

Original File Developer : James B. Pezent - jbpezent - jbpezent@crimson.ua.edu

Current File Maintainers:
    1. James B. Pezent - jbpezent         - jbpezent@crimson.ua.edu
    2. Full Name       - GitHub User Name - Current Email
    3. ....


Usage of this source code is governed by the license found
in the LICENSE file in ASSET's top level directory.

*/

#pragma once
#include "CommonFunctions/CommonFunctions.h"

////////////////////////////////////////////////////////////////////////////
/////////////////////// CMath Overloads ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

template<class Derived, int IR>
auto sin(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseSin<Derived>(func.derived());
}

template<class Derived, int IR>
auto cos(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseCos<Derived>(func.derived());
}

template<class Derived, int IR>
auto tan(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseTan<Derived>(func.derived());
}

template<class Derived, int IR>
auto asin(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseArcSin<Derived>(func.derived());
}

template<class Derived, int IR>
auto acos(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseArcCos<Derived>(func.derived());
}

template<class Derived, int IR>
auto atan(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseArcTan<Derived>(func.derived());
}

template<class Derived1, int IR1, class Derived2, int IR2>
auto atan2(const ASSET::DenseFunctionBase<Derived1, IR1, 1>& yf,
           const ASSET::DenseFunctionBase<Derived2, IR2, 1>& xf) {
  return ASSET::ArcTan2Op().eval(ASSET::StackedOutputs<Derived1, Derived2>(yf.derived(), xf.derived()));
}


template<class Derived, int IR>
auto sqrt(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseSqrt<Derived>(func.derived());
}

template<class Derived, int IR>
auto exp(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseExp<Derived>(func.derived());
}

template<class Derived, int IR>
auto log(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseLog<Derived>(func.derived());
}


template<class Derived, int IR>
auto sinh(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseSinH<Derived>(func.derived());
}

template<class Derived, int IR>
auto cosh(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseCosH<Derived>(func.derived());
}

template<class Derived, int IR>
auto tanh(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseTanH<Derived>(func.derived());
}

template<class Derived, int IR>
auto asinh(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseArcSinH<Derived>(func.derived());
}

template<class Derived, int IR>
auto acosh(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseArcCosH<Derived>(func.derived());
}

template<class Derived, int IR>
auto atanh(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseArcTanH<Derived>(func.derived());
}
template<class Derived, int IR>
auto abs(const ASSET::DenseFunctionBase<Derived, IR, 1>& func) {
  return ASSET::CwiseAbs<Derived>(func.derived());
}
