#pragma once
#include "Utils/SizingHelpers.h"
#include "pch.h"

namespace ASSET {

  template<class DODE>
  struct Blocked_ODE_Wrapper : DODE {
    static const int UV = 0;
    static const int PV = SZ_SUM<DODE::PV, DODE::UV>::value;
    static const int XtUV = DODE::XtV;
    using Base = DODE;

    inline int XtUVars() const {
      return Base::XtVars();
    }
    inline int UVars() const {
      return 0;
    }
    inline int PVars() const {
      return Base::UVars() + Base::PVars();
    }

    Blocked_ODE_Wrapper() {};
    Blocked_ODE_Wrapper(const DODE& ode) : Base(ode) {
    }
  };

}  // namespace ASSET