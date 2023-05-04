#pragma once

#include "Utils/SizingHelpers.h"

namespace ASSET {

  template<int _CS, int ODEXV, int ODEUV, int ODEPV>
  struct DefectConstSizes {
    static const int CS = _CS;
    static const int IS = SZ_NEG<CS - 1>::value;
    static const int XTU = SZ_SUM<ODEXV, ODEUV, 1>::value;
    static const int DefIRC = SZ_SUM<SZ_PROD<CS, XTU>::value, ODEPV>::value;
    static const int DefORC = SZ_PROD<IS, ODEXV>::value;
  };

}  // namespace ASSET
