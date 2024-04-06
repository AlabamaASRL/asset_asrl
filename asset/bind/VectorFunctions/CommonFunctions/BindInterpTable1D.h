#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/InterpTable1D.h>
#include <ASSET/VectorFunctions/CommonFunctions/NestedFunction.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>
#include <bind/pch.h>

namespace ASSET {

  void BindInterpTable1D(py::module&);

}
