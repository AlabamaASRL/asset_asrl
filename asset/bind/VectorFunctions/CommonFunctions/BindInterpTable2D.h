#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/InterpTable2D.h>
#include <ASSET/VectorFunctions/CommonFunctions/NestedFunction.h>
#include <ASSET/VectorFunctions/CommonFunctions/Stacked.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>
#include <bind/pch.h>

namespace ASSET {

  void BindInterpTable2D(py::module&);

}
