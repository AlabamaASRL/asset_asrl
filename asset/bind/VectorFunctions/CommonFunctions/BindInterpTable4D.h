#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/InterpTable4D.h>
#include <ASSET/VectorFunctions/CommonFunctions/NestedFunction.h>
#include <ASSET/VectorFunctions/CommonFunctions/Stacked.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>
#include <bind/pch.h>
#include <pybind11/eigen/tensor.h>

namespace ASSET {

  void BindInterpTable4D(py::module&);

}
