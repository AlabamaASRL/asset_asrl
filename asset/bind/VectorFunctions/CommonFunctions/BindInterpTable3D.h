#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/InterpTable3D.h>
#include <ASSET/VectorFunctions/CommonFunctions/NestedFunction.h>
#include <ASSET/VectorFunctions/CommonFunctions/Stacked.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>
#include <bind/pch.h>
#include <pybind11/eigen/tensor.h>

namespace ASSET {

  void BindInterpTable3D(py::module&);

}
