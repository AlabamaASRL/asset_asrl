#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/MatrixFunction.h>
#include <ASSET/VectorFunctions/CommonFunctions/MatrixProduct.h>
#include <ASSET/VectorFunctions/DynamicStack.h>
#include <ASSET/VectorFunctions/OperatorOverloads.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>
#include <bind/pch.h>

namespace ASSET {

  void BindMatrixFunction(py::module&);

}
