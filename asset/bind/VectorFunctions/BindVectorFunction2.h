#pragma once

#include <ASSET/VectorFunctions/DynamicStack.h>
#include <bind/pch.h>

#include "FunctionRegistry.h"
#include "PythonArgParsing.h"

namespace ASSET {

  void BindVectorFunction2(FunctionRegistry&, py::module&);

}
