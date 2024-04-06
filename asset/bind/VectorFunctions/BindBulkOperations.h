#pragma once

#include <ASSET/VectorFunctions/DynamicStack.h>
#include <ASSET/VectorFunctions/DynamicSum.h>
#include <bind/VectorFunctions/FunctionRegistry.h>
#include <bind/pch.h>

#include "PythonArgParsing.h"

namespace ASSET {

  void BindBulkOperations(FunctionRegistry&, py::module&);

}
