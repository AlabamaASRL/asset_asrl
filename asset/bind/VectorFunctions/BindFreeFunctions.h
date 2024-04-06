#pragma once

#include <bind/pch.h>

#include "FunctionRegistry.h"
#include "common/IfElseBuild.h"

namespace ASSET {

  void BindFreeFunctions(FunctionRegistry&, py::module&);

}
