#pragma once
#include "BenchUtils.h"
#include "CRTPBase.h"
#include "FunctionReturnType.h"
#include "GetCoreCount.h"
#include "LambdaJumpTable.h"
#include "MathFunctions.h"
#include "STDExtensions.h"
#include "SizingHelpers.h"
#include "ThreadPool.h"
#include "TupleIterator.h"
#include "TypeErasure.h"
#include "TypeName.h"


namespace ASSET {

  void UtilsBuild(py::module& m) {
    auto um = m.def_submodule("Utils", "Contains miscilanaeous utilities");
    um.def("get_core_count", &ASSET::get_core_count);
  }

}  // namespace ASSET
