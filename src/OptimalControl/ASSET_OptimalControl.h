#pragma once

#include "FDDerivArbitrary.h"
#include "FDDerivUniform.h"
#include "LGLInterpFunctions.h"
#include "ODE.h"
#include "ODEPhase.h"
#include "OptimalControlProblem.h"
#include "pch.h"

namespace ASSET {

  void GenericODESBuildPart1(FunctionRegistry& reg, py::module& m);
  void GenericODESBuildPart2(FunctionRegistry& reg, py::module& m);
  void GenericODESBuildPart3(FunctionRegistry& reg, py::module& m);
  void GenericODESBuildPart4(FunctionRegistry& reg, py::module& m);
  void GenericODESBuildPart5(FunctionRegistry& reg, py::module& m);
  void GenericODESBuildPart6(FunctionRegistry& reg, py::module& m);

  void OptimalControlBuild(FunctionRegistry& reg, py::module& m);

}  // namespace ASSET
