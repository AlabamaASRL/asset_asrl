#pragma once
#include "J2.h"
#include "KeplerModel.h"
#include "KeplerPropagator.h"
#include "KeplerUtils.h"
#include "LambertSolvers.h"
#include "MEEDynamics.h"
#include "ThrusterModels.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "pch.h"


namespace ASSET {

  void AstroBuild(FunctionRegistry& reg, py::module& m);


}