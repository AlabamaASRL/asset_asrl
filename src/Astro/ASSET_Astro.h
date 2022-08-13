#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "KeplerPropagator.h"
#include "LambertSolver.h"
#include "KeplerModel.h"
#include "KeplerUtils.h"
#include "LambertSolverMultiRevTest.h"
#include "MEEDynamics.h"
#include "J2.h"
#include "ThrusterModels.h"
#include "pch.h"


namespace ASSET {

	void AstroBuild(FunctionRegistry& reg, py::module& m);


}