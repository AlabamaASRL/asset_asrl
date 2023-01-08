#pragma once

#include "ConstraintFunction.h"
#include "NonLinearProgram.h"
#include "OptimizationProblemBase.h"
#include "ObjectiveFunction.h"
#include "PSIOPT.h"
#include "Jet.h"
#include "mkl.h"
#include "OptimizationProblem.h"

namespace ASSET {

	void SolversBuild(FunctionRegistry& reg, py::module& m) {
		//auto sol = m.def_submodule("Solvers","SubModule Containing PSIOPT,NLP, and Solver Flags");
	
		auto& sol = reg.getSolversModule();
		int DSECOND = dsecnd();
		PSIOPT::Build(sol);
		OptimizationProblemBase::Build(sol);
		Jet::Build(sol);
		OptimizationProblem::Build(sol);

	}


}
