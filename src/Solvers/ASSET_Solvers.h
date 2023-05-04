#pragma once

#include "ConstraintFunction.h"
#include "Jet.h"
#include "NonLinearProgram.h"
#include "ObjectiveFunction.h"
#include "OptimizationProblem.h"
#include "OptimizationProblemBase.h"
#include "PSIOPT.h"
#include "mkl.h"

namespace ASSET {

  void SolversBuild(FunctionRegistry& reg, py::module& m) {
    // auto sol = m.def_submodule("Solvers","SubModule Containing PSIOPT,NLP, and Solver Flags");

    auto& sol = reg.getSolversModule();
    int DSECOND = dsecnd();
    PSIOPT::Build(sol);
    OptimizationProblemBase::Build(sol);
    Jet::Build(sol);
    OptimizationProblem::Build(sol);
  }


}  // namespace ASSET
