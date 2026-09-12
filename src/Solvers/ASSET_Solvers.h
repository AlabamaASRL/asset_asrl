#pragma once

#include "ConstraintFunction.h"
#include "Jet.h"
#include "NonLinearProgram.h"
#include "ObjectiveFunction.h"
#include "OptimizationProblem.h"
#include "OptimizationProblemBase.h"
#include "PSIOPT.h"
#ifdef USE_ACCELERATE_SPARSE
#include "AccelerateInterface.h"
#else
#include "mkl.h"
#endif

namespace ASSET {

  void SolversBuild(FunctionRegistry& reg, py::module& m) {
    // auto sol = m.def_submodule("Solvers","SubModule Containing PSIOPT,NLP, and Solver Flags");

    auto& sol = reg.getSolversModule();
#ifndef USE_ACCELERATE_SPARSE
    int DSECOND = dsecnd();
#endif
    PSIOPT::Build(sol);
    OptimizationProblemBase::Build(sol);
    Jet::Build(sol);
    OptimizationProblem::Build(sol);
  }


}  // namespace ASSET
