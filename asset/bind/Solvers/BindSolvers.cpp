#include <bind/Solvers/BindSolvers.h>

void ASSET::BindSolvers(FunctionRegistry& reg, py::module& m) {
  // auto sol = m.def_submodule("Solvers","SubModule Containing PSIOPT,NLP, and Solver Flags");

  auto& sol = reg.getSolversModule();
  int DSECOND = dsecnd();
  BindPSIOPT(sol);
  BindOptimizationProblemBase(sol);
  BindJet(sol);
  BindOptimizationProblem(sol);
}
