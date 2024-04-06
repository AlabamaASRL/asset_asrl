#include <bind/Solvers/BindOptimizationProblemBase.h>

void ASSET::BindOptimizationProblemBase(py::module& m) {
  auto obj = py::class_<OptimizationProblemBase, std::shared_ptr<OptimizationProblemBase>>(
      m, "OptimizationProblemBase");
  obj.def_readwrite("JetJobMode", &OptimizationProblemBase::JetJobMode);
  obj.def_readwrite("Threads", &OptimizationProblemBase::Threads);
  obj.def_readonly("optimizer", &OptimizationProblemBase::optimizer);

  obj.def("setThreads",
          py::overload_cast<int, int>(&OptimizationProblemBase::setThreads),
          py::arg("FuncThreads"),
          py::arg("KKTThreads"));

  obj.def("setThreads", py::overload_cast<int>(&OptimizationProblemBase::setThreads));

  obj.def("setJetJobMode",
          py::overload_cast<OptimizationProblemBase::JetJobModes>(&OptimizationProblemBase::setJetJobMode));
  obj.def("setJetJobMode", py::overload_cast<const std::string&>(&OptimizationProblemBase::setJetJobMode));

  obj.def("solve", &OptimizationProblemBase::solve, py::call_guard<py::gil_scoped_release>());
  obj.def("optimize", &OptimizationProblemBase::optimize, py::call_guard<py::gil_scoped_release>());
  obj.def(
      "solve_optimize", &OptimizationProblemBase::solve_optimize, py::call_guard<py::gil_scoped_release>());
  obj.def("solve_optimize_solve",
          &OptimizationProblemBase::solve_optimize_solve,
          py::call_guard<py::gil_scoped_release>());
  obj.def(
      "optimize_solve", &OptimizationProblemBase::optimize_solve, py::call_guard<py::gil_scoped_release>());

  /// <summary>
  /// Probably need to move these enums somewhere else
  /// </summary>
  /// <param name="m"></param>
  py::enum_<OptimizationProblemBase::JetJobModes>(m, "JetJobModes")
      .value("DoNothing", OptimizationProblemBase::JetJobModes::DoNothing)
      .value("NotSet", OptimizationProblemBase::JetJobModes::NotSet)
      .value("Solve", OptimizationProblemBase::JetJobModes::Solve)
      .value("Optimize", OptimizationProblemBase::JetJobModes::Optimize)
      .value("SolveOptimize", OptimizationProblemBase::JetJobModes::SolveOptimize)
      .value("SolveOptimizeSolve", OptimizationProblemBase::JetJobModes::SolveOptimizeSolve)
      .value("OptimizeSolve", OptimizationProblemBase::JetJobModes::OptimizeSolve);
}
