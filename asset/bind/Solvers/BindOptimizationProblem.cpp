#include <bind/Solvers/BindOptimizationProblem.h>

static void ASSET::BindOptimizationProblem(py::module& m) {
  using VectorFunctionalX = GenericFunction<-1, -1>;
  using ScalarFunctionalX = GenericFunction<-1, 1>;
  using VectorXi = Eigen::VectorXi;
  using MatrixXi = Eigen::MatrixXi;

  auto obj = py::class_<OptimizationProblem, std::shared_ptr<OptimizationProblem>, OptimizationProblemBase>(
      m, "OptimizationProblem");

  obj.def(py::init<>());

  obj.def("setVars", &OptimizationProblem::setVars);
  obj.def("returnVars", &OptimizationProblem::returnVars);

  obj.def(
      "addEqualCon",
      py::overload_cast<VectorFunctionalX, const std::vector<VectorXi>&>(&OptimizationProblem::addEqualCon));

  obj.def("addEqualCon", py::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addEqualCon));

  obj.def("addInequalCon",
          py::overload_cast<VectorFunctionalX, const std::vector<VectorXi>&>(
              &OptimizationProblem::addInequalCon));

  obj.def("addInequalCon",
          py::overload_cast<VectorFunctionalX, VectorXi>(&OptimizationProblem::addInequalCon));

  obj.def(
      "addObjective",
      py::overload_cast<ScalarFunctionalX, const std::vector<VectorXi>&>(&OptimizationProblem::addObjective));

  obj.def("addObjective", py::overload_cast<ScalarFunctionalX, VectorXi>(&OptimizationProblem::addObjective));

  /*obj.def("solve", &OptimizationProblem::solve,
      py::call_guard<py::gil_scoped_release>());
  obj.def("optimize", &OptimizationProblem::optimize,
      py::call_guard<py::gil_scoped_release>());
  obj.def("solve_optimize", &OptimizationProblem::solve_optimize,
      py::call_guard<py::gil_scoped_release>());*/
}
