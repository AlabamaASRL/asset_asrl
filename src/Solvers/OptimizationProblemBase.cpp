#include "OptimizationProblemBase.h"

void ASSET::OptimizationProblemBase::Build(py::module & m) {
	auto obj = py::class_<OptimizationProblemBase, std::shared_ptr<OptimizationProblemBase>>(m, "OptimizationProblemBase");
	obj.def_readwrite("JetJobMode", &OptimizationProblemBase::JetJobMode);
	obj.def_readwrite("Threads", &OptimizationProblemBase::Threads);
	obj.def_readwrite("optimizer", &OptimizationProblemBase::optimizer);


	obj.def("setJetJobMode",
		py::overload_cast<JetJobModes>(&OptimizationProblemBase::setJetJobMode));
	obj.def("setJetJobMode",
		py::overload_cast<const std::string&>(&OptimizationProblemBase::setJetJobMode));


	obj.def("solve", &OptimizationProblemBase::solve,
		py::call_guard<py::gil_scoped_release>());
	obj.def("optimize", &OptimizationProblemBase::optimize,
		py::call_guard<py::gil_scoped_release>());
	obj.def("solve_optimize", &OptimizationProblemBase::solve_optimize,
		py::call_guard<py::gil_scoped_release>());
	obj.def("solve_optimize_solve", &OptimizationProblemBase::solve_optimize_solve,
		py::call_guard<py::gil_scoped_release>());

	/// <summary>
	/// Probably need to move these enums somewhere else
	/// </summary>
	/// <param name="m"></param>
	py::enum_<JetJobModes>(m, "JetJobModes")
		.value("DoNothing", JetJobModes::DoNothing)
		.value("NotSet", JetJobModes::NotSet)
		.value("Solve", JetJobModes::Solve)
		.value("Optimize", JetJobModes::Optimize)
		.value("SolveOptimize", JetJobModes::SolveOptimize)
		.value("SolveOptimizeSolve", JetJobModes::SolveOptimizeSolve);

}