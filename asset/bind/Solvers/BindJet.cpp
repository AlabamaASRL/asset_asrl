#include <bind/Solvers/BindJet.h>

void ASSET::BindJet(py::module& m) {

  auto obj = py::class_<Jet>(m, "Jet");

  obj.def_static(
      "map",
      [](const std::vector<std::shared_ptr<OptimizationProblemBase>>& optprobs, int nt) {
        return Jet::map(optprobs, nt, true);
      },
      py::call_guard<py::gil_scoped_release>());

  obj.def_static(
      "map",
      [](std::function<std::shared_ptr<OptimizationProblemBase>(py::detail::args_proxy)> genfun,
         const std::vector<py::args>& args,
         int nt) { return Jet::map(genfun, args, nt, true); },
      py::call_guard<py::gil_scoped_release>());

  obj.def_static(
      "map",
      [](const std::vector<std::shared_ptr<OptimizationProblemBase>>& optprobs, int nt, bool v) {
        return Jet::map(optprobs, nt, v);
      },
      py::call_guard<py::gil_scoped_release>());

  obj.def_static(
      "map",
      [](std::function<std::shared_ptr<OptimizationProblemBase>(py::detail::args_proxy)> genfun,
         const std::vector<py::args>& args,
         int nt,
         bool v) { return Jet::map(genfun, args, nt, v); },
      py::call_guard<py::gil_scoped_release>());
}
