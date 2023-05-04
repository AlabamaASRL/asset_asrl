#include "KeplerModel.h"

void ASSET::BuildKeplerMod(FunctionRegistry& reg, py::module& m) {
  auto odemod = m.def_submodule("Kepler");
  reg.template Build_Register<Kepler>(odemod, "ode");
  reg.template Build_Register<Integrator<Kepler>>(odemod, "integrator");
  reg.Build_Register<KeplerPropagator>(odemod, "KeplerPropagator");
  KeplerPhase::Build(odemod);
}

void ASSET::Kepler::Build(py::module& m, const char* name) {
  auto obj = py::class_<Kepler>(m, name).def(py::init<double>());
  Base::DenseBaseBuild(obj);
  obj.def("phase", [](const Kepler& od, TranscriptionModes Tmode) {
    return std::make_shared<KeplerPhase>(od, Tmode);
  });
  obj.def(
      "phase",
      [](const Kepler& od, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd>& Traj, int numdef) {
        return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef);
      });

  obj.def("phase",
          [](const Kepler& od, std::string Tmode) { return std::make_shared<KeplerPhase>(od, Tmode); });
  obj.def("phase",
          [](const Kepler& od, std::string Tmode, const std::vector<Eigen::VectorXd>& Traj, int numdef) {
            return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef);
          });

  Integrator<Kepler>::BuildConstructors(obj);
}
