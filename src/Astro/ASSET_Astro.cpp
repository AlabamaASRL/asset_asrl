#include "ASSET_Astro.h"

#include "CR3BPModel.h"
void ASSET::AstroBuild(FunctionRegistry& reg, py::module& m) {
  auto mod = m.def_submodule("Astro");

  BuildKeplerMod(reg, mod);
  KeplerUtilsBuild(reg, mod);
  LambertSolversBuild(reg, mod);


  /////////////////////////////////////////////////////////////
  //////////// Binding Misc CPP Functions here for now ////////
  /////////////////////////////////////////////////////////////

 
  mod.def("ModifiedDynamics", [](double mu) { return GenericFunction<-1, -1>(MEEDynamics(mu)); });


  mod.def("J2Cartesian", [](double mu, double J2, double Rb) {
    return GenericFunction<-1, -1>(J2Cartesian_Impl::Definition(mu, J2, Rb));
  });

  mod.def("NonIdealSolarSail", [](double mu, double beta, double n1, double n2, double t1) {
    return GenericFunction<-1, -1>(NonIdealSolarSail_Impl::Definition(mu, beta, n1, n2, t1));
  });
}
