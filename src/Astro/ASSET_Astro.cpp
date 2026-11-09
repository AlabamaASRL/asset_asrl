#include "ASSET_Astro.h"

#include "CR3BPModel.h"
#include "ForceField3.h"
#include "SpiceForceFrame.h"
#include "SpicePxform.h"
void ASSET::AstroBuild(FunctionRegistry& reg, py::module& m) {
  auto mod = m.def_submodule("Astro");

  BuildKeplerMod(reg, mod);
  KeplerUtilsBuild(reg, mod);
  LambertSolversBuild(reg, mod);

  /////////////////////////////////////////////////////////////
  //////////// SPICE frame transforms and gravity ////////////
  /////////////////////////////////////////////////////////////

  mod.def("SpiceLoadKernels",
    [](const std::vector<std::string>& files) {
      spice_detail::configure_spice_errors();
      for (const auto& f : files)
        spice_detail::load_kernel(f);
    },
    py::arg("files"),
    "Load SPICE kernel files (furnsh) into the global kernel pool.");

  reg.Build_Register<SpicePxform>(mod, "SpicePxform");
  reg.Build_Register<SpicePxformApply>(mod, "SpicePxformApply");
  py::class_<ForceField3>(mod, "ForceField3",
    "Abstract 3->3 position force field, the base of fusable kernels such as PinesGravity.");
  BuildPinesGravity(reg, mod);
  reg.Build_Register<SpiceForceFrame>(mod, "SpiceForceFrame");


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
