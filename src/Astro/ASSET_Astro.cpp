#include "ASSET_Astro.h"

#include "CR3BPModel.h"
#include <pybind11/numpy.h>
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

  mod.def("PinesGravityRHS", [](double mu, double reference_radius, py::array_t<double, py::array::c_style | py::array::forcecast> Cnm,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> Snm, int max_degree) {
    if (max_degree < 0) throw std::invalid_argument("max_degree must be non-negative");
    const auto cbuf = Cnm.request();
    const auto sbuf = Snm.request();
    const int side = max_degree + 1;
    if (cbuf.ndim != 2 || sbuf.ndim != 2 || cbuf.shape[0] < side || cbuf.shape[1] < side ||
        sbuf.shape[0] < side || sbuf.shape[1] < side) {
      throw std::invalid_argument("Cnm and Snm must be two-dimensional arrays at least (max_degree + 1) square");
    }
    const auto* cdata = static_cast<const double*>(cbuf.ptr);
    const auto* sdata = static_cast<const double*>(sbuf.ptr);
    std::vector<double> c(side * side), s(side * side);
    for (int n = 0; n < side; ++n) for (int m = 0; m < side; ++m) {
      c[n * side + m] = cdata[n * cbuf.shape[1] + m];
      s[n * side + m] = sdata[n * sbuf.shape[1] + m];
    }
    return GenericFunction<-1, -1>(PinesGravityRHS(mu, reference_radius, max_degree, std::move(c), std::move(s)));
  });

  mod.def("NonIdealSolarSail", [](double mu, double beta, double n1, double n2, double t1) {
    return GenericFunction<-1, -1>(NonIdealSolarSail_Impl::Definition(mu, beta, n1, n2, t1));
  });
}
