#include "KeplerUtils.h"

#include "VectorFunctions/CommonFunctions/RootFinder.h"


void ASSET::KeplerUtilsBuild(FunctionRegistry& reg, py::module& m) {


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////              Conversions                  /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  m.def("cartesian_to_classic",
        [](const Vector6<double>& XV, double mu) { return cartesian_to_classic(XV, mu); });
  m.def("cartesian_to_modified",
        [](const Vector6<double>& XV, double mu) { return cartesian_to_modified(XV, mu); });
  m.def("classic_to_cartesian",
        [](const Vector6<double>& oelems, double mu) { return classic_to_cartesian(oelems, mu); });
  m.def("classic_to_modified",
        [](const Vector6<double>& oelems, double mu) { return classic_to_modified(oelems, mu); });

  m.def("modified_to_cartesian",
        [](const Vector6<double>& meelems, double mu) { return modified_to_cartesian(meelems, mu); });
  m.def("modified_to_classic",
        [](const Vector6<double>& meelems, double mu) { return modified_to_classic(meelems, mu); });
  m.def("cartesian_to_classic_true",
        [](const Vector6<double>& XV, double mu) { return cartesian_to_classic_true(XV, mu); });


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////         Conversions as ASSET Functions    /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  m.def("modified_to_cartesian", [](const GenericFunction<-1, -1>& meelems, double mu) {
    return GenericFunction<-1, -1>(ModifiedToCartesian(mu).eval(meelems));
  });

  reg.Build_Register<ModifiedToCartesian>(m, "ModifiedToCartesian");


  m.def("cartesian_to_classic", [](const GenericFunction<-1, -1>& RV, double mu) {
    return GenericFunction<-1, -1>(CartesianToClassic(mu).eval(RV));
  });

  reg.Build_Register<CartesianToClassic>(m, "CartesianToClassic");


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////              Propagators                  /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


  m.def("propagate_cartesian",
        [](const Vector6<double>& RV, double dt, double mu) { return propagate_cartesian(RV, dt, mu); });
  m.def("propagate_classic", [](const Vector6<double>& oelems, double dt, double mu) {
    return propagate_classic(oelems, dt, mu);
  });

  m.def("propagate_modified", [](const Vector6<double>& meelems, double dt, double mu) {
    return propagate_modified(meelems, dt, mu);
  });
}
