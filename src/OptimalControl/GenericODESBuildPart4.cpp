#include "ASSET_OptimalControl.h"

namespace ASSET {

  template<int XV, int UV, int PV>
  using GODE = GenericODE<GenericFunction<-1, (XV == 1) ? 1 : -1>, XV, UV, PV>;

  void GenericODESBuildPart4(FunctionRegistry& reg, py::module& m) {

    PythonGenericODE<7, 3, 0>::BuildGenODEModule("ode_7_3", m, reg);
  }

}  // namespace ASSET