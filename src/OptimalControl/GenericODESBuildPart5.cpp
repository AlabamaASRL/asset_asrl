#include "ASSET_OptimalControl.h"

namespace ASSET {

  template<int XV, int UV, int PV>
  using GODE = GenericODE<GenericFunction<-1, (XV == 1) ? 1 : -1>, XV, UV, PV>;

  void GenericODESBuildPart5(FunctionRegistry& reg, py::module& m) {

    PythonGenericODE<2, 1, 0>::BuildGenODEModule("ode_2_1", m, reg);
  }

}  // namespace ASSET