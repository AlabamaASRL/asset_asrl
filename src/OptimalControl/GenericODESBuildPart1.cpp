#include "ASSET_OptimalControl.h"

namespace ASSET {


  void GenericODESBuildPart1(FunctionRegistry& reg, py::module& m) {

    PythonGenericODE<-1, 0, 0>::BuildGenODEModule("ode_x", m, reg);
    PythonGenericODE<-1, -1, 0>::BuildGenODEModule("ode_x_u", m, reg);
  }

}  // namespace ASSET