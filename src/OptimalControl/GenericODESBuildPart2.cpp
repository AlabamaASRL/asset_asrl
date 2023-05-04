#include "ASSET_OptimalControl.h"

namespace ASSET {


  void GenericODESBuildPart2(FunctionRegistry& reg, py::module& m) {

    PythonGenericODE<-1, -1, -1>::BuildGenODEModule("ode_x_u_p", m, reg);
  }

}  // namespace ASSET