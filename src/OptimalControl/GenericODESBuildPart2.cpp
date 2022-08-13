#include "ASSET_OptimalControl.h"

namespace ASSET {

template <int XV, int UV, int PV>
using GODE = GenericODE<GenericFunction<-1, (XV == 1) ? 1 : -1>, XV, UV, PV>;

void GenericODESBuildPart2(FunctionRegistry& reg, py::module& m) {

	GODE<-1, -1, -1>::BuildGenODEModule("ode_x_u_p", m, reg);


}

}  // namespace ASSET