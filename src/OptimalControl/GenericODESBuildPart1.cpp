#include "ASSET_OptimalControl.h"

namespace ASSET {

template <int XV, int UV, int PV>
using GODE = GenericODE<GenericFunction<-1, -1>, XV, UV, PV>;

void GenericODESBuildPart1(FunctionRegistry& reg, py::module& m) {
 
	GODE<-1, 0, 0>::BuildGenODEModule("ode_x", m, reg);
	GODE<-1, -1, 0>::BuildGenODEModule("ode_x_u", m, reg);

 
}

}  // namespace ASSET