#include "ASSET_OptimalControl.h"

namespace ASSET {

	template <int XV, int UV, int PV>
	using GODE = GenericODE<GenericFunction<-1, (XV == 1) ? 1 : -1>, XV, UV, PV>;

	void GenericODESBuildPart6(FunctionRegistry& reg, py::module& m) {

		GODE<6, 0, 0>::BuildGenODEModule("ode_6", m, reg);

		GODE<4, 0, 0>::BuildGenODEModule("ode_4", m, reg);

	}

}  // 