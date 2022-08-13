#include "ASSET_OptimalControl.h"

namespace ASSET {

	template <int XV, int UV, int PV>
	using GODE = GenericODE<GenericFunction<-1, (XV == 1) ? 1 : -1>, XV, UV, PV>;

	void GenericODESBuildPart3(FunctionRegistry& reg, py::module& m) {

		GODE<6, 3, 0>::BuildGenODEModule("ode_6_3", m, reg);
	}

}  // namespace ASSET