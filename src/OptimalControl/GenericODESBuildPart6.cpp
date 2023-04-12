#include "ASSET_OptimalControl.h"

namespace ASSET {

	
	
	void GenericODESBuildPart6(FunctionRegistry& reg, py::module& m) {

		PythonGenericODE<6, 0, 0>::BuildGenODEModule("ode_6", m, reg);

		PythonGenericODE<4, 0, 0>::BuildGenODEModule("ode_4", m, reg);

		m.def("lorenz4", []() {

		auto X = Arguments<5>();

		auto x1 = X.coeff<0>();
		auto x2 = X.coeff<1>();
		auto x3 = X.coeff<2>();
		auto x4 = X.coeff<3>();

		auto x1dot = x4 * (x2 - x3)  ;
		auto x2dot = x1 * (x3 - x4)  ;
		auto x3dot = x2 * (x4 - x1)  ;
		auto x4dot = x3 * (x1 - x2)  ;


		Vector4 < double> ones;
		ones.setOnes();

		auto ode = stack(x1dot, x2dot, x3dot, x4dot) - X.head<4>() + ones;

		return GenericFunction<-1, -1>(ode);
		});

	}

}  // 