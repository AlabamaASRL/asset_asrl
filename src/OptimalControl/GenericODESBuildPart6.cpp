#include "ASSET_OptimalControl.h"

namespace ASSET {

	struct Heyoka_Impl : ODESize<4, 0, 0> {
		static auto Definition(double mu) {
			auto args = Arguments<5>();

			auto x = args.coeff<0>();
			auto y = args.coeff<1>();
			auto vxdot = -2.0 * x * y - x;
			auto vydot = y * y - y - x * x;

			auto ode = stack(args.segment<2, 2>(), vxdot, vydot);
			return ode;
		}
	};

	BUILD_ODE_FROM_EXPRESSION(Heyoka, Heyoka_Impl, double);
	
	void GenericODESBuildPart6(FunctionRegistry& reg, py::module& m) {

		PythonGenericODE<6, 0, 0>::BuildGenODEModule("ode_6", m, reg);

		PythonGenericODE<4, 0, 0>::BuildGenODEModule("ode_4", m, reg);

		Heyoka::BuildODEModule("Heyoka", m, reg);

		m.def("HeyokaODE", []() {
			auto args = Arguments<5>();

			auto x = args.coeff<0>();
			auto y = args.coeff<1>();
			auto vxdot = -2.0 * x * y - x;
			auto vydot = y * y - y - x * x;

			auto ode = stack(args.segment<2, 2>(), vxdot, vydot);
			return GenericFunction<-1, -1>(ode);
			});

	}

}  // 