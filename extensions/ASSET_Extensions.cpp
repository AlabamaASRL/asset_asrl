#include "ASSET_Extensions.h"


void ASSET::ExtensionsBuild(FunctionRegistry& reg, py::module& extmod)
{


	extmod.def("cpp_cr3bp", [](double mu) {

		// Example of how to write CR3BP dynamics as C++ vector function and bind it to python
		// After compilation can import in python w/ asset_asrl.Extensions.cpp_cr3bp(mu) 
		// Docs on CPP vector function interface forthcoming but in general it mimics python


		auto args = Arguments<7>();

		auto X = args.head<3>();

		auto V = args.segment<3, 3>();  //.segment<size,start>()

		Vector3<double> p1loc;
		p1loc[0] = -mu;

		Vector3<double> p2loc;
		p2loc[0] = 1.0 - mu;

		auto dvec = X - p1loc;
		auto rvec = X - p2loc;

		auto x = X.coeff<0>();
		auto y = X.coeff<1>();
		auto xdot = V.coeff<0>();
		auto ydot = V.coeff<1>();

		auto rotterms = stack( 2.0 * ydot + x, (-2.0) * xdot + y );

		auto acc = rotterms.padded_lower<1>() -
			(1.0 - mu) * dvec.normalized_power<3>() -
			mu * rvec.normalized_power<3>();

		auto ode = stack(V, acc);

		return GenericFunction<-1, -1>(ode); // Wrap as dynamic sized generic vector function
		
		});


	
}
