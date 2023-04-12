#include "ASSET_Astro.h"
#include "CR3BPModel.h"
void ASSET::AstroBuild(FunctionRegistry& reg, py::module& m)
{
	auto mod = m.def_submodule("Astro");

	BuildKeplerMod(reg, mod);
	KeplerUtilsBuild(reg, mod);
	LambertSolversBuild(reg, mod);

	
	
	/////////////////////////////////////////////////////////////
	//////////// Binding Misc CPP Functions here for now ////////
	/////////////////////////////////////////////////////////////

	mod.def("ModifiedDynamics",
		[](double mu) {
			return GenericFunction<-1,-1>(ModifiedDynamics_Impl::Definition(mu));
		});

	mod.def("J2Cartesian",
		[](double mu, double J2, double Rb) {
			return GenericFunction<-1, -1>(J2Cartesian_Impl::Definition(mu,J2,Rb));
		});

	mod.def("NonIdealSolarSail",
		[](double mu, double beta, double n1, double n2, double t1) {
			return GenericFunction<-1, -1>(NonIdealSolarSail_Impl::Definition(mu, beta,n1,n2,t1));
		});


	mod.def("cr3bp", [](double mu) {

		auto args = Arguments<7>();

		auto X = args.head<3>();

		auto V = args.segment<3, 3>();

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


		auto rotterms = StackedOutputs{ 2.0 * ydot + x, (-2.0) * xdot + y };

		auto acc = rotterms.padded_lower<1>() -
			(1.0 - mu) * dvec.normalized_power<3>() -
			mu * rvec.normalized_power<3>();

		auto ode = StackedOutputs{ V, acc };

		return GenericFunction<-1, -1>(ode);

	});


	mod.def("cr3bpx", [](double mu) {

		auto args = Arguments<-1>(7);

		auto X = args.head(3);

		auto V = args.segment(3,3);

		Vector3<double> p1loc;
		p1loc[0] = -mu;

		Vector3<double> p2loc;
		p2loc[0] = 1.0 - mu;

		auto dvec = X - p1loc;
		auto rvec = X - p2loc;

		auto x = X.coeff(0);

		auto y = X.coeff(1);

		auto xdot = V.coeff(0);
		auto ydot = V.coeff(1);


		auto rotterms = StackedOutputs{ 2.0 * ydot + x, (-2.0) * xdot + y };

		auto acc = rotterms.padded_lower(1) -
			(1.0 - mu) * dvec.normalized_power<3>() -
			mu * rvec.normalized_power<3>();

		auto ode = StackedOutputs{ V, acc };

		return GenericFunction<-1, -1>(ode);

		});


}
