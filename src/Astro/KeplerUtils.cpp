#include "KeplerUtils.h"
#include "VectorFunctions/CommonFunctions/RootFinder.h"

namespace ASSET {

	struct KPOP_impl {
		
		static auto C2(double tol) {

			auto psi = Arguments<1>().coeff<0>();
			auto sqsi = sqrt(psi);

			auto f =  IfElseFunction{ psi > tol,
			  (1.0 - cos(sqsi)) / psi,
			  (1.0 - cosh(sqrt(-1.0 * psi))) / psi };
			return GenericFunction<1, 1>(f);

		}

		static auto C3(double tol) {

			auto psi = Arguments<1>().coeff<0>();
			auto sqsi = sqrt(psi);

			auto f =  IfElseFunction{ psi > tol,
			  (sqsi - sin(sqsi)) / (sqsi * psi),
			  (sinh(sqrt(-1.0 * psi)) - sqrt(-1.0*psi)) / sqrt(-1.0*psi * psi * psi) };

			return GenericFunction<1, 1>(f);
		}


		static auto C2C3(double tol) {

			auto psi = Arguments<1>().coeff<0>();
			auto sqsi = sqrt(psi);

			auto f = IfElseFunction{ psi > tol,
			  stack((1.0 - cos(sqsi)) / psi, 
				  (sqsi - sin(sqsi)) / (sqsi * psi)),
			  stack((1.0 - cosh(sqrt(-1.0 * psi))) / psi,
				  (sinh(sqrt(-1.0 * psi)) - sqrt(-1.0 * psi)) / sqrt(-1.0 * psi * psi * psi)  ) };

			return GenericFunction<1, 2>(f);

		}

		static auto Funiv(double mu,double tol) {

			auto args = Arguments<5>();

			auto X0    = args.coeff<0>();
			auto dt    = args.coeff<1>();
			auto r     = args.coeff<2>();
			auto drv   = args.coeff<3>();
			auto alpha = args.coeff<4>();


			auto X02 = X0 * X0;

				
			auto psi = X02 * alpha;

			auto c2 = C2(tol).eval(psi);
			auto c3 = C3(tol).eval(psi);

			auto c2c3 = C2C3(tol).eval(psi);

			auto Nargs = stack(args.head<4>(), psi, c2c3);

			auto [F, dF] = [mu]() {
				auto args = Arguments<7>();

				auto X0 = args.coeff<0>();
				auto dt = args.coeff<1>();
				auto r = args.coeff<2>();
				auto drv = args.coeff<3>();
				auto psi = args.coeff<4>();
				auto c2 = args.coeff<5>();
				auto c3 = args.coeff<6>();

				auto X02 = X0 * X0;
				auto X03 = X0 * X0 * X0;


				auto X0tOmPsiC3 = X0 * (1.0 - psi * c3);
			    auto X02tC2 = X02 * c2;

				auto FU = sqrt(mu) * dt - X03 * c3 - drv * X02tC2 - r * X0tOmPsiC3;
				auto dFU = -1.0*(X02tC2 + drv * X0tOmPsiC3 + r * (1.0 - psi * c2));

				return std::tuple<decltype(FU), decltype(dFU)>{ FU,dFU };
			}();

			auto FU = F.eval(Nargs);
			auto dFU = dF.eval(Nargs);


			auto X0F = ScalarRootFinder<decltype(FU),decltype(dFU)>{ FU, dFU, 12, 1.0e-9 };

			return stack(GenericFunction<5, 1>(X0F), args.tail<4>());

		}


		static auto FGs(double mu, double tol) {

			auto args = Arguments<5>();

			auto X0 = args.coeff<0>();
			auto dt = args.coeff<1>();
			auto r = args.coeff<2>();
			auto drv = args.coeff<3>();
			auto alpha = args.coeff<4>();


			auto X02 = X0 * X0;


			auto psi = X02 * alpha;

			auto c2 = C2(tol).eval(psi);
			auto c3 = C3(tol).eval(psi);

			auto c2c3 = C2C3(tol).eval(psi);


			auto Nargs = stack(args.head<4>(), psi, c2c3);

			auto FG = [mu]() {

				double SQM = sqrt(mu);

				auto args = Arguments<7>();

				auto X0 = args.coeff<0>();
				auto dt = args.coeff<1>();
				auto r = args.coeff<2>();
				auto drv = args.coeff<3>();
				auto psi = args.coeff<4>();
				auto c2 = args.coeff<5>();
				auto c3 = args.coeff<6>();

				auto X02 = X0 * X0;
				auto X03 = X0 * X0 * X0;


				auto X0tOmPsiC3 = X0 * (1.0 - psi * c3);
				auto X02tC2 = X02 * c2;

				auto FU = sqrt(mu) * dt - X03 * c3 - drv * X02tC2 - r * X0tOmPsiC3;
				auto rs =(X02tC2 + drv * X0tOmPsiC3 + r * (1.0 - psi * c2));

				auto f = 1.0 - X02 * c2 / r;
				auto g = dt - (X02 * X0) * c3 / SQM;
				auto fdot = X0 * (psi * c3 - 1.0) * SQM / (rs * r);
				auto gdot = 1.0 - c2 * (X02) / rs;


				return stack(f,g,fdot,gdot);
			}();

			
			return FG.eval(Nargs);

		}


		static auto ApplyRVFG() {

			auto args = Arguments<10>();

			auto R = args.head<3>();
			auto V = args.segment<3, 3>();

			auto f    = args.coeff<6>();
			auto g    = args.coeff<7>();
			auto fdot = args.coeff<8>();
			auto gdot = args.coeff<9>();

			auto Rf = f * R + g * V;
			auto Vf = fdot * R + gdot * V;

			return stack(Rf, Vf);

		}


		static auto Definition(double mu) {

			double     SQM = sqrt(mu);

			auto RVdt = Arguments<7>();

			auto R = RVdt.head<3>();
			auto V = RVdt.segment<3, 3>();
			auto dt = RVdt.coeff<6>();


			auto r = R.norm();
			auto v = V.norm();
			auto drv = R.dot(V)/SQM;
			auto alpha = -1.0*(v * v) / (mu)+(2.0 / r);

			auto X0ell = SQM * dt * alpha;

			
			auto signdt = SignFunction{ dt };
			auto X0hyp = signdt * sqrt(-1.0 / alpha) * log(abs((-2.0 * mu * alpha * dt) /
				(R.dot(V) + signdt) * sqrt(-mu / alpha) * (1.0 - r * alpha)));

			auto X0IG = GenericFunction<7,1> (IfElseFunction{ alpha > 0.0,X0ell,X0hyp });

			auto XF = Funiv(mu, 1.0e-10).eval(stack(X0ell, dt, r, drv, alpha));

			auto FG = FGs(mu, 1.0e-10).eval(XF);

			constexpr bool d = FG.IsVectorizable;

			return ApplyRVFG().eval(stack(RVdt.head<6>(), FG));

		}
	};


}



void ASSET::KeplerUtilsBuild(FunctionRegistry& reg, py::module& m) {

	
	////////////////////////////////////////////////////////////////////////////////////////
	////////////////////              Conversions                  /////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////

	m.def("cartesian_to_classic",
		[](const Vector6<double>& XV, double mu) {
			return cartesian_to_classic(XV, mu);
		});
	m.def("cartesian_to_modified",
		[](const Vector6<double>& XV, double mu) {
			return cartesian_to_modified(XV, mu);
		});
	m.def("classic_to_cartesian",
		[](const Vector6<double>& oelems, double mu) {
			return classic_to_cartesian(oelems, mu);
		});
	m.def("classic_to_modified",
		[](const Vector6<double>& oelems, double mu) {
			return classic_to_modified(oelems, mu);
		});

	m.def("modified_to_cartesian",
		[](const Vector6<double>& meelems, double mu) {
			return modified_to_cartesian(meelems, mu);
		});
	m.def("modified_to_classic",
		[](const Vector6<double>& meelems, double mu) {
			return modified_to_classic(meelems, mu);
		});
	m.def("cartesian_to_classic_true",
		[](const Vector6<double>& XV, double mu) {
			return cartesian_to_classic_true(XV, mu);
		});


	////////////////////////////////////////////////////////////////////////////////////////
	////////////////////         Conversions as ASSET Functions    /////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////

	m.def("modified_to_cartesian",
		[](const GenericFunction<-1,-1> & meelems, double mu) {
			return  GenericFunction<-1, -1> (ModifiedToCartesian(mu).eval(meelems));
		});

	reg.Build_Register<ModifiedToCartesian>(m,"ModifiedToCartesian");


	m.def("cartesian_to_classic",
		[](const GenericFunction<-1, -1>& RV, double mu) {
			return  GenericFunction<-1, -1>(CartesianToClassic(mu).eval(RV));
		});

	reg.Build_Register<CartesianToClassic>(m,"CartesianToClassic");

	m.def("kptest",
		[]( double mu) {
			return  GenericFunction<-1, -1>(KPOP_impl::Definition(mu));
		});


	////////////////////////////////////////////////////////////////////////////////////////
	////////////////////              Propagators                  /////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////


	m.def("propagate_cartesian",
		[](const Vector6<double>& RV, double dt, double mu) {
			return propagate_cartesian(RV,dt, mu);
		});
	m.def("propagate_classic",
		[](const Vector6<double>& oelems, double dt, double mu) {
			return propagate_classic(oelems, dt, mu);
		});

	m.def("propagate_modified",
		[](const Vector6<double>& meelems, double dt, double mu) {
			return propagate_modified(meelems, dt, mu);
		});







}


