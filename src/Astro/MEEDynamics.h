#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {

	
	struct ModifiedDynamics_Impl {
		static auto Definition(double mu) {

			auto args = Arguments<9>();

			auto p = args.coeff<0>();
			auto f = args.coeff<1>();
			auto g = args.coeff<2>();
			auto h = args.coeff<3>();
			auto k = args.coeff<4>();
			auto L = args.coeff<5>();

			//auto ur = args.coeff<6>();
			//auto ut = args.coeff<7>();
			//auto un = args.coeff<8>();

			auto  Urtn = args.tail<3>();

			auto sinL = sin(L);
			auto cosL = cos(L);
			auto sqp  = sqrt(p)/mu;
			auto w  = 1. + f * cosL + g * sinL;
			auto s2 = 1. + h.Square() + k.Square();


			auto expr = [mu]() {

				auto args = Arguments<13>();

				auto p = args.coeff<0>();
				auto f = args.coeff<1>();
				auto g = args.coeff<2>();
				auto h = args.coeff<3>();
				auto k = args.coeff<4>();

				auto ur = args.coeff<5>();
				auto ut = args.coeff<6>();
				auto un = args.coeff<7>();

				auto sinL = args.coeff<8>();
				auto cosL = args.coeff<9>();
				auto sqp  = args.coeff<10>();
				auto w    = args.coeff<11>();
				auto s2   = args.coeff<12>();

				

				auto pdot = 2. * (p / w) * ut;
				//auto fdot = ur * sinL + ((w + 1) * cosL + f) * (ut / w) - (h * sinL - k * cosL) * (g * un / w);

				//auto fdot = sum(ur * sinL , ((w + 1) * cosL + f) * (ut / w), -1*(h * sinL - k * cosL) * (g * un / w));

				auto fdot = (ur * sinL + ((w + 1) * cosL + f) * (ut / w) - (h * sinL - k * cosL) * (g * un / w));


				auto gdot = -1 * ur * cosL + ((w + 1) * sinL + g) * (ut / w) + (h * sinL - k * cosL) * (g * un / w);
				auto hkdot = stack( cosL, sinL ) *((s2 * un / w) / 2.0);
				auto Ldot = mu * (w / p) * (w / p) + (1.0 / w) * (h * sinL - k * cosL) * un;
				auto ode = stack( pdot, fdot, gdot, hkdot, Ldot )*sqp;

				return ode;
			};

			auto ode= expr().eval(stack(args.head<5>(), Urtn,stack(sinL,cosL,sqp,w,s2) ));
			return ode;

		}
	};





}