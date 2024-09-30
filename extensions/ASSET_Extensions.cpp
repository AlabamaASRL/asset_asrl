#include "ASSET_Extensions.h"

namespace ASSET {


	struct CR3BPAD : VectorFunction<CR3BPAD, 7, 6, AutodiffFwd, AutodiffFwd> {
		using Base = VectorFunction<CR3BPAD, 7, 6, AutodiffFwd, AutodiffFwd>;
		DENSE_FUNCTION_BASE_TYPES(Base)

		double mu = 0.0123;

		CR3BPAD(double mu) :mu(mu) {}

		template<class InType, class OutType>
		inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {

			
			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();

			Vector3<Scalar> X = x. template head<3>();
			Vector3<Scalar> V = x. template segment<3>(3);

			Vector3<Scalar> p1loc;
			p1loc[0] = -mu;

			Vector3<Scalar> p2loc;
			p2loc[0] = 1.0 - mu;

			Vector3<Scalar> dvec = X - p1loc;
			Vector3<Scalar> rvec = X - p2loc;

			Scalar d = dvec.norm();
			Scalar r = rvec.norm();

			fx.template head<3>() = V;
			fx.template segment<3>(3) = -(1.0 - mu) * dvec / (d * d * d) - mu * rvec / (r * r * r);
			fx[3] += 2.0 * V[1] + X[0];
			fx[4] += -2.0 * V[0] + X[1];

		}

		static void Build(py::module& m, const char* name) {
			auto obj = py::class_<CR3BPAD>(m, name).def(py::init<double>());
			Base::DenseBaseBuild(obj);
		}

	};


	struct ModifiedDynamicsAD : VectorFunction<ModifiedDynamicsAD, 9, 6, AutodiffFwd, AutodiffFwd> {
		using Base = VectorFunction<ModifiedDynamicsAD, 9, 6, AutodiffFwd, AutodiffFwd>;
		DENSE_FUNCTION_BASE_TYPES(Base)

		double mu = 1.00;
		double sqm = 1.0;

		ModifiedDynamicsAD(double mu) :mu(mu),sqm(sqrt(mu)) {}

		template<class InType, class OutType>
		inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {


			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();

			Scalar x0 = x[0];
			Scalar x1 = x[1];
			Scalar x2 = x[2];
			Scalar x3 = x[3];
			Scalar x4 = x[4];
			Scalar x5 = x[5];
			Scalar x6 = x[6];
			Scalar x7 = x[7];
			Scalar x8 = x[8];

			Scalar sqx0 = sqrt(x0);

			Scalar x9 = Scalar(1.0 / sqm);
			Scalar x10 = cos(x5);
			Scalar x11 = sin(x5);
			Scalar x12 = x1 * x10 + x11 * x2;
			Scalar x13 = x12 + 1.0;
			Scalar x14 = 1.0 / x13;
			Scalar x15 = x14 * x7;
			Scalar x16 = x10 * x4;
			Scalar x17 = x11 * x3;
			Scalar x18 = x14 * x8;
			Scalar x19 = x12 + 2.0;
			Scalar x20 = sqx0 * x9;
			Scalar x21 = x18 * (-x16 + x17);
			Scalar x22 = 0.5 * x18 * x20 * ((x3 * x3) + (x4 * x4) + 1.0);

			fx[0] = 2.0 * (x0 * sqx0) * x15 * x9;
			fx[1] = x20 * (x11 * x6 + x15 * (x1 + x10 * x19) + x18 * x2 * (x16 - x17));
			fx[2] = x20 * (x1 * x21 - x10 * x6 + x15 * (x11 * x19 + x2));
			fx[3] = x10 * x22;
			fx[4] = x11 * x22;
			fx[5] = x20 * (mu * x13 * x13 / (x0 * x0) + 1.0 * x21);

		}

		static void Build(py::module& m, const char* name) {
			auto obj = py::class_<ModifiedDynamicsAD>(m, name).def(py::init<double>());
			Base::DenseBaseBuild(obj);
		}

	};




}








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

	reg.Build_Register<CR3BPAD>(extmod, "cpp_cr3bp_ad");
	reg.Build_Register<ModifiedDynamicsAD>(extmod, "ModifiedDynamicsAD");

	
}
