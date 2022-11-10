#pragma once
#include "VectorFunction.h"


namespace ASSET {

	struct InterpTable1D {

		using MatType = Eigen::Matrix<double, -1, -1>;

		Eigen::VectorXd ts;
		MatType vs;
		MatType dvs_dts;

		bool cubic = false;
		bool teven = true;
		int tsize;
		double ttotal;
		int vlen;

		InterpTable1D(){}
		InterpTable1D(const Eigen::VectorXd& Ts, const MatType& Vs, bool cubic) {
			set_data(Ts, Vs, cubic);
		}
		InterpTable1D(const Eigen::VectorXd& Ts, const Eigen::VectorXd& Vs, bool cubic) {
			MatType Vstmp = Vs.transpose();
			set_data(Ts, Vstmp, cubic);
		}

		void set_data(const Eigen::VectorXd& Ts, const MatType& Vs, bool cubic) {

			this->ts = Ts;
			this->vs = Vs;
			this->cubic = cubic;

			tsize = ts.size();
			vlen  = vs.rows();
			ttotal = ts[tsize - 1] - ts[0];

			if (tsize < 3) {
				throw std::invalid_argument("t coordinates must be larger than 2");
			}
			if (tsize != vs.cols()) {
				throw std::invalid_argument("t coordinates must match rows in V matrix");
			}
			for (int i = 0; i < ts.size() - 1; i++) {
				if (ts[i + 1] < ts[i]) {
					throw std::invalid_argument("t Coordinates must be in ascending order");
				}
			}

			Eigen::VectorXd testt;
			testt.setLinSpaced(ts.size(), ts[0], ts[ts.size() - 1]);
			
			double terr = (ts - testt).lpNorm<Eigen::Infinity>();

			if (terr > abs(ttotal) * 1.0e-12) {
				this->teven = false;
			}


			if (cubic) calc_derivs();

		}

		void calc_derivs() {
			
			dvs_dts.resize(vlen, tsize);


			Eigen::Matrix3d stens;
			stens << 1, 1, 1,
				-1, 0, 1,
				-1, 0, 1;

			Eigen::Vector3d rhs;
			rhs << 0, 1, 0;

			Eigen::Vector3d coeffs;

			for (int i = 0; i < tsize; i++) {
				
					if (i == 0) {
						double tstep1 = ts[i + 1] - ts[i];

						if (this->teven) {
							dvs_dts.col(i) = (-.5*vs.col(i + 2)+2.0*vs.col(i + 1) - 1.5*vs.col(i)) / tstep1;

						}
						else {
							dvs_dts.col(i) = (vs.col(i + 1) - vs.col(i)) / tstep1;
						}
					}
					else if (i == tsize - 1) {
						double tstep1 = ts[i] - ts[i - 1];
						if (this->teven) {
							dvs_dts.col(i) = (1.5 * vs.col(i) - 2.0 * vs.col(i - 1) + .5 * vs.col(i - 2)) / tstep1;
						}
						else {
							dvs_dts.col(i) = (vs.col(i) -  vs.col(i - 1) ) / tstep1;
						}
						
					}
					else {
						double tstep1 = ts[i + 1] - ts[i];

						if (this->teven) {
							dvs_dts.col(i) = (vs.col(i + 1) - vs.col(i - 1)) / (2 * tstep1);
						}
						else {
							double tstep2 = ts[i] - ts[i - 1];
							double tn = -1.0 * tstep2 / tstep1;

							stens(1, 0) = tn;
							stens(2, 0) = tn * tn;
							coeffs = stens.inverse() * rhs;

							dvs_dts.col(i) = (coeffs[0] * vs.col(i - 1) + coeffs[1] * vs.col(i) + coeffs[2] * vs.col(i + 1)) / tstep1;

						}
					}
				
			}

		}

		int get_telem(double t) const {
			int telem;
			if (this->teven) {
				double tlocal = t - ts[0];
				double tstep = ts[1] - ts[0];
				telem = std::min(int(tlocal / tstep), this->tsize - 2);
			}
			else {
				int center = int(ts.size() / 2);
				int shift = (ts[center] > t) ? 0 : center;
				auto it = std::upper_bound(ts.cbegin(), ts.cend(), t);
				telem = int(it - ts.begin()) - 1;
			}


			double tstep = ts[telem + 1] - ts[telem];
			double tnd = (t - ts[telem]) / tstep;

			telem = std::min(telem, this->tsize - 2);
			telem = std::max(telem, 0);
			return telem;
		}

		template<class VType>
		void interp_impl(double t, int deriv, VType& v, VType& dv_dt, VType& dv2_dt2) const {

			double telem = this->get_telem(t);
			double tstep = ts[telem + 1] - ts[telem];
			double tnd   = (t - ts[telem]) / tstep;

			if (cubic) {

				double tnd2 = tnd * tnd;
				double tnd3 = tnd2 * tnd;

				double p0 = (2.0 * tnd3 - 3.0 * tnd2 + 1.0);
				double m0 = (tnd3 - 2.0 * tnd2 + tnd) * tstep;
				double p1 = (-2.0 * tnd3 + 3.0 * tnd2);
				double m1 = (tnd3 - tnd2) * tstep;


				v =   vs.col(telem) * p0      + vs.col(telem + 1) * p1
					+ dvs_dts.col(telem) * m0 + dvs_dts.col(telem + 1) * m1;


				if (deriv > 0) {

					double p0_dt = (6.0 * tnd2 - 6.0 * tnd) / tstep;
					double m0_dt = (3.0 * tnd2 - 4.0 * tnd + 1.0);
					double p1_dt = (-6.0 * tnd2 + 6.0 * tnd) / tstep;
					double m1_dt = (3.0 * tnd2 - 2.0 * tnd);


					dv_dt = vs.col(telem) * p0_dt + vs.col(telem + 1) * p1_dt
						+ dvs_dts.col(telem) * m0_dt + dvs_dts.col(telem + 1) * m1_dt;


					if (deriv > 1) {

						double p0_dt2 = (12.0 * tnd - 6.0) / (tstep * tstep);
						double m0_dt2 = (6.0 * tnd - 4.0) / tstep;
						double p1_dt2 = (-12.0 * tnd + 6.0) / (tstep * tstep);
						double m1_dt2 = (6.0 * tnd - 2.0) / tstep;

						dv2_dt2 = vs.col(telem) * p0_dt2 + vs.col(telem + 1) * p1_dt2
							+ dvs_dts.col(telem) * m0_dt2 + dvs_dts.col(telem + 1) * m1_dt2;


					}
				}
				
			}
			else {
				v = vs.col(telem) * (1 - tnd) + vs.col(telem + 1) * tnd;
				if (deriv > 0) {
					dv_dt = (vs.col(telem + 1) - vs.col(telem)) / tstep;
					if (deriv > 1) {
						// Zero
					}
				}
			}
		}

		Eigen::VectorXd interp(double t) const {

			Eigen::VectorXd v;
			v.resize(vlen);
			interp_impl(t, 0, v, v, v);
			return v;

		}

		std::tuple<Eigen::VectorXd, Eigen::VectorXd > interp_deriv1(double t) const {

			Eigen::VectorXd v;
			v.resize(vlen);
			Eigen::VectorXd dv_dt;
			dv_dt.resize(vlen);

			interp_impl(t, 1, v, dv_dt, dv_dt);

			return std::tuple{ v,dv_dt };

		}

		std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd > interp_deriv2(double t) const {

			Eigen::VectorXd v;
			v.resize(vlen);
			Eigen::VectorXd dv_dt;
			dv_dt.resize(vlen);
			Eigen::VectorXd dv2_dt2;
			dv2_dt2.resize(vlen);


			interp_impl(t, 2, v, dv_dt, dv2_dt2);

			return std::tuple{ v,dv_dt, dv2_dt2 };

		}


	};

	template<int ORR>
	struct InterpFunction1D : VectorFunction<InterpFunction1D<ORR>, 1, ORR, Analytic, Analytic> {
		using Base = VectorFunction<InterpFunction1D<ORR>, 1, ORR, Analytic, Analytic>;
		DENSE_FUNCTION_BASE_TYPES(Base);

		std::shared_ptr<InterpTable1D> tab;


		InterpFunction1D() {}
		InterpFunction1D(std::shared_ptr<InterpTable1D> tab) :tab(tab) {
			this->setIORows(1, tab->vlen);
		}


		template <class InType, class OutType>
		inline void compute_impl(ConstVectorBaseRef<InType> x,
			ConstVectorBaseRef<OutType> fx_) const {
			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();

			auto Impl = [&](auto& v) {
				this->tab->interp_impl(x[0], 0, v, v, v);
				fx = v;
			};

			ASSET::MemoryManager::allocate_run(this->ORows(), Impl,
				TempSpec<Output<Scalar>>(this->ORows(), 1));

		}
		template <class InType, class OutType, class JacType>
		inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
			ConstVectorBaseRef<OutType> fx_,
			ConstMatrixBaseRef<JacType> jx_) const {
			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();
			MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

			auto Impl = [&](auto& v, auto& dv_dt) {
				this->tab->interp_impl(x[0], 1, v, dv_dt, v);
				fx = v;
				jx = dv_dt;
			};

			ASSET::MemoryManager::allocate_run(this->ORows(), Impl,
				TempSpec<Output<Scalar>>(this->ORows(), 1),
				TempSpec<Output<Scalar>>(this->ORows(), 1)
			);

		}
		template <class InType, class OutType, class JacType, class AdjGradType,
			class AdjHessType, class AdjVarType>
			inline void compute_jacobian_adjointgradient_adjointhessian_impl(
				ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
				ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
				ConstMatrixBaseRef<AdjHessType> adjhess_,
				ConstVectorBaseRef<AdjVarType> adjvars) const {
			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();
			MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
			VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
			MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

			auto Impl = [&](auto& v, auto& dv_dt, auto& dv2_dt2) {
				this->tab->interp_impl(x[0], 2, v, dv_dt, dv2_dt2);
				fx = v;
				jx = dv_dt;
				adjgrad[0] = dv_dt.dot(adjvars);
				adjhess(0,0) = dv2_dt2.dot(adjvars);
			};

			ASSET::MemoryManager::allocate_run(this->ORows(), Impl,
				TempSpec<Output<Scalar>>(this->ORows(),1),
				TempSpec<Output<Scalar>>(this->ORows(), 1),
				TempSpec<Output<Scalar>>(this->ORows(), 1)
			);

		}
	};


	static void InterpTable1DBuild(py::module& m) {

		using MatType = InterpTable1D::MatType;
		auto obj = py::class_<InterpTable1D, std::shared_ptr<InterpTable1D>>(m, "InterpTable1D");

		obj.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, bool >());

		obj.def(py::init<const Eigen::VectorXd& , const MatType& , bool >());

		obj.def("interp", py::overload_cast<double>(&InterpTable1D::interp, py::const_));

		obj.def("__call__", py::overload_cast<double>(&InterpTable1D::interp, py::const_),py::is_operator());

		obj.def("__call__", [](const InterpTable1D& self, const GenericFunction<-1, 1> & t) {
			py::object pyfun;
				if (self.vlen == 1) {
					auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(std::make_shared<InterpTable1D>(self)));
					pyfun = py::cast(f);
				} 
				else {
					auto f = GenericFunction<-1, -1>(InterpFunction1D<1>(std::make_shared<InterpTable1D>(self)));
					pyfun = py::cast(f);
				}
				return pyfun;
			});


		obj.def("interp_deriv1", &InterpTable1D::interp_deriv1);
		obj.def("interp_deriv2", &InterpTable1D::interp_deriv2);

		

		
		obj.def("sf", [](const InterpTable1D& self) {
			if (self.vlen != 1) {
				throw std::invalid_argument("InterpTable1D storing Vector data cannot be converted to Scalar Function.");
			}
			return GenericFunction<-1, 1>(InterpFunction1D<1>(std::make_shared<InterpTable1D>(self)));
			});
		obj.def("vf", [](const InterpTable1D& self) {
			return GenericFunction<-1, -1>(InterpFunction1D<-1>(std::make_shared<InterpTable1D>(self)));
			});




	}




}
