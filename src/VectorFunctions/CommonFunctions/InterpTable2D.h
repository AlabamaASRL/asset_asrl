#pragma once
#include "VectorFunction.h"


namespace ASSET {

	struct InterpTable2D {

		Eigen::VectorXd xs;
		Eigen::VectorXd ys;

		using MatType = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

		MatType zs;
		MatType dzxs;
		MatType dzys;
		MatType dzys_dxs;


		bool cubic = false;
		bool xeven = true;
		bool yeven = true;
		int xsize;
		double xtotal;
		int ysize;
		double ytotal;


		InterpTable2D(){}

		InterpTable2D(const Eigen::VectorXd & Xs, const Eigen::VectorXd& Ys,
			const MatType & Zs,bool cubic):xs(Xs),ys(Ys),zs(Zs) {
			set_data(Xs, Ys, Zs, cubic);
		}


		void set_data(const Eigen::VectorXd& Xs, const Eigen::VectorXd& Ys,
			const MatType& Zs, bool cubic) {

			this->cubic = cubic;
			this->xs = Xs;
			this->ys = Ys;
			this->zs = Zs;



			xsize = xs.size();
			ysize = ys.size();

			if (xsize < 3) {
				throw std::invalid_argument("X coordinates must be larger than 2");
			}
			if (ysize < 3) {
				throw std::invalid_argument("Y  coordinates must be larger than 2");
			}

			if (xsize != zs.cols()) {
				throw std::invalid_argument("X coordinates must match cols in Z matrix");
			}
			if (ysize != zs.rows()) {
				throw std::invalid_argument("Y coordinates must match rows in Z matrix");
			}

			for (int i = 0; i < xs.size() - 1; i++) {
				if (xs[i + 1] < xs[i]) {
					throw std::invalid_argument("X Coordinates must be in ascending order");
				}
			}
			for (int i = 0; i < ys.size() - 1; i++) {
				if (ys[i + 1] < ys[i]) {
					throw std::invalid_argument("Y Coordinates must be in ascending order");
				}
			}

			xtotal = xs[xsize - 1] - xs[0];
			ytotal = ys[ysize - 1] - ys[0];


			Eigen::VectorXd testx;
			testx.setLinSpaced(xs.size(), xs[0], xs[xs.size() - 1]);
			Eigen::VectorXd testy;
			testy.setLinSpaced(ys.size(), ys[0], ys[ys.size() - 1]);

			double xerr = (xs - testx).lpNorm<Eigen::Infinity>();
			double yerr = (ys - testy).lpNorm<Eigen::Infinity>();

			if (xerr > abs(xtotal) * 1.0e-12) {
				this->xeven = false;
			}
			if (yerr > abs(ytotal) * 1.0e-12) {
				this->yeven = false;
			}

			if (cubic)calc_derivs();

		}


		void calc_derivs() {
			dzxs.resize(ysize, xsize);
			dzys.resize(ysize, xsize);
			dzys_dxs.resize(ysize, xsize);
			Eigen::Matrix<double, -1, -1, Eigen::RowMajor> dzys_dxs_tmp(ysize, xsize);


			Eigen::Matrix3d stens;
			stens << 1, 1, 1,
				-1, 0, 1,
				-1, 0, 1;

			Eigen::Vector3d rhs;
			rhs << 0, 1, 0;

			Eigen::Vector3d coeffs;

			for (int i = 0; i < ysize; i++) {
				for (int j = 0; j < xsize; j++) {
					if (i == 0) {
						double ystep1 = ys[i + 1] - ys[i];
						dzys(i, j) = (zs(i + 1, j) - zs(i, j)) / ystep1;
					}
					else if (i == ysize - 1) {
						double ystep1 = ys[i] - ys[i-1];
						dzys(i, j) = (zs(i, j) - zs(i - 1, j)) / ystep1;
					}
					else {
						double ystep1 = ys[i + 1] - ys[i];
						
						if (this->yeven) {
							dzys(i, j) = (zs(i + 1, j) - zs(i - 1, j)) / (2 * ystep1);
						}
						else {
							double ystep2 = ys[i] - ys[i - 1];
							double yn = -1.0 * ystep2 / ystep1;

							stens(1, 0) = yn;
							stens(2, 0) = yn * yn;
							coeffs = stens.inverse() * rhs;

							dzys(i, j) = (coeffs[0] * zs(i - 1, j) + coeffs[1] * zs(i, j) + coeffs[2] * zs(i + 1, j)) / ystep1;

						}
					}
				}
			}

			for (int i = 0; i < ysize; i++) {
				for (int j = 0; j < xsize; j++) {
					if (j == 0) {
						double xstep1 = xs[j + 1] - xs[j];

						dzxs(i, j) = (zs(i, j + 1) - zs(i, j)) / xstep1;
						dzys_dxs(i, j) = (dzys(i, j + 1) - dzys(i, j)) / xstep1;
					}
					else if (j == xsize - 1) {
						double xstep1 = xs[j] - xs[j-1];

						dzxs(i, j) = (zs(i, j) - zs(i, j - 1)) / xstep1;
						dzys_dxs(i, j) = (dzys(i, j) - dzys(i, j - 1)) / xstep1;
					}
					else {
						double xstep1 = xs[j + 1] - xs[j];

						if (this->xeven) {
							dzxs(i, j) = (zs(i, j + 1) - zs(i, j - 1)) / (2 * xstep1);
							dzys_dxs(i, j) = (dzys(i, j + 1) - dzys(i, j - 1)) / (2 * xstep1);
						}
						else {

							double xstep2 = xs[j] - xs[j - 1];
							double xn = -1.0 * xstep2 / xstep1;

							stens(1, 0) = xn;
							stens(2, 0) = xn * xn;
							coeffs = stens.inverse() * rhs;

							dzxs(i, j)    = (coeffs[0] * zs(i, j - 1)    + coeffs[1] * zs(i, j )   + coeffs[2] * zs(i, j + 1)) / (xstep1);
							dzys_dxs(i, j) = (coeffs[0] * dzys(i, j - 1) + coeffs[1] * dzys(i, j)  + coeffs[2] * dzys(i, j + 1)) / (xstep1);

						}
					}
				}
			}
		}
		

		int find_elem(const Eigen::VectorXd& vs, double v) const {
			int center = int(vs.size() / 2);
			int shift = (vs[center] > v) ? 0 : center;
			auto it = std::upper_bound(vs.begin() + shift, vs.end(), v);
			int elem = int(it - vs.begin()) - 1;
			return elem;
		}

		std::tuple<int, int> get_xyelems(double x, double y) const {
			int xelem, yelem;

			if (this->xeven) {
				double xlocal = x - xs[0];
				double xstep = xs[1] - xs[0];
				xelem = std::min(int(xlocal / xstep), this->xsize - 2);
			}
			else {
				xelem = this->find_elem(this->xs, x);
			}

			if (this->yeven) {
				double ylocal = y - this->ys[0];
				double ystep = this->ys[1] - this->ys[0];
				yelem = std::min(int(ylocal / ystep), this->ysize - 2);
			}
			else {
				yelem = this->find_elem(this->ys, y);

			}

			xelem = std::min(xelem, this->xsize - 2);
			yelem = std::min(yelem, this->ysize - 2);

			xelem = std::max(xelem, 0);
			yelem = std::max(yelem, 0);

			return std::tuple{ xelem,yelem };
		}

		Eigen::Matrix4<double> get_amatrix(int xelem, int yelem) const {

			double xstep = xs[xelem + 1] - xs[xelem];
			double ystep = ys[yelem + 1] - ys[yelem];

			Eigen::Matrix4<double> a;
			Eigen::Matrix4<double> L;
			L << 1,  0,  0,  0,
				 0,  0,  1,  0,
				-3,  3, -2, -1,
				 2, -2,  1,  1;
			Eigen::Matrix4<double> R;

			R << 1, 0, -3, 2,
				 0, 0,  3, -2,
				 0, 1, -2, 1,
				 0, 0, -1, 1;

			Eigen::Matrix4<double> Z;

			double z00 = zs(yelem, xelem);
			double z10 = zs(yelem, xelem+1);
			double z01 = zs(yelem+1, xelem);
			double z11 = zs(yelem+1, xelem+1);

			double dz00_x = dzxs(yelem, xelem)*xstep;
			double dz10_x = dzxs(yelem, xelem + 1) * xstep;
			double dz01_x = dzxs(yelem + 1, xelem) * xstep;
			double dz11_x = dzxs(yelem + 1, xelem + 1) * xstep;


			double dz00_y = dzys(yelem, xelem) * ystep;
			double dz10_y = dzys(yelem, xelem + 1) * ystep;
			double dz01_y = dzys(yelem + 1, xelem) * ystep;
			double dz11_y = dzys(yelem + 1, xelem + 1) * ystep;


			double dz00_xy = dzys_dxs(yelem, xelem) * xstep * ystep;
			double dz10_xy = dzys_dxs(yelem, xelem + 1) * xstep * ystep;
			double dz01_xy = dzys_dxs(yelem + 1, xelem) * xstep * ystep;
			double dz11_xy = dzys_dxs(yelem + 1, xelem + 1) * xstep * ystep;


			Z << z00, z01, dz00_y, dz01_y,
				z10, z11, dz10_y, dz11_y,
				dz00_x, dz01_x, dz00_xy, dz01_xy,
				dz10_x, dz11_x, dz10_xy, dz11_xy;

			a = L * Z * R;
			return a;
		}

		void interp_impl(double x, double y,
			int deriv,
			double& z, 
			Eigen::Vector2<double>& dzxy, 
			Eigen::Matrix2<double>& d2zxy) const {

			auto [xelem, yelem] = get_xyelems(x, y);

			double xstep = xs[xelem + 1] - xs[xelem];
			double ystep = ys[yelem + 1] - ys[yelem];

			double xf = (x - xs[xelem]) / xstep;
			double yf = (y - ys[yelem]) / ystep;

			if (cubic) {

				double yf2 = yf * yf;
				double yf3 = yf2 * yf;
				double xf2 = xf * xf;
				double xf3 = xf2 * xf;

				Eigen::Matrix4<double> amat = get_amatrix(xelem, yelem);
				Vector4<double> xvec;
				xvec << 1, xf, xf2, xf3;

				Vector4<double> yvec;
				yvec << 1, yf, yf2, yf3;

				z = xvec.transpose() * amat * yvec;

				if (deriv >0) {

					Vector4<double> dxvec;
					dxvec << 0, 1/xstep, 2*xf/xstep, 3*xf2/xstep;

					Vector4<double> dyvec;
					dyvec << 0, 1 / ystep, 2 * yf / ystep, 3 * yf2 / ystep;

					dzxy[0] = dxvec.transpose() * amat * yvec;
					dzxy[1] = xvec.transpose() * amat * dyvec;

					if (deriv >1) {

						Vector4<double> d2xvec;
						d2xvec << 0, 0, 2 / (xstep*xstep), 6 * xf / (xstep * xstep);

						Vector4<double> d2yvec;
						d2yvec << 0, 0, 2 / (ystep * ystep), 6 * yf / (ystep * ystep);
						
						d2zxy(0, 0) = d2xvec.transpose() * amat * yvec;
						d2zxy(1, 0) = dxvec.transpose() * amat * dyvec;
						d2zxy(0, 1) = d2zxy(1, 0);
						d2zxy(1, 1) = xvec.transpose() * amat * d2yvec;

					}
				}

			}
			else {
				// Linear 
				double zx0y0 = zs(yelem, xelem);
				double zx1y0 = zs(yelem, xelem + 1);
				double zy0m = zx0y0 * (1 - xf) + zx1y0 * xf;

				double zx0y1 = zs(yelem + 1, xelem);
				double zx1y1 = zs(yelem + 1, xelem + 1);
				double zy1m = zx0y1 * (1 - xf) + zx1y1 * xf;

				z = zy0m * (1 - yf) + zy1m * (yf);

				if (deriv > 0) {
					dzxy[0] = ((zx1y0 - zx0y0) * (1 - yf) + (zx1y1 - zx0y1) * (yf))/xstep;
					dzxy[1] = (zy1m - zy0m) / ystep;
					if (deriv > 1) {
						d2zxy.setZero();
						d2zxy(1, 0) = ((zx1y0 - zx0y0) * (-1) + (zx1y1 - zx0y1) * (1)) / (xstep * ystep);
						d2zxy(0, 1) = d2zxy(1, 0);
					}
				}

			}

		}

		double interp(double x, double y) const {

			double z;
			Eigen::Vector2<double> dzxy;
			Eigen::Matrix2<double> d2zxy;
			interp_impl(x, y, 0, z, dzxy, d2zxy);

			return z; 

		}

		MatType interp(const MatType& xs, const MatType & ys) const{
			MatType zs(xs.rows(), xs.cols());

			for (int i = 0; i < xs.rows(); i++) {
				for (int j = 0; j < xs.cols(); j++) {
					zs(i, j) = interp(xs(i, j), ys(i, j));
				}
			}
			return zs;
		}

		std::tuple<double,Eigen::Vector2<double>> interp_deriv1(double x, double y) const {
			double z;
			Eigen::Vector2<double> dzxy;
			Eigen::Matrix2<double> d2zxy;

			interp_impl(x, y, 1, z, dzxy, d2zxy);

			return std::tuple{ z, dzxy }; // intellisense is confused pls ignore
		}

		std::tuple<double, Eigen::Vector2<double>, Eigen::Matrix2<double>> interp_deriv2(double x, double y) const {
			double z;
			Eigen::Vector2<double> dzxy;
			Eigen::Matrix2<double> d2zxy;

			interp_impl(x, y, 2, z, dzxy, d2zxy);

			return std::tuple{ z, dzxy,d2zxy }; // intellisense is confused pls ignore
		}


	};


	struct InterpFunction2D : VectorFunction<InterpFunction2D, 2, 1, Analytic, Analytic> {
		using Base = VectorFunction<InterpFunction2D, 2, 1, Analytic, Analytic>;
		DENSE_FUNCTION_BASE_TYPES(Base);

		std::shared_ptr<InterpTable2D> tab;


		InterpFunction2D() {}
		InterpFunction2D(std::shared_ptr<InterpTable2D> tab) :tab(tab) {
			this->setIORows(1, 1);
		}


		template <class InType, class OutType>
		inline void compute_impl(ConstVectorBaseRef<InType> x,
			ConstVectorBaseRef<OutType> fx_) const {
			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();
			fx[0] = this->tab->interp(x[0], x[1]);

		}
		template <class InType, class OutType, class JacType>
		inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
			ConstVectorBaseRef<OutType> fx_,
			ConstMatrixBaseRef<JacType> jx_) const {
			typedef typename InType::Scalar Scalar;
			VectorBaseRef<OutType> fx = fx_.const_cast_derived();
			MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

			auto [z, dzdx] = this->tab->interp_deriv1(x[0], x[1]);
			fx[0] = z;
			jx = dzdx.transpose();

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

			auto [z, dzdx, d2zdx] = this->tab->interp_deriv2(x[0], x[1]);
			fx[0] = z;
			jx = dzdx.transpose();
			adjgrad = adjvars[0] * dzdx;
			adjhess = adjvars[0] * d2zdx;

		}




	};

	


	static void InterpTable2DBuild(py::module& m) {
		using MatType = InterpTable2D::MatType;
		auto obj = py::class_<InterpTable2D, std::shared_ptr<InterpTable2D>>(m, "InterpTable2D");

		obj.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>&, bool >());

		obj.def("interp", py::overload_cast<double, double>(&InterpTable2D::interp, py::const_));
		obj.def("interp", py::overload_cast<const MatType&, const MatType&>(&InterpTable2D::interp, py::const_));

		obj.def("interp_deriv1", &InterpTable2D::interp_deriv1);
		obj.def("interp_deriv2", &InterpTable2D::interp_deriv2);

		obj.def("find_elem", &InterpTable2D::find_elem);


		obj.def("sf", [](const InterpTable2D& self) {
			return GenericFunction<-1, 1>(InterpFunction2D(std::make_shared<InterpTable2D>(self)));
			});
		obj.def("vf", [](const InterpTable2D& self) {
			return GenericFunction<-1, -1>(InterpFunction2D(std::make_shared<InterpTable2D>(self)));
			});




	}


}