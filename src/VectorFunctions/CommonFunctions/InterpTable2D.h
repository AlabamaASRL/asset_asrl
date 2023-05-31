#pragma once
#include "Utils/Timer.h"
#include "VectorFunction.h"

namespace ASSET {

  struct InterpTable2D {

    enum class InterpType {
      cubic_interp,
      linear_interp
    };

    Eigen::VectorXd xs;
    Eigen::VectorXd ys;

    using MatType = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

    MatType zs;
    MatType dzxs;
    MatType dzys;
    MatType dzys_dxs;

    Eigen::Matrix<Eigen::Array4d, -1, -1, Eigen::RowMajor> all_dat;


    bool WarnOutOfBounds = true;
    bool ThrowOutOfBounds = false;

    InterpType interp_kind = InterpType::cubic_interp;
    bool xeven = true;
    bool yeven = true;
    int xsize;
    double xtotal;
    int ysize;
    double ytotal;

    InterpTable2D() {
    }

    InterpTable2D(const Eigen::VectorXd& Xs, const Eigen::VectorXd& Ys, const MatType& Zs, std::string kind) {
      set_data(Xs, Ys, Zs, kind);
    }


    void set_data(const Eigen::VectorXd& Xs, const Eigen::VectorXd& Ys, const MatType& Zs, std::string kind) {


      if (kind == "cubic" || kind == "Cubic") {
        this->interp_kind = InterpType::cubic_interp;
      } else if (kind == "linear" || kind == "Linear") {
        this->interp_kind = InterpType::linear_interp;
      } else {
        throw std::invalid_argument("Unrecognized interpolation type");
      }

      this->xs = Xs;
      this->ys = Ys;
      this->zs = Zs;


      xsize = xs.size();
      ysize = ys.size();

      if (xsize < 5) {
        throw std::invalid_argument("X coordinates must be larger than 4");
      }
      if (ysize < 5) {
        throw std::invalid_argument("Y  coordinates must be larger than 4");
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

      if (this->interp_kind == InterpType::cubic_interp)
        calc_derivs();
    }


    void calc_derivs() {
      dzxs.resize(ysize, xsize);
      dzys.resize(ysize, xsize);
      dzys_dxs.resize(ysize, xsize);
      all_dat.resize(ysize, xsize);


      Eigen::Matrix<double, 5, 5> stens;
      stens.row(0).setOnes();
      Eigen::Matrix<double, 5, 1> rhs;
      rhs << 0, 1, 0, 0, 0;
      Eigen::Matrix<double, 5, 1> times;
      Eigen::Matrix<double, 5, 1> coeffs;


      bool hitcent = false;
      for (int i = 0; i < this->ysize; i++) {
        int start = 0;
        bool recalc = true;
        if (i + 2 <= this->ysize - 1 && i - 2 >= 0) {
          // central difference
          if (this->yeven && hitcent) {
            recalc = false;
          }
          hitcent = true;
          start = i - 2;
        } else if (i < this->ysize - 1 - i) {
          // forward difference
          start = 0;
        } else {
          // backward difference
          start = this->ysize - 5;
        }
        int stepdir = (i < this->ysize - 1) ? 1 : -1;
        double yi = this->ys[i];
        double ystep = std::abs(this->ys[i + stepdir] - yi);
        if (recalc) {
          times = this->ys.segment(start, 5);
          times -= Eigen::Matrix<double, 5, 1>::Constant(yi);
          times /= ystep;
          stens.row(1) = times.transpose();
          stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
          stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
          stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
          coeffs = stens.inverse() * rhs;
        }
        dzys.row(i) = (coeffs / ystep).transpose() * this->zs.middleRows(start, 5);
      }


      hitcent = false;
      for (int i = 0; i < this->xsize; i++) {
        int start = 0;
        bool recalc = true;
        if (i + 2 <= this->xsize - 1 && i - 2 >= 0) {
          // central difference
          if (this->xeven && hitcent) {
            recalc = false;
          }
          hitcent = true;
          start = i - 2;
        } else if (i < this->xsize - 1 - i) {
          // forward difference
          start = 0;
        } else {
          // backward difference
          start = this->xsize - 5;
        }
        int stepdir = (i < this->xsize - 1) ? 1 : -1;
        double xi = this->xs[i];
        double xstep = std::abs(this->xs[i + stepdir] - xi);
        if (recalc) {
          times = this->xs.segment(start, 5);
          times -= Eigen::Matrix<double, 5, 1>::Constant(xi);
          times /= xstep;
          stens.row(1) = times.transpose();
          stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
          stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
          stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
          coeffs = stens.inverse() * rhs;
        }

        dzxs.col(i) = this->zs.middleCols(start, 5) * (coeffs / xstep);
        dzys_dxs.col(i) = this->dzys.middleCols(start, 5) * (coeffs / xstep);
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
        double xlocal = x - this->xs[0];
        double xstep = this->xs[1] - this->xs[0];
        xelem = std::min(int(xlocal / xstep), this->xsize - 2);
      } else {
        xelem = this->find_elem(this->xs, x);
      }

      if (this->yeven) {
        double ylocal = y - this->ys[0];
        double ystep = this->ys[1] - this->ys[0];
        yelem = std::min(int(ylocal / ystep), this->ysize - 2);
      } else {
        yelem = this->find_elem(this->ys, y);
      }

      xelem = std::min(xelem, this->xsize - 2);
      yelem = std::min(yelem, this->ysize - 2);

      xelem = std::max(xelem, 0);
      yelem = std::max(yelem, 0);

      return std::tuple {xelem, yelem};
    }

    Eigen::Matrix4<double> get_amatrix(int xelem, int yelem) const {

      double xstep = xs[xelem + 1] - xs[xelem];
      double ystep = ys[yelem + 1] - ys[yelem];

      Eigen::Matrix4<double> a;
      Eigen::Matrix4<double> L;
      L << 1, 0, 0, 0, 0, 0, 1, 0, -3, 3, -2, -1, 2, -2, 1, 1;
      Eigen::Matrix4<double> R;

      R << 1, 0, -3, 2, 0, 0, 3, -2, 0, 1, -2, 1, 0, 0, -1, 1;

      Eigen::Matrix4<double> Z;


      double z00 = zs(yelem, xelem);
      double z10 = zs(yelem, xelem + 1);
      double z01 = zs(yelem + 1, xelem);
      double z11 = zs(yelem + 1, xelem + 1);

      double dz00_x = dzxs(yelem, xelem) * xstep;
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


      Z << z00, z01, dz00_y, dz01_y, z10, z11, dz10_y, dz11_y, dz00_x, dz01_x, dz00_xy, dz01_xy, dz10_x,
          dz11_x, dz10_xy, dz11_xy;


      a = L * Z * R;
      return a;
    }

    void interp_impl(double x,
                     double y,
                     int deriv,
                     double& z,
                     Eigen::Vector2<double>& dzxy,
                     Eigen::Matrix2<double>& d2zxy) const {

      if (WarnOutOfBounds || ThrowOutOfBounds) {
        double xeps = std::numeric_limits<double>::epsilon() * xtotal;
        if (x < (xs[0] - xeps) || x > (xs[xs.size() - 1] + xeps)) {

          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: x coordinate falls outside of InterpTable2D range. Data is being extrapolated!!\n");
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
        double yeps = std::numeric_limits<double>::epsilon() * ytotal;
        if (y < (ys[0] - yeps) || y > (ys[ys.size() - 1]) + yeps) {
          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: y coordinate falls outside of InterpTable2D range. Data is being extrapolated!!\n");
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
      }


      auto [xelem, yelem] = get_xyelems(x, y);

      double xstep = xs[xelem + 1] - xs[xelem];
      double ystep = ys[yelem + 1] - ys[yelem];

      double xf = (x - xs[xelem]) / xstep;
      double yf = (y - ys[yelem]) / ystep;

      if (this->interp_kind == InterpType::cubic_interp) {

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

        if (deriv > 0) {

          Vector4<double> dxvec;
          dxvec << 0, 1 / xstep, 2 * xf / xstep, 3 * xf2 / xstep;

          Vector4<double> dyvec;
          dyvec << 0, 1 / ystep, 2 * yf / ystep, 3 * yf2 / ystep;

          dzxy[0] = dxvec.transpose() * amat * yvec;
          dzxy[1] = xvec.transpose() * amat * dyvec;

          if (deriv > 1) {

            Vector4<double> d2xvec;
            d2xvec << 0, 0, 2 / (xstep * xstep), 6 * xf / (xstep * xstep);

            Vector4<double> d2yvec;
            d2yvec << 0, 0, 2 / (ystep * ystep), 6 * yf / (ystep * ystep);

            d2zxy(0, 0) = d2xvec.transpose() * amat * yvec;
            d2zxy(1, 0) = dxvec.transpose() * amat * dyvec;
            d2zxy(0, 1) = d2zxy(1, 0);
            d2zxy(1, 1) = xvec.transpose() * amat * d2yvec;
          }
        }

      } else {
        // Linear
        double zx0y0 = zs(yelem, xelem);
        double zx1y0 = zs(yelem, xelem + 1);
        double zy0m = zx0y0 * (1 - xf) + zx1y0 * xf;

        double zx0y1 = zs(yelem + 1, xelem);
        double zx1y1 = zs(yelem + 1, xelem + 1);
        double zy1m = zx0y1 * (1 - xf) + zx1y1 * xf;

        z = zy0m * (1 - yf) + zy1m * (yf);

        if (deriv > 0) {
          dzxy[0] = ((zx1y0 - zx0y0) * (1 - yf) + (zx1y1 - zx0y1) * (yf)) / xstep;
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

    MatType interp(const MatType& xs, const MatType& ys) const {
      MatType zs(xs.rows(), xs.cols());

      for (int i = 0; i < xs.rows(); i++) {
        for (int j = 0; j < xs.cols(); j++) {
          zs(i, j) = interp(xs(i, j), ys(i, j));
        }
      }
      return zs;
    }

    std::tuple<double, Eigen::Vector2<double>> interp_deriv1(double x, double y) const {
      double z;
      Eigen::Vector2<double> dzxy;
      Eigen::Matrix2<double> d2zxy;

      interp_impl(x, y, 1, z, dzxy, d2zxy);

      return std::tuple {z, dzxy};  // intellisense is confused pls ignore
    }

    std::tuple<double, Eigen::Vector2<double>, Eigen::Matrix2<double>> interp_deriv2(double x,
                                                                                     double y) const {
      double z;
      Eigen::Vector2<double> dzxy;
      Eigen::Matrix2<double> d2zxy;

      interp_impl(x, y, 2, z, dzxy, d2zxy);

      return std::tuple {z, dzxy, d2zxy};  // intellisense is confused pls ignore
    }
  };


  struct InterpFunction2D : VectorFunction<InterpFunction2D, 2, 1, Analytic, Analytic> {
    using Base = VectorFunction<InterpFunction2D, 2, 1, Analytic, Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    std::shared_ptr<InterpTable2D> tab;


    InterpFunction2D() {
    }
    InterpFunction2D(std::shared_ptr<InterpTable2D> tab) : tab(tab) {
      this->setIORows(2, 1);
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx[0] = this->tab->interp(x[0], x[1]);
    }
    template<class InType, class OutType, class JacType>
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
    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_,
        ConstVectorBaseRef<AdjGradType> adjgrad_,
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

    obj.def(py::init<const Eigen::VectorXd&,
                     const Eigen::VectorXd&,
                     const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>&,
                     std::string>(),
            py::arg("xs"),
            py::arg("ys"),
            py::arg("Z"),
            py::arg("kind") = std::string("cubic"));

    obj.def("interp", py::overload_cast<double, double>(&InterpTable2D::interp, py::const_));
    obj.def("interp", py::overload_cast<const MatType&, const MatType&>(&InterpTable2D::interp, py::const_));

    obj.def_readwrite("WarnOutOfBounds", &InterpTable2D::WarnOutOfBounds);
    obj.def_readwrite("ThrowOutOfBounds", &InterpTable2D::ThrowOutOfBounds);


    obj.def("interp_deriv1", &InterpTable2D::interp_deriv1);
    obj.def("interp_deriv2", &InterpTable2D::interp_deriv2);

    obj.def("find_elem", &InterpTable2D::find_elem);

    obj.def(
        "__call__", py::overload_cast<double, double>(&InterpTable2D::interp, py::const_), py::is_operator());
    obj.def("__call__",
            py::overload_cast<const MatType&, const MatType&>(&InterpTable2D::interp, py::const_),
            py::is_operator());


    obj.def("__call__",
            [](std::shared_ptr<InterpTable2D>& self,
               const GenericFunction<-1, 1>& x,
               const GenericFunction<-1, 1>& y) {
              return GenericFunction<-1, 1>(
                  InterpFunction2D(self).eval(stack(x, y)));
            });

    obj.def("__call__",
            [](std::shared_ptr<InterpTable2D> & self, const Segment<-1, 1, -1>& x, const Segment<-1, 1, -1>& y) {
              return GenericFunction<-1, 1>(
                  InterpFunction2D(self).eval(stack(x, y)));
            });

    obj.def("__call__", [](std::shared_ptr<InterpTable2D>& self, const Segment<-1, 2, -1>& xy) {
      return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
    });

    obj.def("__call__", [](std::shared_ptr<InterpTable2D>& self, const GenericFunction<-1, -1>& xy) {
      return GenericFunction<-1, 1>(InterpFunction2D(self).eval(xy));
    });


    obj.def("sf", [](std::shared_ptr<InterpTable2D>& self) {
      return GenericFunction<-1, 1>(InterpFunction2D(self));
    });
    obj.def("vf", [](std::shared_ptr<InterpTable2D>& self) {
      return GenericFunction<-1, -1>(InterpFunction2D(self));
    });


    m.def("InterpTable2DSpeedTest",
          [](const GenericFunction<-1, 1>& tabf,
             double xl,
             double xu,
             double yl,
             double yu,
             int nsamps,
             bool lin) {
            Eigen::ArrayXd xsamps;
            xsamps.setRandom(nsamps);
            xsamps += 1;
            xsamps /= 2;
            xsamps *= (xu - xl);
            xsamps += xl;

            Eigen::ArrayXd ysamps;
            ysamps.setRandom(nsamps);
            ysamps += 1;
            ysamps /= 2;
            ysamps *= (yu - yl);
            ysamps += yl;

            if (lin) {
              xsamps.setLinSpaced(xl, xu);
              ysamps.setLinSpaced(yl, yu);
            }


            Eigen::VectorXd xy(2);
            Vector1<double> f;
            f.setZero();

            Utils::Timer Runtimer;
            Runtimer.start();

            double tmp = 0;
            for (int i = 0; i < nsamps; i++) {


              xy[0] = xsamps[i];
              xy[1] = ysamps[i];

              tabf.compute(xy, f);
              tmp += f[0] / double(i + 3);

              // fmt::print("{0:} \n",f[0]);


              f.setZero();
            }
            Runtimer.stop();
            double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
            fmt::print("Total Time: {0:} ms \n", tseconds * 1000);


            return tmp;
          });
  }


}  // namespace ASSET