#pragma once
#include "VectorFunction.h"


namespace ASSET {


  
  struct InterpTable1D {

    enum class InterpType {
      cubic_interp,
      linear_interp
    };


    using MatType = Eigen::Matrix<double, -1, -1>;

    Eigen::VectorXd ts;
    MatType vs;
    MatType dvs_dts;

    InterpType interp_kind = InterpType::cubic_interp;
    bool teven = true;
    int axis = 0;
    int tsize;
    double ttotal;
    int vlen;
    bool WarnOutOfBounds = true;
    bool ThrowOutOfBounds = false;

    InterpTable1D() {
    }


    InterpTable1D(const Eigen::VectorXd& Ts, const MatType& Vs, int axis, std::string kind) {
      set_data(Ts, Vs, axis, kind);
    }
    InterpTable1D(const Eigen::VectorXd& Ts, const Eigen::VectorXd& Vs, int axis, std::string kind) {
      MatType Vstmp = Vs.transpose();
      set_data(Ts, Vstmp, 1, kind);
    }
    InterpTable1D(const std::vector<Eigen::VectorXd>& Vts, int tvar, std::string kind) {

      if (Vts.size() == 0) {
        throw std::invalid_argument("Input is empty");
      }
      if (Vts[0].size() < 2) {
        throw std::invalid_argument("Invalid sized value-time data.");
      }

      if (tvar < 0) {
        tvar = Vts[0].size() + tvar;
      }
      if (tvar > Vts[0].size() - 1 || tvar < 0) {
        throw std::invalid_argument("Invalid time variable index");
      }

      Eigen::VectorXd Ts(Vts.size());

      Eigen::MatrixXd Vs(Vts[0].size() - 1, Vts.size());

      for (int i = 0; i < Vts.size(); i++) {
        int isize = Vts[i].size();
        if (isize != Vts[0].size()) {
          throw std::invalid_argument("All value-time vectors must have same size");
        }
        int shift = 0;
        for (int j = 0; j < isize; j++) {
          if (j == tvar) {
            Ts[i] = Vts[i][j];
            shift = 1;
          } else {
            Vs.col(i)[j - shift] = Vts[i][j];
          }
        }
      }
      set_data(Ts, Vs, 1, kind);
    }
    void set_data(const Eigen::VectorXd& Ts, const MatType& Vs, int axis, std::string kind) {

      this->ts = Ts;

      if (axis == 1) {
        this->axis = 1;
        this->vs = Vs;
      } else if (axis == 0) {
        this->axis = 0;
        this->vs = Vs.transpose();
      } else {
        throw std::invalid_argument("Interpolation axis must be 0 or 1");
      }

      if (kind == "cubic" || kind == "Cubic") {
        this->interp_kind = InterpType::cubic_interp;
      } else if (kind == "linear" || kind == "Linear") {
        this->interp_kind = InterpType::linear_interp;
      } else {
        throw std::invalid_argument("Unrecognized interpolation type");
      }


      tsize = ts.size();
      vlen = vs.rows();
      ttotal = ts[tsize - 1] - ts[0];

      if (tsize < 5) {
        throw std::invalid_argument("t coordinates must be larger than 4");
      }
      if (tsize != vs.cols()) {
        throw std::invalid_argument("Length of t coordinates must match length of interpolation axis");
      }
      for (int i = 0; i < tsize - 1; i++) {
        if (ts[i + 1] < ts[i]) {
          throw std::invalid_argument("t Coordinates must be in ascending order");
        }
      }

      Eigen::VectorXd testt;
      testt.setLinSpaced(ts.size(), ts[0], ts[tsize - 1]);

      double terr = (ts - testt).lpNorm<Eigen::Infinity>();

      if (terr > abs(ttotal) * 1.0e-12) {
        this->teven = false;
      }


      if (this->interp_kind == InterpType::cubic_interp)
        calc_derivs();
    }


    void calc_derivs() {

      this->dvs_dts.resize(this->vlen, this->tsize);

      Eigen::Matrix<double, 5, 5> stens;
      stens.row(0).setOnes();
      Eigen::Matrix<double, 5, 1> rhs;
      rhs << 0, 1, 0, 0, 0;
      Eigen::Matrix<double, 5, 1> times;
      Eigen::Matrix<double, 5, 1> coeffs;

      bool hitcent = false;
      for (int i = 0; i < this->tsize; i++) {
        int start = 0;
        bool recalc = true;
        if (i + 2 <= this->tsize - 1 && i - 2 >= 0) {
          // central difference
          if (this->teven && hitcent) {
            recalc = false;
          }
          hitcent = true;
          start = i - 2;
        } else if (i < this->tsize - 1 - i) {
          // forward difference
          start = 0;
        } else {
          // backward difference
          start = this->tsize - 5;
        }
        int stepdir = (i < this->tsize - 1) ? 1 : -1;
        double ti = this->ts[i];
        double tstep = std::abs(this->ts[i + stepdir] - ti);
        if (recalc) {
          times = this->ts.segment(start, 5);
          times -= Eigen::Matrix<double, 5, 1>::Constant(ti);
          times /= tstep;
          stens.row(1) = times.transpose();
          stens.row(2) = stens.row(1).cwiseProduct(times.transpose());
          stens.row(3) = stens.row(2).cwiseProduct(times.transpose());
          stens.row(4) = stens.row(3).cwiseProduct(times.transpose());
          coeffs = stens.inverse() * rhs;
        }

        dvs_dts.col(i) = this->vs.middleCols(start, 5) * (coeffs / tstep);
      }
    }

    int get_telem(double t) const {
      int telem;
      if (this->teven) {
        double tlocal = t - ts[0];
        double tstep = ts[1] - ts[0];
        telem = std::min(int(tlocal / tstep), this->tsize - 2);
      } else {
        int center = int(ts.size() / 2);
        int shift = (ts[center] > t) ? 0 : center;
        auto it = std::upper_bound(ts.cbegin(), ts.cend(), t);
        telem = int(it - ts.begin()) - 1;
      }

      telem = std::min(telem, this->tsize - 2);
      telem = std::max(telem, 0);
      return telem;
    }

    template<class VType>
    void interp_impl(double t, int deriv, VType& v, VType& dv_dt, VType& dv2_dt2) const {

      if (WarnOutOfBounds || ThrowOutOfBounds) {
        double eps = std::numeric_limits<double>::epsilon() * ttotal;
        if (t < (ts[0] - eps) || t > (ts[ts.size() - 1] + eps)) {
          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: t= {0:} falls outside of InterpTable1D time range. Data is being extrapolated!!\n",
              t);
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
      }

      double telem = this->get_telem(t);
      double tstep = ts[telem + 1] - ts[telem];
      double tnd = (t - ts[telem]) / tstep;

      if (this->interp_kind == InterpType::cubic_interp) {

        double tnd2 = tnd * tnd;
        double tnd3 = tnd2 * tnd;

        double p0 = (2.0 * tnd3 - 3.0 * tnd2 + 1.0);
        double m0 = (tnd3 - 2.0 * tnd2 + tnd) * tstep;
        double p1 = (-2.0 * tnd3 + 3.0 * tnd2);
        double m1 = (tnd3 - tnd2) * tstep;


        v = vs.col(telem) * p0 + vs.col(telem + 1) * p1 + dvs_dts.col(telem) * m0
            + dvs_dts.col(telem + 1) * m1;


        if (deriv > 0) {

          double p0_dt = (6.0 * tnd2 - 6.0 * tnd) / tstep;
          double m0_dt = (3.0 * tnd2 - 4.0 * tnd + 1.0);
          double p1_dt = (-6.0 * tnd2 + 6.0 * tnd) / tstep;
          double m1_dt = (3.0 * tnd2 - 2.0 * tnd);


          dv_dt = vs.col(telem) * p0_dt + vs.col(telem + 1) * p1_dt + dvs_dts.col(telem) * m0_dt
                  + dvs_dts.col(telem + 1) * m1_dt;


          if (deriv > 1) {

            double p0_dt2 = (12.0 * tnd - 6.0) / (tstep * tstep);
            double m0_dt2 = (6.0 * tnd - 4.0) / tstep;
            double p1_dt2 = (-12.0 * tnd + 6.0) / (tstep * tstep);
            double m1_dt2 = (6.0 * tnd - 2.0) / tstep;

            dv2_dt2 = vs.col(telem) * p0_dt2 + vs.col(telem + 1) * p1_dt2 + dvs_dts.col(telem) * m0_dt2
                      + dvs_dts.col(telem + 1) * m1_dt2;
          }
        }

      } else {
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

    Eigen::MatrixXd interp(const Eigen::VectorXd& ts) const {

      Eigen::MatrixXd vs;
      vs.resize(vlen, ts.size());
      Eigen::VectorXd v;
      v.resize(vlen);

      for (int i = 0; i < ts.size(); i++) {
        interp_impl(ts[i], 0, v, v, v);
        vs.col(i) = v;
        v.setZero();
      }


      return vs;
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> interp_deriv1(double t) const {

      Eigen::VectorXd v;
      v.resize(vlen);
      Eigen::VectorXd dv_dt;
      dv_dt.resize(vlen);

      interp_impl(t, 1, v, dv_dt, dv_dt);

      return std::tuple {v, dv_dt};
    }

    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> interp_deriv2(double t) const {

      Eigen::VectorXd v;
      v.resize(vlen);
      Eigen::VectorXd dv_dt;
      dv_dt.resize(vlen);
      Eigen::VectorXd dv2_dt2;
      dv2_dt2.resize(vlen);


      interp_impl(t, 2, v, dv_dt, dv2_dt2);

      return std::tuple {v, dv_dt, dv2_dt2};
    }
  };

  template<int ORR>
  struct InterpFunction1D : VectorFunction<InterpFunction1D<ORR>, 1, ORR, Analytic, Analytic> {
    using Base = VectorFunction<InterpFunction1D<ORR>, 1, ORR, Analytic, Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    std::shared_ptr<InterpTable1D> tab;


    InterpFunction1D() {
    }
    InterpFunction1D(std::shared_ptr<InterpTable1D> tab) : tab(tab) {
      this->setIORows(1, tab->vlen);
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto& v) {
        this->tab->interp_impl(x[0], 0, v, v, v);
        fx = v;
      };

      ASSET::MemoryManager::allocate_run(this->ORows(), Impl, TempSpec<Output<Scalar>>(this->ORows(), 1));
    }
    template<class InType, class OutType, class JacType>
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

      ASSET::MemoryManager::allocate_run(this->ORows(),
                                         Impl,
                                         TempSpec<Output<Scalar>>(this->ORows(), 1),
                                         TempSpec<Output<Scalar>>(this->ORows(), 1));
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

      auto Impl = [&](auto& v, auto& dv_dt, auto& dv2_dt2) {
        this->tab->interp_impl(x[0], 2, v, dv_dt, dv2_dt2);
        fx = v;
        jx = dv_dt;
        adjgrad[0] = dv_dt.dot(adjvars);
        adjhess(0, 0) = dv2_dt2.dot(adjvars);
      };

      ASSET::MemoryManager::allocate_run(this->ORows(),
                                         Impl,
                                         TempSpec<Output<Scalar>>(this->ORows(), 1),
                                         TempSpec<Output<Scalar>>(this->ORows(), 1),
                                         TempSpec<Output<Scalar>>(this->ORows(), 1));
    }
  };


  static void InterpTable1DBuild(py::module& m) {

    using MatType = InterpTable1D::MatType;
    auto obj = py::class_<InterpTable1D, std::shared_ptr<InterpTable1D>>(m, "InterpTable1D");

    obj.def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, int, std::string>(),
            py::arg("ts"),
            py::arg("Vs"),
            py::arg("axis") = 0,
            py::arg("kind") = std::string("cubic"));

    obj.def(py::init<const Eigen::VectorXd&, const MatType&, int, std::string>(),
            py::arg("ts"),
            py::arg("Vs"),
            py::arg("axis") = 0,
            py::arg("kind") = std::string("cubic"));

    obj.def(py::init<const std::vector<Eigen::VectorXd>&, int, std::string>(),
            py::arg("Vts"),
            py::arg("tvar") = -1,
            py::arg("kind") = std::string("cubic"));

    obj.def("interp", py::overload_cast<double>(&InterpTable1D::interp, py::const_));
    obj.def("interp", py::overload_cast<const Eigen::VectorXd&>(&InterpTable1D::interp, py::const_));

    obj.def("__call__", py::overload_cast<double>(&InterpTable1D::interp, py::const_), py::is_operator());
    obj.def("__call__",
            py::overload_cast<const Eigen::VectorXd&>(&InterpTable1D::interp, py::const_),
            py::is_operator());

    obj.def("__call__", [](std::shared_ptr<InterpTable1D> & self, const GenericFunction<-1, 1>& t) {
      py::object pyfun;
      if (self->vlen == 1) {
        auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
        pyfun = py::cast(f);
      } else {
        auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
        pyfun = py::cast(f);
      }
      return pyfun;
    });

    
    obj.def("__call__", [](std::shared_ptr<InterpTable1D>& self, const Segment<-1, 1, -1>& t) {
      py::object pyfun;
      fmt::print("Call\n");
      if (self->vlen == 1) {
        auto f = GenericFunction<-1, 1>(InterpFunction1D<1>(self).eval(t));
        pyfun = py::cast(f);
      } else {
        auto f = GenericFunction<-1, -1>(InterpFunction1D<-1>(self).eval(t));
        pyfun = py::cast(f);
      }
      return pyfun;
    });

    


    obj.def("interp_deriv1", &InterpTable1D::interp_deriv1);
    obj.def("interp_deriv2", &InterpTable1D::interp_deriv2);

    obj.def_readwrite("WarnOutOfBounds", &InterpTable1D::WarnOutOfBounds);
    obj.def_readwrite("ThrowOutOfBounds", &InterpTable1D::ThrowOutOfBounds);


    obj.def("sf", [](std::shared_ptr<InterpTable1D>& self) {
      if (self->vlen != 1) {
        throw std::invalid_argument(
            "InterpTable1D storing Vector data cannot be converted to Scalar Function.");
      }
      return GenericFunction<-1, 1>(InterpFunction1D<1>(self));
    });
    obj.def("vf", [](std::shared_ptr<InterpTable1D>& self) {
      return GenericFunction<-1, -1>(InterpFunction1D<-1>(self));
    });
  }


}  // namespace ASSET
