#pragma once
#include <pybind11/eigen/tensor.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include "VectorFunction.h"
    namespace ASSET {

  struct InterpTable4D {

    enum class InterpType
    {
      cubic_interp,
      linear_interp
    };

    Eigen::VectorXd xs;
    Eigen::VectorXd ys;
    Eigen::VectorXd zs;
    Eigen::VectorXd ws;


    // numpy meshgrid ij format (x,y,z,w)
    Eigen::Tensor<double, 4> fs;

    // First Directional Derivs
    Eigen::Tensor<double, 4> fs_dx;
    Eigen::Tensor<double, 4> fs_dy;
    Eigen::Tensor<double, 4> fs_dz;
    Eigen::Tensor<double, 4> fs_dw;

    // Second Cross Derivs
    Eigen::Tensor<double, 4> fs_dxdy;
    Eigen::Tensor<double, 4> fs_dxdz;
    Eigen::Tensor<double, 4> fs_dxdw;
    Eigen::Tensor<double, 4> fs_dydz;
    Eigen::Tensor<double, 4> fs_dydw;
    Eigen::Tensor<double, 4> fs_dzdw;

    // Third Cross Derivs
    Eigen::Tensor<double, 4> fs_dxdydz;
    Eigen::Tensor<double, 4> fs_dxdydw;
    Eigen::Tensor<double, 4> fs_dxdzdw;
    Eigen::Tensor<double, 4> fs_dydzdw;

    // Fourth Cross Deriv
    Eigen::Tensor<double, 4> fs_dxdydzdw;


    Eigen::Tensor<Eigen::Matrix<double, 256, 1>, 4> alphavecs;


    InterpType interp_kind = InterpType::linear_interp;

    bool xeven = true;
    bool yeven = true;
    bool zeven = true;
    bool weven = true;

    int xsize;
    double xtotal;
    int ysize;
    double ytotal;
    int zsize;
    double ztotal;
    int wsize;
    double wtotal;

    bool cache_alpha = false;
    int cache_threads = 1;

    bool WarnOutOfBounds = true;
    bool ThrowOutOfBounds = false;

    InterpTable4D() {
    }

    InterpTable4D(const Eigen::VectorXd& Xs,
                  const Eigen::VectorXd& Ys,
                  const Eigen::VectorXd& Zs,
                  const Eigen::VectorXd& Ws,

                  const Eigen::Tensor<double, 4>& Fs,
                  std::string kind,
                  bool cache) {

      this->xs = Xs;
      this->ys = Ys;
      this->zs = Zs;
      this->ws = Zs;

      this->fs = Fs;
      this->cache_alpha = cache;

      
      if (kind == "cubic" || kind == "Cubic") {
        this->interp_kind = InterpType::cubic_interp;
      } else if (kind == "linear" || kind == "Linear") {
        this->interp_kind = InterpType::linear_interp;
      } else {
        throw std::invalid_argument("Unrecognized interpolation type");
      }


      xsize = xs.size();
      ysize = ys.size();
      zsize = zs.size();
      wsize = ws.size();

      if (xsize < 5) {
        throw std::invalid_argument("X coordinates must be larger than 4");
      }
      if (ysize < 5) {
        throw std::invalid_argument("Y  coordinates must be larger than 4");
      }
      if (zsize < 5) {
        throw std::invalid_argument("Z  coordinates must be larger than 4");
      }
      if (wsize < 5) {
        throw std::invalid_argument("W  coordinates must be larger than 4");
      }


      if (xsize != fs.dimension(0)) {
        throw std::invalid_argument("X coordinates must be first dimension of value tensor");
      }
      if (ysize != fs.dimension(1)) {
        throw std::invalid_argument("Y coordinates must be second dimension of value tensor");
      }
      if (zsize != fs.dimension(2)) {
        throw std::invalid_argument("Z coordinates must be third dimension of value tensor");
      }
      if (wsize != fs.dimension(3)) {
        throw std::invalid_argument("W coordinates must be third dimension of value tensor");
      }

      for (int i = 0; i < xsize - 1; i++) {
        if (xs[i + 1] < xs[i]) {
          throw std::invalid_argument("X coordinates must be in ascending order");
        }
      }
      for (int i = 0; i < ysize - 1; i++) {
        if (ys[i + 1] < ys[i]) {
          throw std::invalid_argument("Y coordinates must be in ascending order");
        }
      }
      for (int i = 0; i < zsize - 1; i++) {
        if (zs[i + 1] < zs[i]) {
          throw std::invalid_argument("Z coordinates must be in ascending order");
        }
      }
      for (int i = 0; i < wsize - 1; i++) {
        if (ws[i + 1] < ws[i]) {
          throw std::invalid_argument("W coordinates must be in ascending order");
        }
      }

      xtotal = xs[xsize - 1] - xs[0];
      ytotal = ys[ysize - 1] - ys[0];
      ztotal = zs[zsize - 1] - zs[0];
      wtotal = ws[wsize - 1] - ws[0];


      Eigen::VectorXd testx;
      testx.setLinSpaced(xsize, xs[0], xs[xsize - 1]);
      Eigen::VectorXd testy;
      testy.setLinSpaced(ysize, ys[0], ys[ysize - 1]);
      Eigen::VectorXd testz;
      testz.setLinSpaced(zsize, zs[0], zs[zsize - 1]);
      Eigen::VectorXd testw;
      testw.setLinSpaced(wsize, ws[0], ws[wsize - 1]);


      double xerr = (xs - testx).lpNorm<Eigen::Infinity>();
      double yerr = (ys - testy).lpNorm<Eigen::Infinity>();
      double zerr = (zs - testz).lpNorm<Eigen::Infinity>();
      double werr = (ws - testw).lpNorm<Eigen::Infinity>();


      if (xerr > abs(xtotal) * 1.0e-12) {
        this->xeven = false;
      }
      if (yerr > abs(ytotal) * 1.0e-12) {
        this->yeven = false;
      }
      if (zerr > abs(ztotal) * 1.0e-12) {
        this->zeven = false;
      }
      if (werr > abs(wtotal) * 1.0e-12) {
        this->weven = false;
      }

      if (this->interp_kind == InterpType::cubic_interp) {
        //this->fill_Cmat();
        //this->calc_derivs();
        if (this->cache_alpha) {
         // this->cache_alphavecs();
        }
      }
    }



     static int find_elem(const Eigen::VectorXd& ts, bool teven, double t) {
      int elem;
      if (teven) {
        double tlocal = t - ts[0];
        double tstep = ts[1] - ts[0];
        elem = int(tlocal / tstep);
      } else {
        auto it = std::upper_bound(ts.begin(), ts.end(), t);
        elem = int(it - ts.begin()) - 1;
      }
      elem = std::min(elem, int(ts.size() - 2));
      elem = std::max(elem, 0);
      return elem;
    }

    std::tuple<int, int, int,int> get_xyzwelems(double x, double y, double z,double w) const {

      int xelem = this->find_elem(this->xs, this->xeven, x);
      int yelem = this->find_elem(this->ys, this->yeven, y);
      int zelem = this->find_elem(this->zs, this->zeven, z);
      int welem = this->find_elem(this->ws, this->weven, w);

      return std::tuple {xelem, yelem, zelem,welem};
    }


    void calc_derivs() {

      fs_dx.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dy.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));

      fs_dxdy.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dxdz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dxdw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dydz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dydw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dzdw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));

      fs_dxdydz.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dxdydw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dxdzdw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));
      fs_dydzdw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));

      fs_dxdydzdw.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));


      auto fdiffimpl = [&](int dir, bool even, const auto& ts, const auto& src, auto& dest) {
        Eigen::Matrix<double, 5, 5> stens;
        stens.row(0).setOnes();
        Eigen::Matrix<double, 5, 1> rhs;
        rhs << 0, 1, 0, 0, 0;
        Eigen::Matrix<double, 5, 1> times;
        Eigen::Matrix<double, 5, 1> coeffs;

        int tsize = ts.size();
        bool hitcent = false;
        for (int i = 0; i < tsize; i++) {
          int start = 0;
          bool recalc = true;
          if (i + 2 <= tsize - 1 && i - 2 >= 0) {
            // central difference
            if (even && hitcent) {
              recalc = false;
            }
            hitcent = true;
            start = i - 2;
          } else if (i < tsize - 1 - i) {
            // forward difference
            start = 0;
          } else {
            // backward difference
            start = tsize - 5;
          }
          int stepdir = (i < tsize - 1) ? 1 : -1;
          double ti = ts[i];
          double tstep = std::abs(ts[i + stepdir] - ti);
          if (recalc) {
            times = ts.segment(start, 5);
            times -= Eigen::Matrix<double, 5, 1>::Constant(ti);
            times /= tstep;
            stens.row(1) = times.transpose();
            stens.row(2) = (stens.row(1).cwiseProduct(times.transpose())).eval();
            stens.row(3) = (stens.row(2).cwiseProduct(times.transpose())).eval();
            stens.row(4) = (stens.row(3).cwiseProduct(times.transpose())).eval();

            coeffs = stens.inverse() * rhs;
          }
          dest.chip(i, dir) = src.chip(start, dir) * (coeffs[0] / tstep)
                              + src.chip(start + 1, dir) * (coeffs[1] / tstep)
                              + src.chip(start + 2, dir) * (coeffs[2] / tstep)
                              + src.chip(start + 3, dir) * (coeffs[3] / tstep)
                              + src.chip(start + 4, dir) * (coeffs[4] / tstep);
        }
      };

      fdiffimpl(0, this->xeven, this->xs, this->fs, this->fs_dx);
      fdiffimpl(1, this->yeven, this->ys, this->fs, this->fs_dy);
      fdiffimpl(2, this->zeven, this->zs, this->fs, this->fs_dz);
      fdiffimpl(3, this->weven, this->ws, this->fs, this->fs_dw);


      fdiffimpl(1, this->yeven, this->ys, this->fs_dx, this->fs_dxdy);
      fdiffimpl(2, this->zeven, this->zs, this->fs_dx, this->fs_dxdz);
      fdiffimpl(3, this->weven, this->ws, this->fs_dx, this->fs_dxdw);
      fdiffimpl(2, this->zeven, this->zs, this->fs_dy, this->fs_dydz);
      fdiffimpl(3, this->weven, this->ws, this->fs_dy, this->fs_dydw);
      fdiffimpl(3, this->weven, this->ws, this->fs_dz, this->fs_dzdw);

      fdiffimpl(2, this->zeven, this->zs, this->fs_dxdy, this->fs_dxdydz);
      fdiffimpl(3, this->weven, this->ws, this->fs_dxdy, this->fs_dxdydw);
      fdiffimpl(3, this->weven, this->ws, this->fs_dxdz, this->fs_dxdzdw);
      fdiffimpl(3, this->weven, this->ws, this->fs_dydz, this->fs_dydzdw);

      fdiffimpl(3, this->weven, this->ws, this->fs_dxdydz, this->fs_dxdydzdw);


    }



     Eigen::Matrix<double, 256, 1> calc_alphavec(int xelem, int yelem, int zelem,int welem) const {

      double xstep = xs[xelem + 1] - xs[xelem];
      double ystep = ys[yelem + 1] - ys[yelem];
      double zstep = zs[zelem + 1] - zs[zelem];
      double wstep = ws[welem + 1] - ws[zelem];

      Eigen::Matrix<double, 256, 1> bvec;
      Eigen::Matrix<double, 256, 1> alphavec;



      // TODO: decide on corner ordering
      auto fillop = [&](auto start, const auto& src,double scale) {
       
          bvec[start] = src(xelem, yelem, zelem,welem)*scale;
          //...
          //...
        
      };


      //

      fillop(0, this->fs,1.0);
      fillop(16, this->fs_dx,xstep);
      fillop(32, this->fs_dy,ystep);
      fillop(48, this->fs_dz,zstep);
      fillop(64, this->fs_dw,wstep);

      fillop(80, this->fs_dxdy, (xstep * ystep));
      fillop(96, this->fs_dxdz, (xstep * zstep));
      fillop(112, this->fs_dxdw, (xstep * wstep));

      fillop(128, this->fs_dydz, (ystep * zstep));
      fillop(144, this->fs_dydw, (ystep * wstep));
      fillop(160, this->fs_dzdw, (zstep * wstep));

      fillop(176, this->fs_dxdydz, (xstep * ystep * zstep));
      fillop(192, this->fs_dxdydw, (xstep * ystep * wstep));
      fillop(208, this->fs_dxdzdw, (xstep * zstep * wstep));

      fillop(224, this->fs_dydzdw, (ystep * zstep * wstep));
      fillop(240, this->fs_dxdydzdw, (xstep*ystep * zstep * wstep));


     
      alphavec.noalias() = calc_bvecproduct(bvec);

      return alphavec;
    }

    Eigen::Matrix<double, 256, 1> get_alphavec(int xelem, int yelem, int zelem,int welem) const {
      if (this->cache_alpha) {
        return this->alphavecs(xelem, yelem, zelem,welem);
      } else {
        return this->calc_alphavec(xelem, yelem, zelem,welem);
      }
    }




    void cache_alphavecs() {
      this->alphavecs.resize(
          fs.dimension(0) - 1, 
          fs.dimension(1) - 1,
          fs.dimension(2) - 1, 
          fs.dimension(3) - 1);

      for (int i = 0; i < wsize - 1; i++) {
        for (int j = 0; j < zsize - 1; j++) {
          for (int k = 0; k < ysize - 1; k++) {
            for (int l = 0; l < xsize - 1; l++) {
              this->alphavecs(l,k, j, i) = this->calc_alphavec(l, k, j, i);
            }
          }
        }
      }
    }





    Eigen::Matrix<double, 256, 1> calc_bvecproduct(const Eigen::Matrix<double, 256, 1>& bvec) const {
     
     // TODO: use sympy to codegen this product, coefficient matrix is large and sparse so this should make it scale better
      

       Eigen::Matrix<double, 256, 1> alpha;
     
       //alpha[0] = bvec[0]*c1 + ....

       return alpha;
    }




    void interp_impl(double x,
                     double y,
                     double z,
                     double w,
                     int deriv,
                     double& fval,
                     Eigen::Vector4<double>& dfxyzw,
                     Eigen::Matrix4<double>& d2fxyzw) const {

      if (WarnOutOfBounds || ThrowOutOfBounds) {
        double xeps = std::numeric_limits<double>::epsilon() * xtotal;
        if (x < (xs[0] - xeps) || x > (xs[xs.size() - 1] + xeps)) {

          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: x coordinate falls outside of InterpTable4D range. Data is being extrapolated!!\n");
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
        double yeps = std::numeric_limits<double>::epsilon() * ytotal;
        if (y < (ys[0] - yeps) || y > (ys[ys.size() - 1]) + yeps) {
          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: y coordinate falls outside of InterpTable4D range. Data is being extrapolated!!\n");
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
        double zeps = std::numeric_limits<double>::epsilon() * ztotal;
        if (z < (zs[0] - zeps) || z > (zs[zs.size() - 1]) + zeps) {
          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: z coordinate falls outside of InterpTable4D range. Data is being extrapolated!!\n");
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
        double weps = std::numeric_limits<double>::epsilon() * wtotal;
        if (w < (ws[0] - weps) || w > (ws[ws.size() - 1]) + weps) {
          fmt::print(
              fmt::fg(fmt::color::red),
              "WARNING: w coordinate falls outside of InterpTable4D range. Data is being extrapolated!!\n");
          if (ThrowOutOfBounds) {
            throw std::invalid_argument("");
          }
        }
      }


      auto [xelem, yelem, zelem,welem] = get_xyzwelems(x, y, z,w);

      double xstep = xs[xelem + 1] - xs[xelem];
      double ystep = ys[yelem + 1] - ys[yelem];
      double zstep = zs[zelem + 1] - zs[zelem];
      double wstep = ws[zelem + 1] - ws[zelem];


      double xf = (x - xs[xelem]) / xstep;
      double yf = (y - ys[yelem]) / ystep;
      double zf = (z - zs[zelem]) / zstep;
      double wf = (z - ws[zelem]) / wstep;


      if (this->interp_kind == InterpType::cubic_interp) {
        Eigen::Matrix<double, 256, 1> alphavec = this->get_alphavec(xelem, yelem, zelem,welem);

        double xf2 = xf * xf;
        double xf3 = xf2 * xf;

        double yf2 = yf * yf;
        double yf3 = yf2 * yf;

        double zf2 = zf * zf;
        double zf3 = zf2 * zf;

        double wf2 = wf * wf;
        double wf3 = wf2 * wf;

        Eigen::Vector4d yfs {1, yf, yf2, yf3};
        Eigen::Vector4d zfs {1, zf, zf2, zf3};
        Eigen::Vector4d wfs {1, wf, wf2, wf3};


        fval = 0;

        for (int i = 0, start = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++, start += 4) {
              fval += (yfs[k] * zfs[j] * wfs[i])
                      * (alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2]
                         + xf3 * alphavec[start + 3]);
            }
          }
        }

        if (deriv > 0) {

          if (deriv > 1) {
            
          }
        }

      } else {
        
        fval =0;

        if (deriv > 0) {
          

          if (deriv > 1) {
            
          }
        }
      }
    }

    double interp(double x, double y, double z,double w) const {
      double f;
      Eigen::Vector4<double> dfxyzw;
      Eigen::Matrix4<double> d2fxyzw;
      interp_impl(x, y, z,w, 0, f, dfxyzw, d2fxyzw);
      return f;
    }
    std::tuple<double, Eigen::Vector4<double>> 
        interp_deriv1(double x, double y, double z, double w) const {
      double f;
      Eigen::Vector4<double> dfxyzw;
      Eigen::Matrix4<double> d2fxyzw;
      interp_impl(x, y, z, w, 1, f, dfxyzw, d2fxyzw);
      return std::tuple {f, dfxyzw};
    }
    std::tuple<double, Eigen::Vector4<double>, Eigen::Matrix4<double>>
        interp_deriv2(double x, double y, double z, double w) const {
      double f;
      Eigen::Vector4<double> dfxyzw;
      Eigen::Matrix4<double> d2fxyzw;
      interp_impl(x, y, z, w, 2, f, dfxyzw, d2fxyzw);
      return std::tuple {f, dfxyzw, d2fxyzw};
    }

    

  };


  struct InterpFunction4D : VectorFunction<InterpFunction4D, 4, 1, Analytic, Analytic> {
    using Base = VectorFunction<InterpFunction4D, 4, 1, Analytic, Analytic>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    std::shared_ptr<InterpTable4D> tab;


    InterpFunction4D() {
    }
    InterpFunction4D(std::shared_ptr<InterpTable4D> tab) : tab(tab) {
      this->setIORows(4, 1);
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx[0] = this->tab->interp(x[0], x[1], x[2],x[3);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      auto [f, dxdydzdw] = this->tab->interp_deriv1(x[0], x[1], x[2],x[3]);
      fx[0] = f;
      jx = dxdydzdw.transpose();
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

      auto [f, dxdydzdw, d2xdydzdw] = this->tab->interp_deriv2(x[0], x[1], x[2], x[3]);
      fx[0] = f;
      jx = dxdydzdw.transpose();
      adjgrad = adjvars[0] * dxdydzdw;
      adjhess = adjvars[0] * d2xdydzdw;
    }
  };



}