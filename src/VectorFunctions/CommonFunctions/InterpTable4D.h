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

    // Holds f, and all derivatives at each data point contiguaously
    // Improved runtime by factor of two over holding separately like in the 3d table
    Eigen::Tensor<Eigen::Matrix<double, 16, 1>, 4> fs_all;

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
      this->ws = Ws;

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
        this->calc_derivs();
        if (this->cache_alpha) {
          this->cache_alphavecs();
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

      fdiffimpl(0, this->xeven, this->xs, this->fs, fs_dx);
      fdiffimpl(1, this->yeven, this->ys, this->fs, fs_dy);
      fdiffimpl(2, this->zeven, this->zs, this->fs, fs_dz);
      fdiffimpl(3, this->weven, this->ws, this->fs, fs_dw);


      fdiffimpl(1, this->yeven, this->ys, fs_dx, fs_dxdy);
      fdiffimpl(2, this->zeven, this->zs, fs_dx, fs_dxdz);
      fdiffimpl(3, this->weven, this->ws, fs_dx, fs_dxdw);
      fdiffimpl(2, this->zeven, this->zs, fs_dy, fs_dydz);
      fdiffimpl(3, this->weven, this->ws, fs_dy, fs_dydw);
      fdiffimpl(3, this->weven, this->ws, fs_dz, fs_dzdw);

      fdiffimpl(2, this->zeven, this->zs, fs_dxdy, fs_dxdydz);
      fdiffimpl(3, this->weven, this->ws, fs_dxdy, fs_dxdydw);
      fdiffimpl(3, this->weven, this->ws, fs_dxdz, fs_dxdzdw);
      fdiffimpl(3, this->weven, this->ws, fs_dydz, fs_dydzdw);

      fdiffimpl(3, this->weven, this->ws, fs_dxdydz, fs_dxdydzdw);

      Eigen::Matrix<double, 16, 1> tmp;

      fs_all.resize(fs.dimension(0), fs.dimension(1), fs.dimension(2), fs.dimension(3));

      for (int i = 0; i < wsize ; i++) {
          for (int j = 0; j < zsize ; j++) {
              for (int k = 0; k < ysize ; k++) {
                  for (int l = 0; l < xsize ; l++) {
                      tmp[0] = fs(l, k, j, i);
                      tmp[1] = fs_dx(l, k, j, i);
                      tmp[2] = fs_dy(l, k, j, i);
                      tmp[3] = fs_dz(l, k, j, i);
                      tmp[4] = fs_dw(l, k, j, i);
                      tmp[5] = fs_dxdy(l, k, j, i);
                      tmp[6] = fs_dxdz(l, k, j, i);
                      tmp[7] = fs_dxdw(l, k, j, i);
                      tmp[8] = fs_dydz(l, k, j, i);
                      tmp[9] = fs_dydw(l, k, j, i);
                      tmp[10] = fs_dzdw(l, k, j, i);
                      tmp[11] = fs_dxdydz(l, k, j, i);
                      tmp[12] = fs_dxdydw(l, k, j, i);
                      tmp[13] = fs_dxdzdw(l, k, j, i);
                      tmp[14] = fs_dydzdw(l, k, j, i);
                      tmp[15] = fs_dxdydzdw(l, k, j, i);
                      fs_all(l, k, j, i) = tmp;
                  }
              }
          }
      }
    }

     Eigen::Matrix<double, 256, 1> calc_alphavec(int xelem, int yelem, int zelem,int welem) const {

      double xstep = xs[xelem + 1] - xs[xelem];
      double ystep = ys[yelem + 1] - ys[yelem];
      double zstep = zs[zelem + 1] - zs[zelem];
      double wstep = ws[welem + 1] - ws[welem];

      Eigen::Matrix<double, 256, 1> bvec;
      Eigen::Matrix<double, 16, 1> tmp;
      Eigen::Matrix<double, 16, 1> scales;

      scales[0] = 1.0;
      scales[1] = xstep;
      scales[2] = ystep;
      scales[3] = zstep;
      scales[4] = wstep;
      scales[5] = (xstep * ystep);
      scales[6] = (xstep * zstep);
      scales[7] = (xstep * wstep);
      scales[8] = (ystep * zstep);
      scales[9] = (ystep * wstep);
      scales[10] = (zstep * wstep);
      scales[11] = (xstep * ystep * zstep);
      scales[12] = (xstep * ystep * wstep);
      scales[13] = (xstep * zstep * wstep);
      scales[14] = (ystep * zstep * wstep);
      scales[15] = (xstep * ystep * zstep * wstep);


      int corner = 0;
      auto fillop = [&](int xoffs, int yoffs, int zoffs, int woffs) {
          tmp = this->fs_all(xelem + xoffs, yelem + yoffs, zelem + zoffs, welem + woffs).cwiseProduct(scales);
          for (int i = 0; i < 16; i++) {
              bvec[corner + 16 * i] = tmp[i];
          }
          corner++;
      };

      fillop(0, 0, 0, 0);
      fillop(1, 0, 0, 0);
      fillop(0, 1, 0, 0);
      fillop(1, 1, 0, 0);
      fillop(0, 0, 1, 0);
      fillop(1, 0, 1, 0);
      fillop(0, 1, 1, 0);
      fillop(1, 1, 1, 0);

      fillop(0, 0, 0, 1);
      fillop(1, 0, 0, 1);
      fillop(0, 1, 0, 1);
      fillop(1, 1, 0, 1);
      fillop(0, 0, 1, 1);
      fillop(1, 0, 1, 1);
      fillop(0, 1, 1, 1);
      fillop(1, 1, 1, 1);

      return apply_coeefs(bvec);
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


    Eigen::Matrix<double, 256, 1> apply_coeefs(const Eigen::Matrix<double, 256, 1>& bvec) const {
     
      

       Eigen::Matrix<double, 256, 1> alphavec;
     
       alphavec[0] = +1 * bvec[0];
       alphavec[1] = +1 * bvec[16];
       alphavec[2] = -3 * bvec[0] + 3 * bvec[1] - 2 * bvec[16] - 1 * bvec[17];
       alphavec[3] = +2 * bvec[0] - 2 * bvec[1] + 1 * bvec[16] + 1 * bvec[17];
       alphavec[4] = +1 * bvec[32];
       alphavec[5] = +1 * bvec[80];
       alphavec[6] = -3 * bvec[32] + 3 * bvec[33] - 2 * bvec[80] - 1 * bvec[81];
       alphavec[7] = +2 * bvec[32] - 2 * bvec[33] + 1 * bvec[80] + 1 * bvec[81];
       alphavec[8] = -3 * bvec[0] + 3 * bvec[2] - 2 * bvec[32] - 1 * bvec[34];
       alphavec[9] = -3 * bvec[16] + 3 * bvec[18] - 2 * bvec[80] - 1 * bvec[82];
       alphavec[10] = +9 * bvec[0] - 9 * bvec[1] - 9 * bvec[2] + 9 * bvec[3] + 6 * bvec[16] + 3 * bvec[17] - 6 * bvec[18] - 3 * bvec[19] + 6 * bvec[32] - 6 * bvec[33] + 3 * bvec[34] - 3 * bvec[35] + 4 * bvec[80] + 2 * bvec[81] + 2 * bvec[82] + 1 * bvec[83];
       alphavec[11] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[2] - 6 * bvec[3] - 3 * bvec[16] - 3 * bvec[17] + 3 * bvec[18] + 3 * bvec[19] - 4 * bvec[32] + 4 * bvec[33] - 2 * bvec[34] + 2 * bvec[35] - 2 * bvec[80] - 2 * bvec[81] - 1 * bvec[82] - 1 * bvec[83];
       alphavec[12] = +2 * bvec[0] - 2 * bvec[2] + 1 * bvec[32] + 1 * bvec[34];
       alphavec[13] = +2 * bvec[16] - 2 * bvec[18] + 1 * bvec[80] + 1 * bvec[82];
       alphavec[14] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[2] - 6 * bvec[3] - 4 * bvec[16] - 2 * bvec[17] + 4 * bvec[18] + 2 * bvec[19] - 3 * bvec[32] + 3 * bvec[33] - 3 * bvec[34] + 3 * bvec[35] - 2 * bvec[80] - 1 * bvec[81] - 2 * bvec[82] - 1 * bvec[83];
       alphavec[15] = +4 * bvec[0] - 4 * bvec[1] - 4 * bvec[2] + 4 * bvec[3] + 2 * bvec[16] + 2 * bvec[17] - 2 * bvec[18] - 2 * bvec[19] + 2 * bvec[32] - 2 * bvec[33] + 2 * bvec[34] - 2 * bvec[35] + 1 * bvec[80] + 1 * bvec[81] + 1 * bvec[82] + 1 * bvec[83];
       alphavec[16] = +1 * bvec[48];
       alphavec[17] = +1 * bvec[96];
       alphavec[18] = -3 * bvec[48] + 3 * bvec[49] - 2 * bvec[96] - 1 * bvec[97];
       alphavec[19] = +2 * bvec[48] - 2 * bvec[49] + 1 * bvec[96] + 1 * bvec[97];
       alphavec[20] = +1 * bvec[128];
       alphavec[21] = +1 * bvec[176];
       alphavec[22] = -3 * bvec[128] + 3 * bvec[129] - 2 * bvec[176] - 1 * bvec[177];
       alphavec[23] = +2 * bvec[128] - 2 * bvec[129] + 1 * bvec[176] + 1 * bvec[177];
       alphavec[24] = -3 * bvec[48] + 3 * bvec[50] - 2 * bvec[128] - 1 * bvec[130];
       alphavec[25] = -3 * bvec[96] + 3 * bvec[98] - 2 * bvec[176] - 1 * bvec[178];
       alphavec[26] = +9 * bvec[48] - 9 * bvec[49] - 9 * bvec[50] + 9 * bvec[51] + 6 * bvec[96] + 3 * bvec[97] - 6 * bvec[98] - 3 * bvec[99] + 6 * bvec[128] - 6 * bvec[129] + 3 * bvec[130] - 3 * bvec[131] + 4 * bvec[176] + 2 * bvec[177] + 2 * bvec[178] + 1 * bvec[179];
       alphavec[27] = -6 * bvec[48] + 6 * bvec[49] + 6 * bvec[50] - 6 * bvec[51] - 3 * bvec[96] - 3 * bvec[97] + 3 * bvec[98] + 3 * bvec[99] - 4 * bvec[128] + 4 * bvec[129] - 2 * bvec[130] + 2 * bvec[131] - 2 * bvec[176] - 2 * bvec[177] - 1 * bvec[178] - 1 * bvec[179];
       alphavec[28] = +2 * bvec[48] - 2 * bvec[50] + 1 * bvec[128] + 1 * bvec[130];
       alphavec[29] = +2 * bvec[96] - 2 * bvec[98] + 1 * bvec[176] + 1 * bvec[178];
       alphavec[30] = -6 * bvec[48] + 6 * bvec[49] + 6 * bvec[50] - 6 * bvec[51] - 4 * bvec[96] - 2 * bvec[97] + 4 * bvec[98] + 2 * bvec[99] - 3 * bvec[128] + 3 * bvec[129] - 3 * bvec[130] + 3 * bvec[131] - 2 * bvec[176] - 1 * bvec[177] - 2 * bvec[178] - 1 * bvec[179];
       alphavec[31] = +4 * bvec[48] - 4 * bvec[49] - 4 * bvec[50] + 4 * bvec[51] + 2 * bvec[96] + 2 * bvec[97] - 2 * bvec[98] - 2 * bvec[99] + 2 * bvec[128] - 2 * bvec[129] + 2 * bvec[130] - 2 * bvec[131] + 1 * bvec[176] + 1 * bvec[177] + 1 * bvec[178] + 1 * bvec[179];
       alphavec[32] = -3 * bvec[0] + 3 * bvec[4] - 2 * bvec[48] - 1 * bvec[52];
       alphavec[33] = -3 * bvec[16] + 3 * bvec[20] - 2 * bvec[96] - 1 * bvec[100];
       alphavec[34] = +9 * bvec[0] - 9 * bvec[1] - 9 * bvec[4] + 9 * bvec[5] + 6 * bvec[16] + 3 * bvec[17] - 6 * bvec[20] - 3 * bvec[21] + 6 * bvec[48] - 6 * bvec[49] + 3 * bvec[52] - 3 * bvec[53] + 4 * bvec[96] + 2 * bvec[97] + 2 * bvec[100] + 1 * bvec[101];
       alphavec[35] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[4] - 6 * bvec[5] - 3 * bvec[16] - 3 * bvec[17] + 3 * bvec[20] + 3 * bvec[21] - 4 * bvec[48] + 4 * bvec[49] - 2 * bvec[52] + 2 * bvec[53] - 2 * bvec[96] - 2 * bvec[97] - 1 * bvec[100] - 1 * bvec[101];
       alphavec[36] = -3 * bvec[32] + 3 * bvec[36] - 2 * bvec[128] - 1 * bvec[132];
       alphavec[37] = -3 * bvec[80] + 3 * bvec[84] - 2 * bvec[176] - 1 * bvec[180];
       alphavec[38] = +9 * bvec[32] - 9 * bvec[33] - 9 * bvec[36] + 9 * bvec[37] + 6 * bvec[80] + 3 * bvec[81] - 6 * bvec[84] - 3 * bvec[85] + 6 * bvec[128] - 6 * bvec[129] + 3 * bvec[132] - 3 * bvec[133] + 4 * bvec[176] + 2 * bvec[177] + 2 * bvec[180] + 1 * bvec[181];
       alphavec[39] = -6 * bvec[32] + 6 * bvec[33] + 6 * bvec[36] - 6 * bvec[37] - 3 * bvec[80] - 3 * bvec[81] + 3 * bvec[84] + 3 * bvec[85] - 4 * bvec[128] + 4 * bvec[129] - 2 * bvec[132] + 2 * bvec[133] - 2 * bvec[176] - 2 * bvec[177] - 1 * bvec[180] - 1 * bvec[181];
       alphavec[40] = +9 * bvec[0] - 9 * bvec[2] - 9 * bvec[4] + 9 * bvec[6] + 6 * bvec[32] + 3 * bvec[34] - 6 * bvec[36] - 3 * bvec[38] + 6 * bvec[48] - 6 * bvec[50] + 3 * bvec[52] - 3 * bvec[54] + 4 * bvec[128] + 2 * bvec[130] + 2 * bvec[132] + 1 * bvec[134];
       alphavec[41] = +9 * bvec[16] - 9 * bvec[18] - 9 * bvec[20] + 9 * bvec[22] + 6 * bvec[80] + 3 * bvec[82] - 6 * bvec[84] - 3 * bvec[86] + 6 * bvec[96] - 6 * bvec[98] + 3 * bvec[100] - 3 * bvec[102] + 4 * bvec[176] + 2 * bvec[178] + 2 * bvec[180] + 1 * bvec[182];
       alphavec[42] = -27 * bvec[0] + 27 * bvec[1] + 27 * bvec[2] - 27 * bvec[3] + 27 * bvec[4] - 27 * bvec[5] - 27 * bvec[6] + 27 * bvec[7] - 18 * bvec[16] - 9 * bvec[17] + 18 * bvec[18] + 9 * bvec[19] + 18 * bvec[20] + 9 * bvec[21] - 18 * bvec[22] - 9 * bvec[23] - 18 * bvec[32] + 18 * bvec[33] - 9 * bvec[34] + 9 * bvec[35] + 18 * bvec[36] - 18 * bvec[37] + 9 * bvec[38] - 9 * bvec[39] - 18 * bvec[48] + 18 * bvec[49] + 18 * bvec[50] - 18 * bvec[51] - 9 * bvec[52] + 9 * bvec[53] + 9 * bvec[54] - 9 * bvec[55] - 12 * bvec[80] - 6 * bvec[81] - 6 * bvec[82] - 3 * bvec[83] + 12 * bvec[84] + 6 * bvec[85] + 6 * bvec[86] + 3 * bvec[87] - 12 * bvec[96] - 6 * bvec[97] + 12 * bvec[98] + 6 * bvec[99] - 6 * bvec[100] - 3 * bvec[101] + 6 * bvec[102] + 3 * bvec[103] - 12 * bvec[128] + 12 * bvec[129] - 6 * bvec[130] + 6 * bvec[131] - 6 * bvec[132] + 6 * bvec[133] - 3 * bvec[134] + 3 * bvec[135] - 8 * bvec[176] - 4 * bvec[177] - 4 * bvec[178] - 2 * bvec[179] - 4 * bvec[180] - 2 * bvec[181] - 2 * bvec[182] - 1 * bvec[183];
       alphavec[43] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[4] + 18 * bvec[5] + 18 * bvec[6] - 18 * bvec[7] + 9 * bvec[16] + 9 * bvec[17] - 9 * bvec[18] - 9 * bvec[19] - 9 * bvec[20] - 9 * bvec[21] + 9 * bvec[22] + 9 * bvec[23] + 12 * bvec[32] - 12 * bvec[33] + 6 * bvec[34] - 6 * bvec[35] - 12 * bvec[36] + 12 * bvec[37] - 6 * bvec[38] + 6 * bvec[39] + 12 * bvec[48] - 12 * bvec[49] - 12 * bvec[50] + 12 * bvec[51] + 6 * bvec[52] - 6 * bvec[53] - 6 * bvec[54] + 6 * bvec[55] + 6 * bvec[80] + 6 * bvec[81] + 3 * bvec[82] + 3 * bvec[83] - 6 * bvec[84] - 6 * bvec[85] - 3 * bvec[86] - 3 * bvec[87] + 6 * bvec[96] + 6 * bvec[97] - 6 * bvec[98] - 6 * bvec[99] + 3 * bvec[100] + 3 * bvec[101] - 3 * bvec[102] - 3 * bvec[103] + 8 * bvec[128] - 8 * bvec[129] + 4 * bvec[130] - 4 * bvec[131] + 4 * bvec[132] - 4 * bvec[133] + 2 * bvec[134] - 2 * bvec[135] + 4 * bvec[176] + 4 * bvec[177] + 2 * bvec[178] + 2 * bvec[179] + 2 * bvec[180] + 2 * bvec[181] + 1 * bvec[182] + 1 * bvec[183];
       alphavec[44] = -6 * bvec[0] + 6 * bvec[2] + 6 * bvec[4] - 6 * bvec[6] - 3 * bvec[32] - 3 * bvec[34] + 3 * bvec[36] + 3 * bvec[38] - 4 * bvec[48] + 4 * bvec[50] - 2 * bvec[52] + 2 * bvec[54] - 2 * bvec[128] - 2 * bvec[130] - 1 * bvec[132] - 1 * bvec[134];
       alphavec[45] = -6 * bvec[16] + 6 * bvec[18] + 6 * bvec[20] - 6 * bvec[22] - 3 * bvec[80] - 3 * bvec[82] + 3 * bvec[84] + 3 * bvec[86] - 4 * bvec[96] + 4 * bvec[98] - 2 * bvec[100] + 2 * bvec[102] - 2 * bvec[176] - 2 * bvec[178] - 1 * bvec[180] - 1 * bvec[182];
       alphavec[46] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[4] + 18 * bvec[5] + 18 * bvec[6] - 18 * bvec[7] + 12 * bvec[16] + 6 * bvec[17] - 12 * bvec[18] - 6 * bvec[19] - 12 * bvec[20] - 6 * bvec[21] + 12 * bvec[22] + 6 * bvec[23] + 9 * bvec[32] - 9 * bvec[33] + 9 * bvec[34] - 9 * bvec[35] - 9 * bvec[36] + 9 * bvec[37] - 9 * bvec[38] + 9 * bvec[39] + 12 * bvec[48] - 12 * bvec[49] - 12 * bvec[50] + 12 * bvec[51] + 6 * bvec[52] - 6 * bvec[53] - 6 * bvec[54] + 6 * bvec[55] + 6 * bvec[80] + 3 * bvec[81] + 6 * bvec[82] + 3 * bvec[83] - 6 * bvec[84] - 3 * bvec[85] - 6 * bvec[86] - 3 * bvec[87] + 8 * bvec[96] + 4 * bvec[97] - 8 * bvec[98] - 4 * bvec[99] + 4 * bvec[100] + 2 * bvec[101] - 4 * bvec[102] - 2 * bvec[103] + 6 * bvec[128] - 6 * bvec[129] + 6 * bvec[130] - 6 * bvec[131] + 3 * bvec[132] - 3 * bvec[133] + 3 * bvec[134] - 3 * bvec[135] + 4 * bvec[176] + 2 * bvec[177] + 4 * bvec[178] + 2 * bvec[179] + 2 * bvec[180] + 1 * bvec[181] + 2 * bvec[182] + 1 * bvec[183];
       alphavec[47] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[4] - 12 * bvec[5] - 12 * bvec[6] + 12 * bvec[7] - 6 * bvec[16] - 6 * bvec[17] + 6 * bvec[18] + 6 * bvec[19] + 6 * bvec[20] + 6 * bvec[21] - 6 * bvec[22] - 6 * bvec[23] - 6 * bvec[32] + 6 * bvec[33] - 6 * bvec[34] + 6 * bvec[35] + 6 * bvec[36] - 6 * bvec[37] + 6 * bvec[38] - 6 * bvec[39] - 8 * bvec[48] + 8 * bvec[49] + 8 * bvec[50] - 8 * bvec[51] - 4 * bvec[52] + 4 * bvec[53] + 4 * bvec[54] - 4 * bvec[55] - 3 * bvec[80] - 3 * bvec[81] - 3 * bvec[82] - 3 * bvec[83] + 3 * bvec[84] + 3 * bvec[85] + 3 * bvec[86] + 3 * bvec[87] - 4 * bvec[96] - 4 * bvec[97] + 4 * bvec[98] + 4 * bvec[99] - 2 * bvec[100] - 2 * bvec[101] + 2 * bvec[102] + 2 * bvec[103] - 4 * bvec[128] + 4 * bvec[129] - 4 * bvec[130] + 4 * bvec[131] - 2 * bvec[132] + 2 * bvec[133] - 2 * bvec[134] + 2 * bvec[135] - 2 * bvec[176] - 2 * bvec[177] - 2 * bvec[178] - 2 * bvec[179] - 1 * bvec[180] - 1 * bvec[181] - 1 * bvec[182] - 1 * bvec[183];
       alphavec[48] = +2 * bvec[0] - 2 * bvec[4] + 1 * bvec[48] + 1 * bvec[52];
       alphavec[49] = +2 * bvec[16] - 2 * bvec[20] + 1 * bvec[96] + 1 * bvec[100];
       alphavec[50] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[4] - 6 * bvec[5] - 4 * bvec[16] - 2 * bvec[17] + 4 * bvec[20] + 2 * bvec[21] - 3 * bvec[48] + 3 * bvec[49] - 3 * bvec[52] + 3 * bvec[53] - 2 * bvec[96] - 1 * bvec[97] - 2 * bvec[100] - 1 * bvec[101];
       alphavec[51] = +4 * bvec[0] - 4 * bvec[1] - 4 * bvec[4] + 4 * bvec[5] + 2 * bvec[16] + 2 * bvec[17] - 2 * bvec[20] - 2 * bvec[21] + 2 * bvec[48] - 2 * bvec[49] + 2 * bvec[52] - 2 * bvec[53] + 1 * bvec[96] + 1 * bvec[97] + 1 * bvec[100] + 1 * bvec[101];
       alphavec[52] = +2 * bvec[32] - 2 * bvec[36] + 1 * bvec[128] + 1 * bvec[132];
       alphavec[53] = +2 * bvec[80] - 2 * bvec[84] + 1 * bvec[176] + 1 * bvec[180];
       alphavec[54] = -6 * bvec[32] + 6 * bvec[33] + 6 * bvec[36] - 6 * bvec[37] - 4 * bvec[80] - 2 * bvec[81] + 4 * bvec[84] + 2 * bvec[85] - 3 * bvec[128] + 3 * bvec[129] - 3 * bvec[132] + 3 * bvec[133] - 2 * bvec[176] - 1 * bvec[177] - 2 * bvec[180] - 1 * bvec[181];
       alphavec[55] = +4 * bvec[32] - 4 * bvec[33] - 4 * bvec[36] + 4 * bvec[37] + 2 * bvec[80] + 2 * bvec[81] - 2 * bvec[84] - 2 * bvec[85] + 2 * bvec[128] - 2 * bvec[129] + 2 * bvec[132] - 2 * bvec[133] + 1 * bvec[176] + 1 * bvec[177] + 1 * bvec[180] + 1 * bvec[181];
       alphavec[56] = -6 * bvec[0] + 6 * bvec[2] + 6 * bvec[4] - 6 * bvec[6] - 4 * bvec[32] - 2 * bvec[34] + 4 * bvec[36] + 2 * bvec[38] - 3 * bvec[48] + 3 * bvec[50] - 3 * bvec[52] + 3 * bvec[54] - 2 * bvec[128] - 1 * bvec[130] - 2 * bvec[132] - 1 * bvec[134];
       alphavec[57] = -6 * bvec[16] + 6 * bvec[18] + 6 * bvec[20] - 6 * bvec[22] - 4 * bvec[80] - 2 * bvec[82] + 4 * bvec[84] + 2 * bvec[86] - 3 * bvec[96] + 3 * bvec[98] - 3 * bvec[100] + 3 * bvec[102] - 2 * bvec[176] - 1 * bvec[178] - 2 * bvec[180] - 1 * bvec[182];
       alphavec[58] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[4] + 18 * bvec[5] + 18 * bvec[6] - 18 * bvec[7] + 12 * bvec[16] + 6 * bvec[17] - 12 * bvec[18] - 6 * bvec[19] - 12 * bvec[20] - 6 * bvec[21] + 12 * bvec[22] + 6 * bvec[23] + 12 * bvec[32] - 12 * bvec[33] + 6 * bvec[34] - 6 * bvec[35] - 12 * bvec[36] + 12 * bvec[37] - 6 * bvec[38] + 6 * bvec[39] + 9 * bvec[48] - 9 * bvec[49] - 9 * bvec[50] + 9 * bvec[51] + 9 * bvec[52] - 9 * bvec[53] - 9 * bvec[54] + 9 * bvec[55] + 8 * bvec[80] + 4 * bvec[81] + 4 * bvec[82] + 2 * bvec[83] - 8 * bvec[84] - 4 * bvec[85] - 4 * bvec[86] - 2 * bvec[87] + 6 * bvec[96] + 3 * bvec[97] - 6 * bvec[98] - 3 * bvec[99] + 6 * bvec[100] + 3 * bvec[101] - 6 * bvec[102] - 3 * bvec[103] + 6 * bvec[128] - 6 * bvec[129] + 3 * bvec[130] - 3 * bvec[131] + 6 * bvec[132] - 6 * bvec[133] + 3 * bvec[134] - 3 * bvec[135] + 4 * bvec[176] + 2 * bvec[177] + 2 * bvec[178] + 1 * bvec[179] + 4 * bvec[180] + 2 * bvec[181] + 2 * bvec[182] + 1 * bvec[183];
       alphavec[59] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[4] - 12 * bvec[5] - 12 * bvec[6] + 12 * bvec[7] - 6 * bvec[16] - 6 * bvec[17] + 6 * bvec[18] + 6 * bvec[19] + 6 * bvec[20] + 6 * bvec[21] - 6 * bvec[22] - 6 * bvec[23] - 8 * bvec[32] + 8 * bvec[33] - 4 * bvec[34] + 4 * bvec[35] + 8 * bvec[36] - 8 * bvec[37] + 4 * bvec[38] - 4 * bvec[39] - 6 * bvec[48] + 6 * bvec[49] + 6 * bvec[50] - 6 * bvec[51] - 6 * bvec[52] + 6 * bvec[53] + 6 * bvec[54] - 6 * bvec[55] - 4 * bvec[80] - 4 * bvec[81] - 2 * bvec[82] - 2 * bvec[83] + 4 * bvec[84] + 4 * bvec[85] + 2 * bvec[86] + 2 * bvec[87] - 3 * bvec[96] - 3 * bvec[97] + 3 * bvec[98] + 3 * bvec[99] - 3 * bvec[100] - 3 * bvec[101] + 3 * bvec[102] + 3 * bvec[103] - 4 * bvec[128] + 4 * bvec[129] - 2 * bvec[130] + 2 * bvec[131] - 4 * bvec[132] + 4 * bvec[133] - 2 * bvec[134] + 2 * bvec[135] - 2 * bvec[176] - 2 * bvec[177] - 1 * bvec[178] - 1 * bvec[179] - 2 * bvec[180] - 2 * bvec[181] - 1 * bvec[182] - 1 * bvec[183];
       alphavec[60] = +4 * bvec[0] - 4 * bvec[2] - 4 * bvec[4] + 4 * bvec[6] + 2 * bvec[32] + 2 * bvec[34] - 2 * bvec[36] - 2 * bvec[38] + 2 * bvec[48] - 2 * bvec[50] + 2 * bvec[52] - 2 * bvec[54] + 1 * bvec[128] + 1 * bvec[130] + 1 * bvec[132] + 1 * bvec[134];
       alphavec[61] = +4 * bvec[16] - 4 * bvec[18] - 4 * bvec[20] + 4 * bvec[22] + 2 * bvec[80] + 2 * bvec[82] - 2 * bvec[84] - 2 * bvec[86] + 2 * bvec[96] - 2 * bvec[98] + 2 * bvec[100] - 2 * bvec[102] + 1 * bvec[176] + 1 * bvec[178] + 1 * bvec[180] + 1 * bvec[182];
       alphavec[62] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[4] - 12 * bvec[5] - 12 * bvec[6] + 12 * bvec[7] - 8 * bvec[16] - 4 * bvec[17] + 8 * bvec[18] + 4 * bvec[19] + 8 * bvec[20] + 4 * bvec[21] - 8 * bvec[22] - 4 * bvec[23] - 6 * bvec[32] + 6 * bvec[33] - 6 * bvec[34] + 6 * bvec[35] + 6 * bvec[36] - 6 * bvec[37] + 6 * bvec[38] - 6 * bvec[39] - 6 * bvec[48] + 6 * bvec[49] + 6 * bvec[50] - 6 * bvec[51] - 6 * bvec[52] + 6 * bvec[53] + 6 * bvec[54] - 6 * bvec[55] - 4 * bvec[80] - 2 * bvec[81] - 4 * bvec[82] - 2 * bvec[83] + 4 * bvec[84] + 2 * bvec[85] + 4 * bvec[86] + 2 * bvec[87] - 4 * bvec[96] - 2 * bvec[97] + 4 * bvec[98] + 2 * bvec[99] - 4 * bvec[100] - 2 * bvec[101] + 4 * bvec[102] + 2 * bvec[103] - 3 * bvec[128] + 3 * bvec[129] - 3 * bvec[130] + 3 * bvec[131] - 3 * bvec[132] + 3 * bvec[133] - 3 * bvec[134] + 3 * bvec[135] - 2 * bvec[176] - 1 * bvec[177] - 2 * bvec[178] - 1 * bvec[179] - 2 * bvec[180] - 1 * bvec[181] - 2 * bvec[182] - 1 * bvec[183];
       alphavec[63] = +8 * bvec[0] - 8 * bvec[1] - 8 * bvec[2] + 8 * bvec[3] - 8 * bvec[4] + 8 * bvec[5] + 8 * bvec[6] - 8 * bvec[7] + 4 * bvec[16] + 4 * bvec[17] - 4 * bvec[18] - 4 * bvec[19] - 4 * bvec[20] - 4 * bvec[21] + 4 * bvec[22] + 4 * bvec[23] + 4 * bvec[32] - 4 * bvec[33] + 4 * bvec[34] - 4 * bvec[35] - 4 * bvec[36] + 4 * bvec[37] - 4 * bvec[38] + 4 * bvec[39] + 4 * bvec[48] - 4 * bvec[49] - 4 * bvec[50] + 4 * bvec[51] + 4 * bvec[52] - 4 * bvec[53] - 4 * bvec[54] + 4 * bvec[55] + 2 * bvec[80] + 2 * bvec[81] + 2 * bvec[82] + 2 * bvec[83] - 2 * bvec[84] - 2 * bvec[85] - 2 * bvec[86] - 2 * bvec[87] + 2 * bvec[96] + 2 * bvec[97] - 2 * bvec[98] - 2 * bvec[99] + 2 * bvec[100] + 2 * bvec[101] - 2 * bvec[102] - 2 * bvec[103] + 2 * bvec[128] - 2 * bvec[129] + 2 * bvec[130] - 2 * bvec[131] + 2 * bvec[132] - 2 * bvec[133] + 2 * bvec[134] - 2 * bvec[135] + 1 * bvec[176] + 1 * bvec[177] + 1 * bvec[178] + 1 * bvec[179] + 1 * bvec[180] + 1 * bvec[181] + 1 * bvec[182] + 1 * bvec[183];
       alphavec[64] = +1 * bvec[64];
       alphavec[65] = +1 * bvec[112];
       alphavec[66] = -3 * bvec[64] + 3 * bvec[65] - 2 * bvec[112] - 1 * bvec[113];
       alphavec[67] = +2 * bvec[64] - 2 * bvec[65] + 1 * bvec[112] + 1 * bvec[113];
       alphavec[68] = +1 * bvec[144];
       alphavec[69] = +1 * bvec[192];
       alphavec[70] = -3 * bvec[144] + 3 * bvec[145] - 2 * bvec[192] - 1 * bvec[193];
       alphavec[71] = +2 * bvec[144] - 2 * bvec[145] + 1 * bvec[192] + 1 * bvec[193];
       alphavec[72] = -3 * bvec[64] + 3 * bvec[66] - 2 * bvec[144] - 1 * bvec[146];
       alphavec[73] = -3 * bvec[112] + 3 * bvec[114] - 2 * bvec[192] - 1 * bvec[194];
       alphavec[74] = +9 * bvec[64] - 9 * bvec[65] - 9 * bvec[66] + 9 * bvec[67] + 6 * bvec[112] + 3 * bvec[113] - 6 * bvec[114] - 3 * bvec[115] + 6 * bvec[144] - 6 * bvec[145] + 3 * bvec[146] - 3 * bvec[147] + 4 * bvec[192] + 2 * bvec[193] + 2 * bvec[194] + 1 * bvec[195];
       alphavec[75] = -6 * bvec[64] + 6 * bvec[65] + 6 * bvec[66] - 6 * bvec[67] - 3 * bvec[112] - 3 * bvec[113] + 3 * bvec[114] + 3 * bvec[115] - 4 * bvec[144] + 4 * bvec[145] - 2 * bvec[146] + 2 * bvec[147] - 2 * bvec[192] - 2 * bvec[193] - 1 * bvec[194] - 1 * bvec[195];
       alphavec[76] = +2 * bvec[64] - 2 * bvec[66] + 1 * bvec[144] + 1 * bvec[146];
       alphavec[77] = +2 * bvec[112] - 2 * bvec[114] + 1 * bvec[192] + 1 * bvec[194];
       alphavec[78] = -6 * bvec[64] + 6 * bvec[65] + 6 * bvec[66] - 6 * bvec[67] - 4 * bvec[112] - 2 * bvec[113] + 4 * bvec[114] + 2 * bvec[115] - 3 * bvec[144] + 3 * bvec[145] - 3 * bvec[146] + 3 * bvec[147] - 2 * bvec[192] - 1 * bvec[193] - 2 * bvec[194] - 1 * bvec[195];
       alphavec[79] = +4 * bvec[64] - 4 * bvec[65] - 4 * bvec[66] + 4 * bvec[67] + 2 * bvec[112] + 2 * bvec[113] - 2 * bvec[114] - 2 * bvec[115] + 2 * bvec[144] - 2 * bvec[145] + 2 * bvec[146] - 2 * bvec[147] + 1 * bvec[192] + 1 * bvec[193] + 1 * bvec[194] + 1 * bvec[195];
       alphavec[80] = +1 * bvec[160];
       alphavec[81] = +1 * bvec[208];
       alphavec[82] = -3 * bvec[160] + 3 * bvec[161] - 2 * bvec[208] - 1 * bvec[209];
       alphavec[83] = +2 * bvec[160] - 2 * bvec[161] + 1 * bvec[208] + 1 * bvec[209];
       alphavec[84] = +1 * bvec[224];
       alphavec[85] = +1 * bvec[240];
       alphavec[86] = -3 * bvec[224] + 3 * bvec[225] - 2 * bvec[240] - 1 * bvec[241];
       alphavec[87] = +2 * bvec[224] - 2 * bvec[225] + 1 * bvec[240] + 1 * bvec[241];
       alphavec[88] = -3 * bvec[160] + 3 * bvec[162] - 2 * bvec[224] - 1 * bvec[226];
       alphavec[89] = -3 * bvec[208] + 3 * bvec[210] - 2 * bvec[240] - 1 * bvec[242];
       alphavec[90] = +9 * bvec[160] - 9 * bvec[161] - 9 * bvec[162] + 9 * bvec[163] + 6 * bvec[208] + 3 * bvec[209] - 6 * bvec[210] - 3 * bvec[211] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[226] - 3 * bvec[227] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[242] + 1 * bvec[243];
       alphavec[91] = -6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 3 * bvec[208] - 3 * bvec[209] + 3 * bvec[210] + 3 * bvec[211] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[226] + 2 * bvec[227] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[242] - 1 * bvec[243];
       alphavec[92] = +2 * bvec[160] - 2 * bvec[162] + 1 * bvec[224] + 1 * bvec[226];
       alphavec[93] = +2 * bvec[208] - 2 * bvec[210] + 1 * bvec[240] + 1 * bvec[242];
       alphavec[94] = -6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 4 * bvec[208] - 2 * bvec[209] + 4 * bvec[210] + 2 * bvec[211] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[226] + 3 * bvec[227] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[242] - 1 * bvec[243];
       alphavec[95] = +4 * bvec[160] - 4 * bvec[161] - 4 * bvec[162] + 4 * bvec[163] + 2 * bvec[208] + 2 * bvec[209] - 2 * bvec[210] - 2 * bvec[211] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[226] - 2 * bvec[227] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[242] + 1 * bvec[243];
       alphavec[96] = -3 * bvec[64] + 3 * bvec[68] - 2 * bvec[160] - 1 * bvec[164];
       alphavec[97] = -3 * bvec[112] + 3 * bvec[116] - 2 * bvec[208] - 1 * bvec[212];
       alphavec[98] = +9 * bvec[64] - 9 * bvec[65] - 9 * bvec[68] + 9 * bvec[69] + 6 * bvec[112] + 3 * bvec[113] - 6 * bvec[116] - 3 * bvec[117] + 6 * bvec[160] - 6 * bvec[161] + 3 * bvec[164] - 3 * bvec[165] + 4 * bvec[208] + 2 * bvec[209] + 2 * bvec[212] + 1 * bvec[213];
       alphavec[99] = -6 * bvec[64] + 6 * bvec[65] + 6 * bvec[68] - 6 * bvec[69] - 3 * bvec[112] - 3 * bvec[113] + 3 * bvec[116] + 3 * bvec[117] - 4 * bvec[160] + 4 * bvec[161] - 2 * bvec[164] + 2 * bvec[165] - 2 * bvec[208] - 2 * bvec[209] - 1 * bvec[212] - 1 * bvec[213];
       alphavec[100] = -3 * bvec[144] + 3 * bvec[148] - 2 * bvec[224] - 1 * bvec[228];
       alphavec[101] = -3 * bvec[192] + 3 * bvec[196] - 2 * bvec[240] - 1 * bvec[244];
       alphavec[102] = +9 * bvec[144] - 9 * bvec[145] - 9 * bvec[148] + 9 * bvec[149] + 6 * bvec[192] + 3 * bvec[193] - 6 * bvec[196] - 3 * bvec[197] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[228] - 3 * bvec[229] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[244] + 1 * bvec[245];
       alphavec[103] = -6 * bvec[144] + 6 * bvec[145] + 6 * bvec[148] - 6 * bvec[149] - 3 * bvec[192] - 3 * bvec[193] + 3 * bvec[196] + 3 * bvec[197] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[228] + 2 * bvec[229] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[244] - 1 * bvec[245];
       alphavec[104] = +9 * bvec[64] - 9 * bvec[66] - 9 * bvec[68] + 9 * bvec[70] + 6 * bvec[144] + 3 * bvec[146] - 6 * bvec[148] - 3 * bvec[150] + 6 * bvec[160] - 6 * bvec[162] + 3 * bvec[164] - 3 * bvec[166] + 4 * bvec[224] + 2 * bvec[226] + 2 * bvec[228] + 1 * bvec[230];
       alphavec[105] = +9 * bvec[112] - 9 * bvec[114] - 9 * bvec[116] + 9 * bvec[118] + 6 * bvec[192] + 3 * bvec[194] - 6 * bvec[196] - 3 * bvec[198] + 6 * bvec[208] - 6 * bvec[210] + 3 * bvec[212] - 3 * bvec[214] + 4 * bvec[240] + 2 * bvec[242] + 2 * bvec[244] + 1 * bvec[246];
       alphavec[106] = -27 * bvec[64] + 27 * bvec[65] + 27 * bvec[66] - 27 * bvec[67] + 27 * bvec[68] - 27 * bvec[69] - 27 * bvec[70] + 27 * bvec[71] - 18 * bvec[112] - 9 * bvec[113] + 18 * bvec[114] + 9 * bvec[115] + 18 * bvec[116] + 9 * bvec[117] - 18 * bvec[118] - 9 * bvec[119] - 18 * bvec[144] + 18 * bvec[145] - 9 * bvec[146] + 9 * bvec[147] + 18 * bvec[148] - 18 * bvec[149] + 9 * bvec[150] - 9 * bvec[151] - 18 * bvec[160] + 18 * bvec[161] + 18 * bvec[162] - 18 * bvec[163] - 9 * bvec[164] + 9 * bvec[165] + 9 * bvec[166] - 9 * bvec[167] - 12 * bvec[192] - 6 * bvec[193] - 6 * bvec[194] - 3 * bvec[195] + 12 * bvec[196] + 6 * bvec[197] + 6 * bvec[198] + 3 * bvec[199] - 12 * bvec[208] - 6 * bvec[209] + 12 * bvec[210] + 6 * bvec[211] - 6 * bvec[212] - 3 * bvec[213] + 6 * bvec[214] + 3 * bvec[215] - 12 * bvec[224] + 12 * bvec[225] - 6 * bvec[226] + 6 * bvec[227] - 6 * bvec[228] + 6 * bvec[229] - 3 * bvec[230] + 3 * bvec[231] - 8 * bvec[240] - 4 * bvec[241] - 4 * bvec[242] - 2 * bvec[243] - 4 * bvec[244] - 2 * bvec[245] - 2 * bvec[246] - 1 * bvec[247];
       alphavec[107] = +18 * bvec[64] - 18 * bvec[65] - 18 * bvec[66] + 18 * bvec[67] - 18 * bvec[68] + 18 * bvec[69] + 18 * bvec[70] - 18 * bvec[71] + 9 * bvec[112] + 9 * bvec[113] - 9 * bvec[114] - 9 * bvec[115] - 9 * bvec[116] - 9 * bvec[117] + 9 * bvec[118] + 9 * bvec[119] + 12 * bvec[144] - 12 * bvec[145] + 6 * bvec[146] - 6 * bvec[147] - 12 * bvec[148] + 12 * bvec[149] - 6 * bvec[150] + 6 * bvec[151] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 6 * bvec[164] - 6 * bvec[165] - 6 * bvec[166] + 6 * bvec[167] + 6 * bvec[192] + 6 * bvec[193] + 3 * bvec[194] + 3 * bvec[195] - 6 * bvec[196] - 6 * bvec[197] - 3 * bvec[198] - 3 * bvec[199] + 6 * bvec[208] + 6 * bvec[209] - 6 * bvec[210] - 6 * bvec[211] + 3 * bvec[212] + 3 * bvec[213] - 3 * bvec[214] - 3 * bvec[215] + 8 * bvec[224] - 8 * bvec[225] + 4 * bvec[226] - 4 * bvec[227] + 4 * bvec[228] - 4 * bvec[229] + 2 * bvec[230] - 2 * bvec[231] + 4 * bvec[240] + 4 * bvec[241] + 2 * bvec[242] + 2 * bvec[243] + 2 * bvec[244] + 2 * bvec[245] + 1 * bvec[246] + 1 * bvec[247];
       alphavec[108] = -6 * bvec[64] + 6 * bvec[66] + 6 * bvec[68] - 6 * bvec[70] - 3 * bvec[144] - 3 * bvec[146] + 3 * bvec[148] + 3 * bvec[150] - 4 * bvec[160] + 4 * bvec[162] - 2 * bvec[164] + 2 * bvec[166] - 2 * bvec[224] - 2 * bvec[226] - 1 * bvec[228] - 1 * bvec[230];
       alphavec[109] = -6 * bvec[112] + 6 * bvec[114] + 6 * bvec[116] - 6 * bvec[118] - 3 * bvec[192] - 3 * bvec[194] + 3 * bvec[196] + 3 * bvec[198] - 4 * bvec[208] + 4 * bvec[210] - 2 * bvec[212] + 2 * bvec[214] - 2 * bvec[240] - 2 * bvec[242] - 1 * bvec[244] - 1 * bvec[246];
       alphavec[110] = +18 * bvec[64] - 18 * bvec[65] - 18 * bvec[66] + 18 * bvec[67] - 18 * bvec[68] + 18 * bvec[69] + 18 * bvec[70] - 18 * bvec[71] + 12 * bvec[112] + 6 * bvec[113] - 12 * bvec[114] - 6 * bvec[115] - 12 * bvec[116] - 6 * bvec[117] + 12 * bvec[118] + 6 * bvec[119] + 9 * bvec[144] - 9 * bvec[145] + 9 * bvec[146] - 9 * bvec[147] - 9 * bvec[148] + 9 * bvec[149] - 9 * bvec[150] + 9 * bvec[151] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 6 * bvec[164] - 6 * bvec[165] - 6 * bvec[166] + 6 * bvec[167] + 6 * bvec[192] + 3 * bvec[193] + 6 * bvec[194] + 3 * bvec[195] - 6 * bvec[196] - 3 * bvec[197] - 6 * bvec[198] - 3 * bvec[199] + 8 * bvec[208] + 4 * bvec[209] - 8 * bvec[210] - 4 * bvec[211] + 4 * bvec[212] + 2 * bvec[213] - 4 * bvec[214] - 2 * bvec[215] + 6 * bvec[224] - 6 * bvec[225] + 6 * bvec[226] - 6 * bvec[227] + 3 * bvec[228] - 3 * bvec[229] + 3 * bvec[230] - 3 * bvec[231] + 4 * bvec[240] + 2 * bvec[241] + 4 * bvec[242] + 2 * bvec[243] + 2 * bvec[244] + 1 * bvec[245] + 2 * bvec[246] + 1 * bvec[247];
       alphavec[111] = -12 * bvec[64] + 12 * bvec[65] + 12 * bvec[66] - 12 * bvec[67] + 12 * bvec[68] - 12 * bvec[69] - 12 * bvec[70] + 12 * bvec[71] - 6 * bvec[112] - 6 * bvec[113] + 6 * bvec[114] + 6 * bvec[115] + 6 * bvec[116] + 6 * bvec[117] - 6 * bvec[118] - 6 * bvec[119] - 6 * bvec[144] + 6 * bvec[145] - 6 * bvec[146] + 6 * bvec[147] + 6 * bvec[148] - 6 * bvec[149] + 6 * bvec[150] - 6 * bvec[151] - 8 * bvec[160] + 8 * bvec[161] + 8 * bvec[162] - 8 * bvec[163] - 4 * bvec[164] + 4 * bvec[165] + 4 * bvec[166] - 4 * bvec[167] - 3 * bvec[192] - 3 * bvec[193] - 3 * bvec[194] - 3 * bvec[195] + 3 * bvec[196] + 3 * bvec[197] + 3 * bvec[198] + 3 * bvec[199] - 4 * bvec[208] - 4 * bvec[209] + 4 * bvec[210] + 4 * bvec[211] - 2 * bvec[212] - 2 * bvec[213] + 2 * bvec[214] + 2 * bvec[215] - 4 * bvec[224] + 4 * bvec[225] - 4 * bvec[226] + 4 * bvec[227] - 2 * bvec[228] + 2 * bvec[229] - 2 * bvec[230] + 2 * bvec[231] - 2 * bvec[240] - 2 * bvec[241] - 2 * bvec[242] - 2 * bvec[243] - 1 * bvec[244] - 1 * bvec[245] - 1 * bvec[246] - 1 * bvec[247];
       alphavec[112] = +2 * bvec[64] - 2 * bvec[68] + 1 * bvec[160] + 1 * bvec[164];
       alphavec[113] = +2 * bvec[112] - 2 * bvec[116] + 1 * bvec[208] + 1 * bvec[212];
       alphavec[114] = -6 * bvec[64] + 6 * bvec[65] + 6 * bvec[68] - 6 * bvec[69] - 4 * bvec[112] - 2 * bvec[113] + 4 * bvec[116] + 2 * bvec[117] - 3 * bvec[160] + 3 * bvec[161] - 3 * bvec[164] + 3 * bvec[165] - 2 * bvec[208] - 1 * bvec[209] - 2 * bvec[212] - 1 * bvec[213];
       alphavec[115] = +4 * bvec[64] - 4 * bvec[65] - 4 * bvec[68] + 4 * bvec[69] + 2 * bvec[112] + 2 * bvec[113] - 2 * bvec[116] - 2 * bvec[117] + 2 * bvec[160] - 2 * bvec[161] + 2 * bvec[164] - 2 * bvec[165] + 1 * bvec[208] + 1 * bvec[209] + 1 * bvec[212] + 1 * bvec[213];
       alphavec[116] = +2 * bvec[144] - 2 * bvec[148] + 1 * bvec[224] + 1 * bvec[228];
       alphavec[117] = +2 * bvec[192] - 2 * bvec[196] + 1 * bvec[240] + 1 * bvec[244];
       alphavec[118] = -6 * bvec[144] + 6 * bvec[145] + 6 * bvec[148] - 6 * bvec[149] - 4 * bvec[192] - 2 * bvec[193] + 4 * bvec[196] + 2 * bvec[197] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[228] + 3 * bvec[229] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[244] - 1 * bvec[245];
       alphavec[119] = +4 * bvec[144] - 4 * bvec[145] - 4 * bvec[148] + 4 * bvec[149] + 2 * bvec[192] + 2 * bvec[193] - 2 * bvec[196] - 2 * bvec[197] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[228] - 2 * bvec[229] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[244] + 1 * bvec[245];
       alphavec[120] = -6 * bvec[64] + 6 * bvec[66] + 6 * bvec[68] - 6 * bvec[70] - 4 * bvec[144] - 2 * bvec[146] + 4 * bvec[148] + 2 * bvec[150] - 3 * bvec[160] + 3 * bvec[162] - 3 * bvec[164] + 3 * bvec[166] - 2 * bvec[224] - 1 * bvec[226] - 2 * bvec[228] - 1 * bvec[230];
       alphavec[121] = -6 * bvec[112] + 6 * bvec[114] + 6 * bvec[116] - 6 * bvec[118] - 4 * bvec[192] - 2 * bvec[194] + 4 * bvec[196] + 2 * bvec[198] - 3 * bvec[208] + 3 * bvec[210] - 3 * bvec[212] + 3 * bvec[214] - 2 * bvec[240] - 1 * bvec[242] - 2 * bvec[244] - 1 * bvec[246];
       alphavec[122] = +18 * bvec[64] - 18 * bvec[65] - 18 * bvec[66] + 18 * bvec[67] - 18 * bvec[68] + 18 * bvec[69] + 18 * bvec[70] - 18 * bvec[71] + 12 * bvec[112] + 6 * bvec[113] - 12 * bvec[114] - 6 * bvec[115] - 12 * bvec[116] - 6 * bvec[117] + 12 * bvec[118] + 6 * bvec[119] + 12 * bvec[144] - 12 * bvec[145] + 6 * bvec[146] - 6 * bvec[147] - 12 * bvec[148] + 12 * bvec[149] - 6 * bvec[150] + 6 * bvec[151] + 9 * bvec[160] - 9 * bvec[161] - 9 * bvec[162] + 9 * bvec[163] + 9 * bvec[164] - 9 * bvec[165] - 9 * bvec[166] + 9 * bvec[167] + 8 * bvec[192] + 4 * bvec[193] + 4 * bvec[194] + 2 * bvec[195] - 8 * bvec[196] - 4 * bvec[197] - 4 * bvec[198] - 2 * bvec[199] + 6 * bvec[208] + 3 * bvec[209] - 6 * bvec[210] - 3 * bvec[211] + 6 * bvec[212] + 3 * bvec[213] - 6 * bvec[214] - 3 * bvec[215] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[226] - 3 * bvec[227] + 6 * bvec[228] - 6 * bvec[229] + 3 * bvec[230] - 3 * bvec[231] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[242] + 1 * bvec[243] + 4 * bvec[244] + 2 * bvec[245] + 2 * bvec[246] + 1 * bvec[247];
       alphavec[123] = -12 * bvec[64] + 12 * bvec[65] + 12 * bvec[66] - 12 * bvec[67] + 12 * bvec[68] - 12 * bvec[69] - 12 * bvec[70] + 12 * bvec[71] - 6 * bvec[112] - 6 * bvec[113] + 6 * bvec[114] + 6 * bvec[115] + 6 * bvec[116] + 6 * bvec[117] - 6 * bvec[118] - 6 * bvec[119] - 8 * bvec[144] + 8 * bvec[145] - 4 * bvec[146] + 4 * bvec[147] + 8 * bvec[148] - 8 * bvec[149] + 4 * bvec[150] - 4 * bvec[151] - 6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 6 * bvec[164] + 6 * bvec[165] + 6 * bvec[166] - 6 * bvec[167] - 4 * bvec[192] - 4 * bvec[193] - 2 * bvec[194] - 2 * bvec[195] + 4 * bvec[196] + 4 * bvec[197] + 2 * bvec[198] + 2 * bvec[199] - 3 * bvec[208] - 3 * bvec[209] + 3 * bvec[210] + 3 * bvec[211] - 3 * bvec[212] - 3 * bvec[213] + 3 * bvec[214] + 3 * bvec[215] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[226] + 2 * bvec[227] - 4 * bvec[228] + 4 * bvec[229] - 2 * bvec[230] + 2 * bvec[231] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[242] - 1 * bvec[243] - 2 * bvec[244] - 2 * bvec[245] - 1 * bvec[246] - 1 * bvec[247];
       alphavec[124] = +4 * bvec[64] - 4 * bvec[66] - 4 * bvec[68] + 4 * bvec[70] + 2 * bvec[144] + 2 * bvec[146] - 2 * bvec[148] - 2 * bvec[150] + 2 * bvec[160] - 2 * bvec[162] + 2 * bvec[164] - 2 * bvec[166] + 1 * bvec[224] + 1 * bvec[226] + 1 * bvec[228] + 1 * bvec[230];
       alphavec[125] = +4 * bvec[112] - 4 * bvec[114] - 4 * bvec[116] + 4 * bvec[118] + 2 * bvec[192] + 2 * bvec[194] - 2 * bvec[196] - 2 * bvec[198] + 2 * bvec[208] - 2 * bvec[210] + 2 * bvec[212] - 2 * bvec[214] + 1 * bvec[240] + 1 * bvec[242] + 1 * bvec[244] + 1 * bvec[246];
       alphavec[126] = -12 * bvec[64] + 12 * bvec[65] + 12 * bvec[66] - 12 * bvec[67] + 12 * bvec[68] - 12 * bvec[69] - 12 * bvec[70] + 12 * bvec[71] - 8 * bvec[112] - 4 * bvec[113] + 8 * bvec[114] + 4 * bvec[115] + 8 * bvec[116] + 4 * bvec[117] - 8 * bvec[118] - 4 * bvec[119] - 6 * bvec[144] + 6 * bvec[145] - 6 * bvec[146] + 6 * bvec[147] + 6 * bvec[148] - 6 * bvec[149] + 6 * bvec[150] - 6 * bvec[151] - 6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 6 * bvec[164] + 6 * bvec[165] + 6 * bvec[166] - 6 * bvec[167] - 4 * bvec[192] - 2 * bvec[193] - 4 * bvec[194] - 2 * bvec[195] + 4 * bvec[196] + 2 * bvec[197] + 4 * bvec[198] + 2 * bvec[199] - 4 * bvec[208] - 2 * bvec[209] + 4 * bvec[210] + 2 * bvec[211] - 4 * bvec[212] - 2 * bvec[213] + 4 * bvec[214] + 2 * bvec[215] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[226] + 3 * bvec[227] - 3 * bvec[228] + 3 * bvec[229] - 3 * bvec[230] + 3 * bvec[231] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[242] - 1 * bvec[243] - 2 * bvec[244] - 1 * bvec[245] - 2 * bvec[246] - 1 * bvec[247];
       alphavec[127] = +8 * bvec[64] - 8 * bvec[65] - 8 * bvec[66] + 8 * bvec[67] - 8 * bvec[68] + 8 * bvec[69] + 8 * bvec[70] - 8 * bvec[71] + 4 * bvec[112] + 4 * bvec[113] - 4 * bvec[114] - 4 * bvec[115] - 4 * bvec[116] - 4 * bvec[117] + 4 * bvec[118] + 4 * bvec[119] + 4 * bvec[144] - 4 * bvec[145] + 4 * bvec[146] - 4 * bvec[147] - 4 * bvec[148] + 4 * bvec[149] - 4 * bvec[150] + 4 * bvec[151] + 4 * bvec[160] - 4 * bvec[161] - 4 * bvec[162] + 4 * bvec[163] + 4 * bvec[164] - 4 * bvec[165] - 4 * bvec[166] + 4 * bvec[167] + 2 * bvec[192] + 2 * bvec[193] + 2 * bvec[194] + 2 * bvec[195] - 2 * bvec[196] - 2 * bvec[197] - 2 * bvec[198] - 2 * bvec[199] + 2 * bvec[208] + 2 * bvec[209] - 2 * bvec[210] - 2 * bvec[211] + 2 * bvec[212] + 2 * bvec[213] - 2 * bvec[214] - 2 * bvec[215] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[226] - 2 * bvec[227] + 2 * bvec[228] - 2 * bvec[229] + 2 * bvec[230] - 2 * bvec[231] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[242] + 1 * bvec[243] + 1 * bvec[244] + 1 * bvec[245] + 1 * bvec[246] + 1 * bvec[247];
       alphavec[128] = -3 * bvec[0] + 3 * bvec[8] - 2 * bvec[64] - 1 * bvec[72];
       alphavec[129] = -3 * bvec[16] + 3 * bvec[24] - 2 * bvec[112] - 1 * bvec[120];
       alphavec[130] = +9 * bvec[0] - 9 * bvec[1] - 9 * bvec[8] + 9 * bvec[9] + 6 * bvec[16] + 3 * bvec[17] - 6 * bvec[24] - 3 * bvec[25] + 6 * bvec[64] - 6 * bvec[65] + 3 * bvec[72] - 3 * bvec[73] + 4 * bvec[112] + 2 * bvec[113] + 2 * bvec[120] + 1 * bvec[121];
       alphavec[131] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[8] - 6 * bvec[9] - 3 * bvec[16] - 3 * bvec[17] + 3 * bvec[24] + 3 * bvec[25] - 4 * bvec[64] + 4 * bvec[65] - 2 * bvec[72] + 2 * bvec[73] - 2 * bvec[112] - 2 * bvec[113] - 1 * bvec[120] - 1 * bvec[121];
       alphavec[132] = -3 * bvec[32] + 3 * bvec[40] - 2 * bvec[144] - 1 * bvec[152];
       alphavec[133] = -3 * bvec[80] + 3 * bvec[88] - 2 * bvec[192] - 1 * bvec[200];
       alphavec[134] = +9 * bvec[32] - 9 * bvec[33] - 9 * bvec[40] + 9 * bvec[41] + 6 * bvec[80] + 3 * bvec[81] - 6 * bvec[88] - 3 * bvec[89] + 6 * bvec[144] - 6 * bvec[145] + 3 * bvec[152] - 3 * bvec[153] + 4 * bvec[192] + 2 * bvec[193] + 2 * bvec[200] + 1 * bvec[201];
       alphavec[135] = -6 * bvec[32] + 6 * bvec[33] + 6 * bvec[40] - 6 * bvec[41] - 3 * bvec[80] - 3 * bvec[81] + 3 * bvec[88] + 3 * bvec[89] - 4 * bvec[144] + 4 * bvec[145] - 2 * bvec[152] + 2 * bvec[153] - 2 * bvec[192] - 2 * bvec[193] - 1 * bvec[200] - 1 * bvec[201];
       alphavec[136] = +9 * bvec[0] - 9 * bvec[2] - 9 * bvec[8] + 9 * bvec[10] + 6 * bvec[32] + 3 * bvec[34] - 6 * bvec[40] - 3 * bvec[42] + 6 * bvec[64] - 6 * bvec[66] + 3 * bvec[72] - 3 * bvec[74] + 4 * bvec[144] + 2 * bvec[146] + 2 * bvec[152] + 1 * bvec[154];
       alphavec[137] = +9 * bvec[16] - 9 * bvec[18] - 9 * bvec[24] + 9 * bvec[26] + 6 * bvec[80] + 3 * bvec[82] - 6 * bvec[88] - 3 * bvec[90] + 6 * bvec[112] - 6 * bvec[114] + 3 * bvec[120] - 3 * bvec[122] + 4 * bvec[192] + 2 * bvec[194] + 2 * bvec[200] + 1 * bvec[202];
       alphavec[138] = -27 * bvec[0] + 27 * bvec[1] + 27 * bvec[2] - 27 * bvec[3] + 27 * bvec[8] - 27 * bvec[9] - 27 * bvec[10] + 27 * bvec[11] - 18 * bvec[16] - 9 * bvec[17] + 18 * bvec[18] + 9 * bvec[19] + 18 * bvec[24] + 9 * bvec[25] - 18 * bvec[26] - 9 * bvec[27] - 18 * bvec[32] + 18 * bvec[33] - 9 * bvec[34] + 9 * bvec[35] + 18 * bvec[40] - 18 * bvec[41] + 9 * bvec[42] - 9 * bvec[43] - 18 * bvec[64] + 18 * bvec[65] + 18 * bvec[66] - 18 * bvec[67] - 9 * bvec[72] + 9 * bvec[73] + 9 * bvec[74] - 9 * bvec[75] - 12 * bvec[80] - 6 * bvec[81] - 6 * bvec[82] - 3 * bvec[83] + 12 * bvec[88] + 6 * bvec[89] + 6 * bvec[90] + 3 * bvec[91] - 12 * bvec[112] - 6 * bvec[113] + 12 * bvec[114] + 6 * bvec[115] - 6 * bvec[120] - 3 * bvec[121] + 6 * bvec[122] + 3 * bvec[123] - 12 * bvec[144] + 12 * bvec[145] - 6 * bvec[146] + 6 * bvec[147] - 6 * bvec[152] + 6 * bvec[153] - 3 * bvec[154] + 3 * bvec[155] - 8 * bvec[192] - 4 * bvec[193] - 4 * bvec[194] - 2 * bvec[195] - 4 * bvec[200] - 2 * bvec[201] - 2 * bvec[202] - 1 * bvec[203];
       alphavec[139] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[8] + 18 * bvec[9] + 18 * bvec[10] - 18 * bvec[11] + 9 * bvec[16] + 9 * bvec[17] - 9 * bvec[18] - 9 * bvec[19] - 9 * bvec[24] - 9 * bvec[25] + 9 * bvec[26] + 9 * bvec[27] + 12 * bvec[32] - 12 * bvec[33] + 6 * bvec[34] - 6 * bvec[35] - 12 * bvec[40] + 12 * bvec[41] - 6 * bvec[42] + 6 * bvec[43] + 12 * bvec[64] - 12 * bvec[65] - 12 * bvec[66] + 12 * bvec[67] + 6 * bvec[72] - 6 * bvec[73] - 6 * bvec[74] + 6 * bvec[75] + 6 * bvec[80] + 6 * bvec[81] + 3 * bvec[82] + 3 * bvec[83] - 6 * bvec[88] - 6 * bvec[89] - 3 * bvec[90] - 3 * bvec[91] + 6 * bvec[112] + 6 * bvec[113] - 6 * bvec[114] - 6 * bvec[115] + 3 * bvec[120] + 3 * bvec[121] - 3 * bvec[122] - 3 * bvec[123] + 8 * bvec[144] - 8 * bvec[145] + 4 * bvec[146] - 4 * bvec[147] + 4 * bvec[152] - 4 * bvec[153] + 2 * bvec[154] - 2 * bvec[155] + 4 * bvec[192] + 4 * bvec[193] + 2 * bvec[194] + 2 * bvec[195] + 2 * bvec[200] + 2 * bvec[201] + 1 * bvec[202] + 1 * bvec[203];
       alphavec[140] = -6 * bvec[0] + 6 * bvec[2] + 6 * bvec[8] - 6 * bvec[10] - 3 * bvec[32] - 3 * bvec[34] + 3 * bvec[40] + 3 * bvec[42] - 4 * bvec[64] + 4 * bvec[66] - 2 * bvec[72] + 2 * bvec[74] - 2 * bvec[144] - 2 * bvec[146] - 1 * bvec[152] - 1 * bvec[154];
       alphavec[141] = -6 * bvec[16] + 6 * bvec[18] + 6 * bvec[24] - 6 * bvec[26] - 3 * bvec[80] - 3 * bvec[82] + 3 * bvec[88] + 3 * bvec[90] - 4 * bvec[112] + 4 * bvec[114] - 2 * bvec[120] + 2 * bvec[122] - 2 * bvec[192] - 2 * bvec[194] - 1 * bvec[200] - 1 * bvec[202];
       alphavec[142] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[8] + 18 * bvec[9] + 18 * bvec[10] - 18 * bvec[11] + 12 * bvec[16] + 6 * bvec[17] - 12 * bvec[18] - 6 * bvec[19] - 12 * bvec[24] - 6 * bvec[25] + 12 * bvec[26] + 6 * bvec[27] + 9 * bvec[32] - 9 * bvec[33] + 9 * bvec[34] - 9 * bvec[35] - 9 * bvec[40] + 9 * bvec[41] - 9 * bvec[42] + 9 * bvec[43] + 12 * bvec[64] - 12 * bvec[65] - 12 * bvec[66] + 12 * bvec[67] + 6 * bvec[72] - 6 * bvec[73] - 6 * bvec[74] + 6 * bvec[75] + 6 * bvec[80] + 3 * bvec[81] + 6 * bvec[82] + 3 * bvec[83] - 6 * bvec[88] - 3 * bvec[89] - 6 * bvec[90] - 3 * bvec[91] + 8 * bvec[112] + 4 * bvec[113] - 8 * bvec[114] - 4 * bvec[115] + 4 * bvec[120] + 2 * bvec[121] - 4 * bvec[122] - 2 * bvec[123] + 6 * bvec[144] - 6 * bvec[145] + 6 * bvec[146] - 6 * bvec[147] + 3 * bvec[152] - 3 * bvec[153] + 3 * bvec[154] - 3 * bvec[155] + 4 * bvec[192] + 2 * bvec[193] + 4 * bvec[194] + 2 * bvec[195] + 2 * bvec[200] + 1 * bvec[201] + 2 * bvec[202] + 1 * bvec[203];
       alphavec[143] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[8] - 12 * bvec[9] - 12 * bvec[10] + 12 * bvec[11] - 6 * bvec[16] - 6 * bvec[17] + 6 * bvec[18] + 6 * bvec[19] + 6 * bvec[24] + 6 * bvec[25] - 6 * bvec[26] - 6 * bvec[27] - 6 * bvec[32] + 6 * bvec[33] - 6 * bvec[34] + 6 * bvec[35] + 6 * bvec[40] - 6 * bvec[41] + 6 * bvec[42] - 6 * bvec[43] - 8 * bvec[64] + 8 * bvec[65] + 8 * bvec[66] - 8 * bvec[67] - 4 * bvec[72] + 4 * bvec[73] + 4 * bvec[74] - 4 * bvec[75] - 3 * bvec[80] - 3 * bvec[81] - 3 * bvec[82] - 3 * bvec[83] + 3 * bvec[88] + 3 * bvec[89] + 3 * bvec[90] + 3 * bvec[91] - 4 * bvec[112] - 4 * bvec[113] + 4 * bvec[114] + 4 * bvec[115] - 2 * bvec[120] - 2 * bvec[121] + 2 * bvec[122] + 2 * bvec[123] - 4 * bvec[144] + 4 * bvec[145] - 4 * bvec[146] + 4 * bvec[147] - 2 * bvec[152] + 2 * bvec[153] - 2 * bvec[154] + 2 * bvec[155] - 2 * bvec[192] - 2 * bvec[193] - 2 * bvec[194] - 2 * bvec[195] - 1 * bvec[200] - 1 * bvec[201] - 1 * bvec[202] - 1 * bvec[203];
       alphavec[144] = -3 * bvec[48] + 3 * bvec[56] - 2 * bvec[160] - 1 * bvec[168];
       alphavec[145] = -3 * bvec[96] + 3 * bvec[104] - 2 * bvec[208] - 1 * bvec[216];
       alphavec[146] = +9 * bvec[48] - 9 * bvec[49] - 9 * bvec[56] + 9 * bvec[57] + 6 * bvec[96] + 3 * bvec[97] - 6 * bvec[104] - 3 * bvec[105] + 6 * bvec[160] - 6 * bvec[161] + 3 * bvec[168] - 3 * bvec[169] + 4 * bvec[208] + 2 * bvec[209] + 2 * bvec[216] + 1 * bvec[217];
       alphavec[147] = -6 * bvec[48] + 6 * bvec[49] + 6 * bvec[56] - 6 * bvec[57] - 3 * bvec[96] - 3 * bvec[97] + 3 * bvec[104] + 3 * bvec[105] - 4 * bvec[160] + 4 * bvec[161] - 2 * bvec[168] + 2 * bvec[169] - 2 * bvec[208] - 2 * bvec[209] - 1 * bvec[216] - 1 * bvec[217];
       alphavec[148] = -3 * bvec[128] + 3 * bvec[136] - 2 * bvec[224] - 1 * bvec[232];
       alphavec[149] = -3 * bvec[176] + 3 * bvec[184] - 2 * bvec[240] - 1 * bvec[248];
       alphavec[150] = +9 * bvec[128] - 9 * bvec[129] - 9 * bvec[136] + 9 * bvec[137] + 6 * bvec[176] + 3 * bvec[177] - 6 * bvec[184] - 3 * bvec[185] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[232] - 3 * bvec[233] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[248] + 1 * bvec[249];
       alphavec[151] = -6 * bvec[128] + 6 * bvec[129] + 6 * bvec[136] - 6 * bvec[137] - 3 * bvec[176] - 3 * bvec[177] + 3 * bvec[184] + 3 * bvec[185] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[232] + 2 * bvec[233] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[248] - 1 * bvec[249];
       alphavec[152] = +9 * bvec[48] - 9 * bvec[50] - 9 * bvec[56] + 9 * bvec[58] + 6 * bvec[128] + 3 * bvec[130] - 6 * bvec[136] - 3 * bvec[138] + 6 * bvec[160] - 6 * bvec[162] + 3 * bvec[168] - 3 * bvec[170] + 4 * bvec[224] + 2 * bvec[226] + 2 * bvec[232] + 1 * bvec[234];
       alphavec[153] = +9 * bvec[96] - 9 * bvec[98] - 9 * bvec[104] + 9 * bvec[106] + 6 * bvec[176] + 3 * bvec[178] - 6 * bvec[184] - 3 * bvec[186] + 6 * bvec[208] - 6 * bvec[210] + 3 * bvec[216] - 3 * bvec[218] + 4 * bvec[240] + 2 * bvec[242] + 2 * bvec[248] + 1 * bvec[250];
       alphavec[154] = -27 * bvec[48] + 27 * bvec[49] + 27 * bvec[50] - 27 * bvec[51] + 27 * bvec[56] - 27 * bvec[57] - 27 * bvec[58] + 27 * bvec[59] - 18 * bvec[96] - 9 * bvec[97] + 18 * bvec[98] + 9 * bvec[99] + 18 * bvec[104] + 9 * bvec[105] - 18 * bvec[106] - 9 * bvec[107] - 18 * bvec[128] + 18 * bvec[129] - 9 * bvec[130] + 9 * bvec[131] + 18 * bvec[136] - 18 * bvec[137] + 9 * bvec[138] - 9 * bvec[139] - 18 * bvec[160] + 18 * bvec[161] + 18 * bvec[162] - 18 * bvec[163] - 9 * bvec[168] + 9 * bvec[169] + 9 * bvec[170] - 9 * bvec[171] - 12 * bvec[176] - 6 * bvec[177] - 6 * bvec[178] - 3 * bvec[179] + 12 * bvec[184] + 6 * bvec[185] + 6 * bvec[186] + 3 * bvec[187] - 12 * bvec[208] - 6 * bvec[209] + 12 * bvec[210] + 6 * bvec[211] - 6 * bvec[216] - 3 * bvec[217] + 6 * bvec[218] + 3 * bvec[219] - 12 * bvec[224] + 12 * bvec[225] - 6 * bvec[226] + 6 * bvec[227] - 6 * bvec[232] + 6 * bvec[233] - 3 * bvec[234] + 3 * bvec[235] - 8 * bvec[240] - 4 * bvec[241] - 4 * bvec[242] - 2 * bvec[243] - 4 * bvec[248] - 2 * bvec[249] - 2 * bvec[250] - 1 * bvec[251];
       alphavec[155] = +18 * bvec[48] - 18 * bvec[49] - 18 * bvec[50] + 18 * bvec[51] - 18 * bvec[56] + 18 * bvec[57] + 18 * bvec[58] - 18 * bvec[59] + 9 * bvec[96] + 9 * bvec[97] - 9 * bvec[98] - 9 * bvec[99] - 9 * bvec[104] - 9 * bvec[105] + 9 * bvec[106] + 9 * bvec[107] + 12 * bvec[128] - 12 * bvec[129] + 6 * bvec[130] - 6 * bvec[131] - 12 * bvec[136] + 12 * bvec[137] - 6 * bvec[138] + 6 * bvec[139] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 6 * bvec[168] - 6 * bvec[169] - 6 * bvec[170] + 6 * bvec[171] + 6 * bvec[176] + 6 * bvec[177] + 3 * bvec[178] + 3 * bvec[179] - 6 * bvec[184] - 6 * bvec[185] - 3 * bvec[186] - 3 * bvec[187] + 6 * bvec[208] + 6 * bvec[209] - 6 * bvec[210] - 6 * bvec[211] + 3 * bvec[216] + 3 * bvec[217] - 3 * bvec[218] - 3 * bvec[219] + 8 * bvec[224] - 8 * bvec[225] + 4 * bvec[226] - 4 * bvec[227] + 4 * bvec[232] - 4 * bvec[233] + 2 * bvec[234] - 2 * bvec[235] + 4 * bvec[240] + 4 * bvec[241] + 2 * bvec[242] + 2 * bvec[243] + 2 * bvec[248] + 2 * bvec[249] + 1 * bvec[250] + 1 * bvec[251];
       alphavec[156] = -6 * bvec[48] + 6 * bvec[50] + 6 * bvec[56] - 6 * bvec[58] - 3 * bvec[128] - 3 * bvec[130] + 3 * bvec[136] + 3 * bvec[138] - 4 * bvec[160] + 4 * bvec[162] - 2 * bvec[168] + 2 * bvec[170] - 2 * bvec[224] - 2 * bvec[226] - 1 * bvec[232] - 1 * bvec[234];
       alphavec[157] = -6 * bvec[96] + 6 * bvec[98] + 6 * bvec[104] - 6 * bvec[106] - 3 * bvec[176] - 3 * bvec[178] + 3 * bvec[184] + 3 * bvec[186] - 4 * bvec[208] + 4 * bvec[210] - 2 * bvec[216] + 2 * bvec[218] - 2 * bvec[240] - 2 * bvec[242] - 1 * bvec[248] - 1 * bvec[250];
       alphavec[158] = +18 * bvec[48] - 18 * bvec[49] - 18 * bvec[50] + 18 * bvec[51] - 18 * bvec[56] + 18 * bvec[57] + 18 * bvec[58] - 18 * bvec[59] + 12 * bvec[96] + 6 * bvec[97] - 12 * bvec[98] - 6 * bvec[99] - 12 * bvec[104] - 6 * bvec[105] + 12 * bvec[106] + 6 * bvec[107] + 9 * bvec[128] - 9 * bvec[129] + 9 * bvec[130] - 9 * bvec[131] - 9 * bvec[136] + 9 * bvec[137] - 9 * bvec[138] + 9 * bvec[139] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 6 * bvec[168] - 6 * bvec[169] - 6 * bvec[170] + 6 * bvec[171] + 6 * bvec[176] + 3 * bvec[177] + 6 * bvec[178] + 3 * bvec[179] - 6 * bvec[184] - 3 * bvec[185] - 6 * bvec[186] - 3 * bvec[187] + 8 * bvec[208] + 4 * bvec[209] - 8 * bvec[210] - 4 * bvec[211] + 4 * bvec[216] + 2 * bvec[217] - 4 * bvec[218] - 2 * bvec[219] + 6 * bvec[224] - 6 * bvec[225] + 6 * bvec[226] - 6 * bvec[227] + 3 * bvec[232] - 3 * bvec[233] + 3 * bvec[234] - 3 * bvec[235] + 4 * bvec[240] + 2 * bvec[241] + 4 * bvec[242] + 2 * bvec[243] + 2 * bvec[248] + 1 * bvec[249] + 2 * bvec[250] + 1 * bvec[251];
       alphavec[159] = -12 * bvec[48] + 12 * bvec[49] + 12 * bvec[50] - 12 * bvec[51] + 12 * bvec[56] - 12 * bvec[57] - 12 * bvec[58] + 12 * bvec[59] - 6 * bvec[96] - 6 * bvec[97] + 6 * bvec[98] + 6 * bvec[99] + 6 * bvec[104] + 6 * bvec[105] - 6 * bvec[106] - 6 * bvec[107] - 6 * bvec[128] + 6 * bvec[129] - 6 * bvec[130] + 6 * bvec[131] + 6 * bvec[136] - 6 * bvec[137] + 6 * bvec[138] - 6 * bvec[139] - 8 * bvec[160] + 8 * bvec[161] + 8 * bvec[162] - 8 * bvec[163] - 4 * bvec[168] + 4 * bvec[169] + 4 * bvec[170] - 4 * bvec[171] - 3 * bvec[176] - 3 * bvec[177] - 3 * bvec[178] - 3 * bvec[179] + 3 * bvec[184] + 3 * bvec[185] + 3 * bvec[186] + 3 * bvec[187] - 4 * bvec[208] - 4 * bvec[209] + 4 * bvec[210] + 4 * bvec[211] - 2 * bvec[216] - 2 * bvec[217] + 2 * bvec[218] + 2 * bvec[219] - 4 * bvec[224] + 4 * bvec[225] - 4 * bvec[226] + 4 * bvec[227] - 2 * bvec[232] + 2 * bvec[233] - 2 * bvec[234] + 2 * bvec[235] - 2 * bvec[240] - 2 * bvec[241] - 2 * bvec[242] - 2 * bvec[243] - 1 * bvec[248] - 1 * bvec[249] - 1 * bvec[250] - 1 * bvec[251];
       alphavec[160] = +9 * bvec[0] - 9 * bvec[4] - 9 * bvec[8] + 9 * bvec[12] + 6 * bvec[48] + 3 * bvec[52] - 6 * bvec[56] - 3 * bvec[60] + 6 * bvec[64] - 6 * bvec[68] + 3 * bvec[72] - 3 * bvec[76] + 4 * bvec[160] + 2 * bvec[164] + 2 * bvec[168] + 1 * bvec[172];
       alphavec[161] = +9 * bvec[16] - 9 * bvec[20] - 9 * bvec[24] + 9 * bvec[28] + 6 * bvec[96] + 3 * bvec[100] - 6 * bvec[104] - 3 * bvec[108] + 6 * bvec[112] - 6 * bvec[116] + 3 * bvec[120] - 3 * bvec[124] + 4 * bvec[208] + 2 * bvec[212] + 2 * bvec[216] + 1 * bvec[220];
       alphavec[162] = -27 * bvec[0] + 27 * bvec[1] + 27 * bvec[4] - 27 * bvec[5] + 27 * bvec[8] - 27 * bvec[9] - 27 * bvec[12] + 27 * bvec[13] - 18 * bvec[16] - 9 * bvec[17] + 18 * bvec[20] + 9 * bvec[21] + 18 * bvec[24] + 9 * bvec[25] - 18 * bvec[28] - 9 * bvec[29] - 18 * bvec[48] + 18 * bvec[49] - 9 * bvec[52] + 9 * bvec[53] + 18 * bvec[56] - 18 * bvec[57] + 9 * bvec[60] - 9 * bvec[61] - 18 * bvec[64] + 18 * bvec[65] + 18 * bvec[68] - 18 * bvec[69] - 9 * bvec[72] + 9 * bvec[73] + 9 * bvec[76] - 9 * bvec[77] - 12 * bvec[96] - 6 * bvec[97] - 6 * bvec[100] - 3 * bvec[101] + 12 * bvec[104] + 6 * bvec[105] + 6 * bvec[108] + 3 * bvec[109] - 12 * bvec[112] - 6 * bvec[113] + 12 * bvec[116] + 6 * bvec[117] - 6 * bvec[120] - 3 * bvec[121] + 6 * bvec[124] + 3 * bvec[125] - 12 * bvec[160] + 12 * bvec[161] - 6 * bvec[164] + 6 * bvec[165] - 6 * bvec[168] + 6 * bvec[169] - 3 * bvec[172] + 3 * bvec[173] - 8 * bvec[208] - 4 * bvec[209] - 4 * bvec[212] - 2 * bvec[213] - 4 * bvec[216] - 2 * bvec[217] - 2 * bvec[220] - 1 * bvec[221];
       alphavec[163] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[4] + 18 * bvec[5] - 18 * bvec[8] + 18 * bvec[9] + 18 * bvec[12] - 18 * bvec[13] + 9 * bvec[16] + 9 * bvec[17] - 9 * bvec[20] - 9 * bvec[21] - 9 * bvec[24] - 9 * bvec[25] + 9 * bvec[28] + 9 * bvec[29] + 12 * bvec[48] - 12 * bvec[49] + 6 * bvec[52] - 6 * bvec[53] - 12 * bvec[56] + 12 * bvec[57] - 6 * bvec[60] + 6 * bvec[61] + 12 * bvec[64] - 12 * bvec[65] - 12 * bvec[68] + 12 * bvec[69] + 6 * bvec[72] - 6 * bvec[73] - 6 * bvec[76] + 6 * bvec[77] + 6 * bvec[96] + 6 * bvec[97] + 3 * bvec[100] + 3 * bvec[101] - 6 * bvec[104] - 6 * bvec[105] - 3 * bvec[108] - 3 * bvec[109] + 6 * bvec[112] + 6 * bvec[113] - 6 * bvec[116] - 6 * bvec[117] + 3 * bvec[120] + 3 * bvec[121] - 3 * bvec[124] - 3 * bvec[125] + 8 * bvec[160] - 8 * bvec[161] + 4 * bvec[164] - 4 * bvec[165] + 4 * bvec[168] - 4 * bvec[169] + 2 * bvec[172] - 2 * bvec[173] + 4 * bvec[208] + 4 * bvec[209] + 2 * bvec[212] + 2 * bvec[213] + 2 * bvec[216] + 2 * bvec[217] + 1 * bvec[220] + 1 * bvec[221];
       alphavec[164] = +9 * bvec[32] - 9 * bvec[36] - 9 * bvec[40] + 9 * bvec[44] + 6 * bvec[128] + 3 * bvec[132] - 6 * bvec[136] - 3 * bvec[140] + 6 * bvec[144] - 6 * bvec[148] + 3 * bvec[152] - 3 * bvec[156] + 4 * bvec[224] + 2 * bvec[228] + 2 * bvec[232] + 1 * bvec[236];
       alphavec[165] = +9 * bvec[80] - 9 * bvec[84] - 9 * bvec[88] + 9 * bvec[92] + 6 * bvec[176] + 3 * bvec[180] - 6 * bvec[184] - 3 * bvec[188] + 6 * bvec[192] - 6 * bvec[196] + 3 * bvec[200] - 3 * bvec[204] + 4 * bvec[240] + 2 * bvec[244] + 2 * bvec[248] + 1 * bvec[252];
       alphavec[166] = -27 * bvec[32] + 27 * bvec[33] + 27 * bvec[36] - 27 * bvec[37] + 27 * bvec[40] - 27 * bvec[41] - 27 * bvec[44] + 27 * bvec[45] - 18 * bvec[80] - 9 * bvec[81] + 18 * bvec[84] + 9 * bvec[85] + 18 * bvec[88] + 9 * bvec[89] - 18 * bvec[92] - 9 * bvec[93] - 18 * bvec[128] + 18 * bvec[129] - 9 * bvec[132] + 9 * bvec[133] + 18 * bvec[136] - 18 * bvec[137] + 9 * bvec[140] - 9 * bvec[141] - 18 * bvec[144] + 18 * bvec[145] + 18 * bvec[148] - 18 * bvec[149] - 9 * bvec[152] + 9 * bvec[153] + 9 * bvec[156] - 9 * bvec[157] - 12 * bvec[176] - 6 * bvec[177] - 6 * bvec[180] - 3 * bvec[181] + 12 * bvec[184] + 6 * bvec[185] + 6 * bvec[188] + 3 * bvec[189] - 12 * bvec[192] - 6 * bvec[193] + 12 * bvec[196] + 6 * bvec[197] - 6 * bvec[200] - 3 * bvec[201] + 6 * bvec[204] + 3 * bvec[205] - 12 * bvec[224] + 12 * bvec[225] - 6 * bvec[228] + 6 * bvec[229] - 6 * bvec[232] + 6 * bvec[233] - 3 * bvec[236] + 3 * bvec[237] - 8 * bvec[240] - 4 * bvec[241] - 4 * bvec[244] - 2 * bvec[245] - 4 * bvec[248] - 2 * bvec[249] - 2 * bvec[252] - 1 * bvec[253];
       alphavec[167] = +18 * bvec[32] - 18 * bvec[33] - 18 * bvec[36] + 18 * bvec[37] - 18 * bvec[40] + 18 * bvec[41] + 18 * bvec[44] - 18 * bvec[45] + 9 * bvec[80] + 9 * bvec[81] - 9 * bvec[84] - 9 * bvec[85] - 9 * bvec[88] - 9 * bvec[89] + 9 * bvec[92] + 9 * bvec[93] + 12 * bvec[128] - 12 * bvec[129] + 6 * bvec[132] - 6 * bvec[133] - 12 * bvec[136] + 12 * bvec[137] - 6 * bvec[140] + 6 * bvec[141] + 12 * bvec[144] - 12 * bvec[145] - 12 * bvec[148] + 12 * bvec[149] + 6 * bvec[152] - 6 * bvec[153] - 6 * bvec[156] + 6 * bvec[157] + 6 * bvec[176] + 6 * bvec[177] + 3 * bvec[180] + 3 * bvec[181] - 6 * bvec[184] - 6 * bvec[185] - 3 * bvec[188] - 3 * bvec[189] + 6 * bvec[192] + 6 * bvec[193] - 6 * bvec[196] - 6 * bvec[197] + 3 * bvec[200] + 3 * bvec[201] - 3 * bvec[204] - 3 * bvec[205] + 8 * bvec[224] - 8 * bvec[225] + 4 * bvec[228] - 4 * bvec[229] + 4 * bvec[232] - 4 * bvec[233] + 2 * bvec[236] - 2 * bvec[237] + 4 * bvec[240] + 4 * bvec[241] + 2 * bvec[244] + 2 * bvec[245] + 2 * bvec[248] + 2 * bvec[249] + 1 * bvec[252] + 1 * bvec[253];
       alphavec[168] = -27 * bvec[0] + 27 * bvec[2] + 27 * bvec[4] - 27 * bvec[6] + 27 * bvec[8] - 27 * bvec[10] - 27 * bvec[12] + 27 * bvec[14] - 18 * bvec[32] - 9 * bvec[34] + 18 * bvec[36] + 9 * bvec[38] + 18 * bvec[40] + 9 * bvec[42] - 18 * bvec[44] - 9 * bvec[46] - 18 * bvec[48] + 18 * bvec[50] - 9 * bvec[52] + 9 * bvec[54] + 18 * bvec[56] - 18 * bvec[58] + 9 * bvec[60] - 9 * bvec[62] - 18 * bvec[64] + 18 * bvec[66] + 18 * bvec[68] - 18 * bvec[70] - 9 * bvec[72] + 9 * bvec[74] + 9 * bvec[76] - 9 * bvec[78] - 12 * bvec[128] - 6 * bvec[130] - 6 * bvec[132] - 3 * bvec[134] + 12 * bvec[136] + 6 * bvec[138] + 6 * bvec[140] + 3 * bvec[142] - 12 * bvec[144] - 6 * bvec[146] + 12 * bvec[148] + 6 * bvec[150] - 6 * bvec[152] - 3 * bvec[154] + 6 * bvec[156] + 3 * bvec[158] - 12 * bvec[160] + 12 * bvec[162] - 6 * bvec[164] + 6 * bvec[166] - 6 * bvec[168] + 6 * bvec[170] - 3 * bvec[172] + 3 * bvec[174] - 8 * bvec[224] - 4 * bvec[226] - 4 * bvec[228] - 2 * bvec[230] - 4 * bvec[232] - 2 * bvec[234] - 2 * bvec[236] - 1 * bvec[238];
       alphavec[169] = -27 * bvec[16] + 27 * bvec[18] + 27 * bvec[20] - 27 * bvec[22] + 27 * bvec[24] - 27 * bvec[26] - 27 * bvec[28] + 27 * bvec[30] - 18 * bvec[80] - 9 * bvec[82] + 18 * bvec[84] + 9 * bvec[86] + 18 * bvec[88] + 9 * bvec[90] - 18 * bvec[92] - 9 * bvec[94] - 18 * bvec[96] + 18 * bvec[98] - 9 * bvec[100] + 9 * bvec[102] + 18 * bvec[104] - 18 * bvec[106] + 9 * bvec[108] - 9 * bvec[110] - 18 * bvec[112] + 18 * bvec[114] + 18 * bvec[116] - 18 * bvec[118] - 9 * bvec[120] + 9 * bvec[122] + 9 * bvec[124] - 9 * bvec[126] - 12 * bvec[176] - 6 * bvec[178] - 6 * bvec[180] - 3 * bvec[182] + 12 * bvec[184] + 6 * bvec[186] + 6 * bvec[188] + 3 * bvec[190] - 12 * bvec[192] - 6 * bvec[194] + 12 * bvec[196] + 6 * bvec[198] - 6 * bvec[200] - 3 * bvec[202] + 6 * bvec[204] + 3 * bvec[206] - 12 * bvec[208] + 12 * bvec[210] - 6 * bvec[212] + 6 * bvec[214] - 6 * bvec[216] + 6 * bvec[218] - 3 * bvec[220] + 3 * bvec[222] - 8 * bvec[240] - 4 * bvec[242] - 4 * bvec[244] - 2 * bvec[246] - 4 * bvec[248] - 2 * bvec[250] - 2 * bvec[252] - 1 * bvec[254];
       alphavec[170] = +81 * bvec[0] - 81 * bvec[1] - 81 * bvec[2] + 81 * bvec[3] - 81 * bvec[4] + 81 * bvec[5] + 81 * bvec[6] - 81 * bvec[7] - 81 * bvec[8] + 81 * bvec[9] + 81 * bvec[10] - 81 * bvec[11] + 81 * bvec[12] - 81 * bvec[13] - 81 * bvec[14] + 81 * bvec[15] + 54 * bvec[16] + 27 * bvec[17] - 54 * bvec[18] - 27 * bvec[19] - 54 * bvec[20] - 27 * bvec[21] + 54 * bvec[22] + 27 * bvec[23] - 54 * bvec[24] - 27 * bvec[25] + 54 * bvec[26] + 27 * bvec[27] + 54 * bvec[28] + 27 * bvec[29] - 54 * bvec[30] - 27 * bvec[31] + 54 * bvec[32] - 54 * bvec[33] + 27 * bvec[34] - 27 * bvec[35] - 54 * bvec[36] + 54 * bvec[37] - 27 * bvec[38] + 27 * bvec[39] - 54 * bvec[40] + 54 * bvec[41] - 27 * bvec[42] + 27 * bvec[43] + 54 * bvec[44] - 54 * bvec[45] + 27 * bvec[46] - 27 * bvec[47] + 54 * bvec[48] - 54 * bvec[49] - 54 * bvec[50] + 54 * bvec[51] + 27 * bvec[52] - 27 * bvec[53] - 27 * bvec[54] + 27 * bvec[55] - 54 * bvec[56] + 54 * bvec[57] + 54 * bvec[58] - 54 * bvec[59] - 27 * bvec[60] + 27 * bvec[61] + 27 * bvec[62] - 27 * bvec[63] + 54 * bvec[64] - 54 * bvec[65] - 54 * bvec[66] + 54 * bvec[67] - 54 * bvec[68] + 54 * bvec[69] + 54 * bvec[70] - 54 * bvec[71] + 27 * bvec[72] - 27 * bvec[73] - 27 * bvec[74] + 27 * bvec[75] - 27 * bvec[76] + 27 * bvec[77] + 27 * bvec[78] - 27 * bvec[79] + 36 * bvec[80] + 18 * bvec[81] + 18 * bvec[82] + 9 * bvec[83] - 36 * bvec[84] - 18 * bvec[85] - 18 * bvec[86] - 9 * bvec[87] - 36 * bvec[88] - 18 * bvec[89] - 18 * bvec[90] - 9 * bvec[91] + 36 * bvec[92] + 18 * bvec[93] + 18 * bvec[94] + 9 * bvec[95] + 36 * bvec[96] + 18 * bvec[97] - 36 * bvec[98] - 18 * bvec[99] + 18 * bvec[100] + 9 * bvec[101] - 18 * bvec[102] - 9 * bvec[103] - 36 * bvec[104] - 18 * bvec[105] + 36 * bvec[106] + 18 * bvec[107] - 18 * bvec[108] - 9 * bvec[109] + 18 * bvec[110] + 9 * bvec[111] + 36 * bvec[112] + 18 * bvec[113] - 36 * bvec[114] - 18 * bvec[115] - 36 * bvec[116] - 18 * bvec[117] + 36 * bvec[118] + 18 * bvec[119] + 18 * bvec[120] + 9 * bvec[121] - 18 * bvec[122] - 9 * bvec[123] - 18 * bvec[124] - 9 * bvec[125] + 18 * bvec[126] + 9 * bvec[127] + 36 * bvec[128] - 36 * bvec[129] + 18 * bvec[130] - 18 * bvec[131] + 18 * bvec[132] - 18 * bvec[133] + 9 * bvec[134] - 9 * bvec[135] - 36 * bvec[136] + 36 * bvec[137] - 18 * bvec[138] + 18 * bvec[139] - 18 * bvec[140] + 18 * bvec[141] - 9 * bvec[142] + 9 * bvec[143] + 36 * bvec[144] - 36 * bvec[145] + 18 * bvec[146] - 18 * bvec[147] - 36 * bvec[148] + 36 * bvec[149] - 18 * bvec[150] + 18 * bvec[151] + 18 * bvec[152] - 18 * bvec[153] + 9 * bvec[154] - 9 * bvec[155] - 18 * bvec[156] + 18 * bvec[157] - 9 * bvec[158] + 9 * bvec[159] + 36 * bvec[160] - 36 * bvec[161] - 36 * bvec[162] + 36 * bvec[163] + 18 * bvec[164] - 18 * bvec[165] - 18 * bvec[166] + 18 * bvec[167] + 18 * bvec[168] - 18 * bvec[169] - 18 * bvec[170] + 18 * bvec[171] + 9 * bvec[172] - 9 * bvec[173] - 9 * bvec[174] + 9 * bvec[175] + 24 * bvec[176] + 12 * bvec[177] + 12 * bvec[178] + 6 * bvec[179] + 12 * bvec[180] + 6 * bvec[181] + 6 * bvec[182] + 3 * bvec[183] - 24 * bvec[184] - 12 * bvec[185] - 12 * bvec[186] - 6 * bvec[187] - 12 * bvec[188] - 6 * bvec[189] - 6 * bvec[190] - 3 * bvec[191] + 24 * bvec[192] + 12 * bvec[193] + 12 * bvec[194] + 6 * bvec[195] - 24 * bvec[196] - 12 * bvec[197] - 12 * bvec[198] - 6 * bvec[199] + 12 * bvec[200] + 6 * bvec[201] + 6 * bvec[202] + 3 * bvec[203] - 12 * bvec[204] - 6 * bvec[205] - 6 * bvec[206] - 3 * bvec[207] + 24 * bvec[208] + 12 * bvec[209] - 24 * bvec[210] - 12 * bvec[211] + 12 * bvec[212] + 6 * bvec[213] - 12 * bvec[214] - 6 * bvec[215] + 12 * bvec[216] + 6 * bvec[217] - 12 * bvec[218] - 6 * bvec[219] + 6 * bvec[220] + 3 * bvec[221] - 6 * bvec[222] - 3 * bvec[223] + 24 * bvec[224] - 24 * bvec[225] + 12 * bvec[226] - 12 * bvec[227] + 12 * bvec[228] - 12 * bvec[229] + 6 * bvec[230] - 6 * bvec[231] + 12 * bvec[232] - 12 * bvec[233] + 6 * bvec[234] - 6 * bvec[235] + 6 * bvec[236] - 6 * bvec[237] + 3 * bvec[238] - 3 * bvec[239] + 16 * bvec[240] + 8 * bvec[241] + 8 * bvec[242] + 4 * bvec[243] + 8 * bvec[244] + 4 * bvec[245] + 4 * bvec[246] + 2 * bvec[247] + 8 * bvec[248] + 4 * bvec[249] + 4 * bvec[250] + 2 * bvec[251] + 4 * bvec[252] + 2 * bvec[253] + 2 * bvec[254] + 1 * bvec[255];
       alphavec[171] = -54 * bvec[0] + 54 * bvec[1] + 54 * bvec[2] - 54 * bvec[3] + 54 * bvec[4] - 54 * bvec[5] - 54 * bvec[6] + 54 * bvec[7] + 54 * bvec[8] - 54 * bvec[9] - 54 * bvec[10] + 54 * bvec[11] - 54 * bvec[12] + 54 * bvec[13] + 54 * bvec[14] - 54 * bvec[15] - 27 * bvec[16] - 27 * bvec[17] + 27 * bvec[18] + 27 * bvec[19] + 27 * bvec[20] + 27 * bvec[21] - 27 * bvec[22] - 27 * bvec[23] + 27 * bvec[24] + 27 * bvec[25] - 27 * bvec[26] - 27 * bvec[27] - 27 * bvec[28] - 27 * bvec[29] + 27 * bvec[30] + 27 * bvec[31] - 36 * bvec[32] + 36 * bvec[33] - 18 * bvec[34] + 18 * bvec[35] + 36 * bvec[36] - 36 * bvec[37] + 18 * bvec[38] - 18 * bvec[39] + 36 * bvec[40] - 36 * bvec[41] + 18 * bvec[42] - 18 * bvec[43] - 36 * bvec[44] + 36 * bvec[45] - 18 * bvec[46] + 18 * bvec[47] - 36 * bvec[48] + 36 * bvec[49] + 36 * bvec[50] - 36 * bvec[51] - 18 * bvec[52] + 18 * bvec[53] + 18 * bvec[54] - 18 * bvec[55] + 36 * bvec[56] - 36 * bvec[57] - 36 * bvec[58] + 36 * bvec[59] + 18 * bvec[60] - 18 * bvec[61] - 18 * bvec[62] + 18 * bvec[63] - 36 * bvec[64] + 36 * bvec[65] + 36 * bvec[66] - 36 * bvec[67] + 36 * bvec[68] - 36 * bvec[69] - 36 * bvec[70] + 36 * bvec[71] - 18 * bvec[72] + 18 * bvec[73] + 18 * bvec[74] - 18 * bvec[75] + 18 * bvec[76] - 18 * bvec[77] - 18 * bvec[78] + 18 * bvec[79] - 18 * bvec[80] - 18 * bvec[81] - 9 * bvec[82] - 9 * bvec[83] + 18 * bvec[84] + 18 * bvec[85] + 9 * bvec[86] + 9 * bvec[87] + 18 * bvec[88] + 18 * bvec[89] + 9 * bvec[90] + 9 * bvec[91] - 18 * bvec[92] - 18 * bvec[93] - 9 * bvec[94] - 9 * bvec[95] - 18 * bvec[96] - 18 * bvec[97] + 18 * bvec[98] + 18 * bvec[99] - 9 * bvec[100] - 9 * bvec[101] + 9 * bvec[102] + 9 * bvec[103] + 18 * bvec[104] + 18 * bvec[105] - 18 * bvec[106] - 18 * bvec[107] + 9 * bvec[108] + 9 * bvec[109] - 9 * bvec[110] - 9 * bvec[111] - 18 * bvec[112] - 18 * bvec[113] + 18 * bvec[114] + 18 * bvec[115] + 18 * bvec[116] + 18 * bvec[117] - 18 * bvec[118] - 18 * bvec[119] - 9 * bvec[120] - 9 * bvec[121] + 9 * bvec[122] + 9 * bvec[123] + 9 * bvec[124] + 9 * bvec[125] - 9 * bvec[126] - 9 * bvec[127] - 24 * bvec[128] + 24 * bvec[129] - 12 * bvec[130] + 12 * bvec[131] - 12 * bvec[132] + 12 * bvec[133] - 6 * bvec[134] + 6 * bvec[135] + 24 * bvec[136] - 24 * bvec[137] + 12 * bvec[138] - 12 * bvec[139] + 12 * bvec[140] - 12 * bvec[141] + 6 * bvec[142] - 6 * bvec[143] - 24 * bvec[144] + 24 * bvec[145] - 12 * bvec[146] + 12 * bvec[147] + 24 * bvec[148] - 24 * bvec[149] + 12 * bvec[150] - 12 * bvec[151] - 12 * bvec[152] + 12 * bvec[153] - 6 * bvec[154] + 6 * bvec[155] + 12 * bvec[156] - 12 * bvec[157] + 6 * bvec[158] - 6 * bvec[159] - 24 * bvec[160] + 24 * bvec[161] + 24 * bvec[162] - 24 * bvec[163] - 12 * bvec[164] + 12 * bvec[165] + 12 * bvec[166] - 12 * bvec[167] - 12 * bvec[168] + 12 * bvec[169] + 12 * bvec[170] - 12 * bvec[171] - 6 * bvec[172] + 6 * bvec[173] + 6 * bvec[174] - 6 * bvec[175] - 12 * bvec[176] - 12 * bvec[177] - 6 * bvec[178] - 6 * bvec[179] - 6 * bvec[180] - 6 * bvec[181] - 3 * bvec[182] - 3 * bvec[183] + 12 * bvec[184] + 12 * bvec[185] + 6 * bvec[186] + 6 * bvec[187] + 6 * bvec[188] + 6 * bvec[189] + 3 * bvec[190] + 3 * bvec[191] - 12 * bvec[192] - 12 * bvec[193] - 6 * bvec[194] - 6 * bvec[195] + 12 * bvec[196] + 12 * bvec[197] + 6 * bvec[198] + 6 * bvec[199] - 6 * bvec[200] - 6 * bvec[201] - 3 * bvec[202] - 3 * bvec[203] + 6 * bvec[204] + 6 * bvec[205] + 3 * bvec[206] + 3 * bvec[207] - 12 * bvec[208] - 12 * bvec[209] + 12 * bvec[210] + 12 * bvec[211] - 6 * bvec[212] - 6 * bvec[213] + 6 * bvec[214] + 6 * bvec[215] - 6 * bvec[216] - 6 * bvec[217] + 6 * bvec[218] + 6 * bvec[219] - 3 * bvec[220] - 3 * bvec[221] + 3 * bvec[222] + 3 * bvec[223] - 16 * bvec[224] + 16 * bvec[225] - 8 * bvec[226] + 8 * bvec[227] - 8 * bvec[228] + 8 * bvec[229] - 4 * bvec[230] + 4 * bvec[231] - 8 * bvec[232] + 8 * bvec[233] - 4 * bvec[234] + 4 * bvec[235] - 4 * bvec[236] + 4 * bvec[237] - 2 * bvec[238] + 2 * bvec[239] - 8 * bvec[240] - 8 * bvec[241] - 4 * bvec[242] - 4 * bvec[243] - 4 * bvec[244] - 4 * bvec[245] - 2 * bvec[246] - 2 * bvec[247] - 4 * bvec[248] - 4 * bvec[249] - 2 * bvec[250] - 2 * bvec[251] - 2 * bvec[252] - 2 * bvec[253] - 1 * bvec[254] - 1 * bvec[255];
       alphavec[172] = +18 * bvec[0] - 18 * bvec[2] - 18 * bvec[4] + 18 * bvec[6] - 18 * bvec[8] + 18 * bvec[10] + 18 * bvec[12] - 18 * bvec[14] + 9 * bvec[32] + 9 * bvec[34] - 9 * bvec[36] - 9 * bvec[38] - 9 * bvec[40] - 9 * bvec[42] + 9 * bvec[44] + 9 * bvec[46] + 12 * bvec[48] - 12 * bvec[50] + 6 * bvec[52] - 6 * bvec[54] - 12 * bvec[56] + 12 * bvec[58] - 6 * bvec[60] + 6 * bvec[62] + 12 * bvec[64] - 12 * bvec[66] - 12 * bvec[68] + 12 * bvec[70] + 6 * bvec[72] - 6 * bvec[74] - 6 * bvec[76] + 6 * bvec[78] + 6 * bvec[128] + 6 * bvec[130] + 3 * bvec[132] + 3 * bvec[134] - 6 * bvec[136] - 6 * bvec[138] - 3 * bvec[140] - 3 * bvec[142] + 6 * bvec[144] + 6 * bvec[146] - 6 * bvec[148] - 6 * bvec[150] + 3 * bvec[152] + 3 * bvec[154] - 3 * bvec[156] - 3 * bvec[158] + 8 * bvec[160] - 8 * bvec[162] + 4 * bvec[164] - 4 * bvec[166] + 4 * bvec[168] - 4 * bvec[170] + 2 * bvec[172] - 2 * bvec[174] + 4 * bvec[224] + 4 * bvec[226] + 2 * bvec[228] + 2 * bvec[230] + 2 * bvec[232] + 2 * bvec[234] + 1 * bvec[236] + 1 * bvec[238];
       alphavec[173] = +18 * bvec[16] - 18 * bvec[18] - 18 * bvec[20] + 18 * bvec[22] - 18 * bvec[24] + 18 * bvec[26] + 18 * bvec[28] - 18 * bvec[30] + 9 * bvec[80] + 9 * bvec[82] - 9 * bvec[84] - 9 * bvec[86] - 9 * bvec[88] - 9 * bvec[90] + 9 * bvec[92] + 9 * bvec[94] + 12 * bvec[96] - 12 * bvec[98] + 6 * bvec[100] - 6 * bvec[102] - 12 * bvec[104] + 12 * bvec[106] - 6 * bvec[108] + 6 * bvec[110] + 12 * bvec[112] - 12 * bvec[114] - 12 * bvec[116] + 12 * bvec[118] + 6 * bvec[120] - 6 * bvec[122] - 6 * bvec[124] + 6 * bvec[126] + 6 * bvec[176] + 6 * bvec[178] + 3 * bvec[180] + 3 * bvec[182] - 6 * bvec[184] - 6 * bvec[186] - 3 * bvec[188] - 3 * bvec[190] + 6 * bvec[192] + 6 * bvec[194] - 6 * bvec[196] - 6 * bvec[198] + 3 * bvec[200] + 3 * bvec[202] - 3 * bvec[204] - 3 * bvec[206] + 8 * bvec[208] - 8 * bvec[210] + 4 * bvec[212] - 4 * bvec[214] + 4 * bvec[216] - 4 * bvec[218] + 2 * bvec[220] - 2 * bvec[222] + 4 * bvec[240] + 4 * bvec[242] + 2 * bvec[244] + 2 * bvec[246] + 2 * bvec[248] + 2 * bvec[250] + 1 * bvec[252] + 1 * bvec[254];
       alphavec[174] = -54 * bvec[0] + 54 * bvec[1] + 54 * bvec[2] - 54 * bvec[3] + 54 * bvec[4] - 54 * bvec[5] - 54 * bvec[6] + 54 * bvec[7] + 54 * bvec[8] - 54 * bvec[9] - 54 * bvec[10] + 54 * bvec[11] - 54 * bvec[12] + 54 * bvec[13] + 54 * bvec[14] - 54 * bvec[15] - 36 * bvec[16] - 18 * bvec[17] + 36 * bvec[18] + 18 * bvec[19] + 36 * bvec[20] + 18 * bvec[21] - 36 * bvec[22] - 18 * bvec[23] + 36 * bvec[24] + 18 * bvec[25] - 36 * bvec[26] - 18 * bvec[27] - 36 * bvec[28] - 18 * bvec[29] + 36 * bvec[30] + 18 * bvec[31] - 27 * bvec[32] + 27 * bvec[33] - 27 * bvec[34] + 27 * bvec[35] + 27 * bvec[36] - 27 * bvec[37] + 27 * bvec[38] - 27 * bvec[39] + 27 * bvec[40] - 27 * bvec[41] + 27 * bvec[42] - 27 * bvec[43] - 27 * bvec[44] + 27 * bvec[45] - 27 * bvec[46] + 27 * bvec[47] - 36 * bvec[48] + 36 * bvec[49] + 36 * bvec[50] - 36 * bvec[51] - 18 * bvec[52] + 18 * bvec[53] + 18 * bvec[54] - 18 * bvec[55] + 36 * bvec[56] - 36 * bvec[57] - 36 * bvec[58] + 36 * bvec[59] + 18 * bvec[60] - 18 * bvec[61] - 18 * bvec[62] + 18 * bvec[63] - 36 * bvec[64] + 36 * bvec[65] + 36 * bvec[66] - 36 * bvec[67] + 36 * bvec[68] - 36 * bvec[69] - 36 * bvec[70] + 36 * bvec[71] - 18 * bvec[72] + 18 * bvec[73] + 18 * bvec[74] - 18 * bvec[75] + 18 * bvec[76] - 18 * bvec[77] - 18 * bvec[78] + 18 * bvec[79] - 18 * bvec[80] - 9 * bvec[81] - 18 * bvec[82] - 9 * bvec[83] + 18 * bvec[84] + 9 * bvec[85] + 18 * bvec[86] + 9 * bvec[87] + 18 * bvec[88] + 9 * bvec[89] + 18 * bvec[90] + 9 * bvec[91] - 18 * bvec[92] - 9 * bvec[93] - 18 * bvec[94] - 9 * bvec[95] - 24 * bvec[96] - 12 * bvec[97] + 24 * bvec[98] + 12 * bvec[99] - 12 * bvec[100] - 6 * bvec[101] + 12 * bvec[102] + 6 * bvec[103] + 24 * bvec[104] + 12 * bvec[105] - 24 * bvec[106] - 12 * bvec[107] + 12 * bvec[108] + 6 * bvec[109] - 12 * bvec[110] - 6 * bvec[111] - 24 * bvec[112] - 12 * bvec[113] + 24 * bvec[114] + 12 * bvec[115] + 24 * bvec[116] + 12 * bvec[117] - 24 * bvec[118] - 12 * bvec[119] - 12 * bvec[120] - 6 * bvec[121] + 12 * bvec[122] + 6 * bvec[123] + 12 * bvec[124] + 6 * bvec[125] - 12 * bvec[126] - 6 * bvec[127] - 18 * bvec[128] + 18 * bvec[129] - 18 * bvec[130] + 18 * bvec[131] - 9 * bvec[132] + 9 * bvec[133] - 9 * bvec[134] + 9 * bvec[135] + 18 * bvec[136] - 18 * bvec[137] + 18 * bvec[138] - 18 * bvec[139] + 9 * bvec[140] - 9 * bvec[141] + 9 * bvec[142] - 9 * bvec[143] - 18 * bvec[144] + 18 * bvec[145] - 18 * bvec[146] + 18 * bvec[147] + 18 * bvec[148] - 18 * bvec[149] + 18 * bvec[150] - 18 * bvec[151] - 9 * bvec[152] + 9 * bvec[153] - 9 * bvec[154] + 9 * bvec[155] + 9 * bvec[156] - 9 * bvec[157] + 9 * bvec[158] - 9 * bvec[159] - 24 * bvec[160] + 24 * bvec[161] + 24 * bvec[162] - 24 * bvec[163] - 12 * bvec[164] + 12 * bvec[165] + 12 * bvec[166] - 12 * bvec[167] - 12 * bvec[168] + 12 * bvec[169] + 12 * bvec[170] - 12 * bvec[171] - 6 * bvec[172] + 6 * bvec[173] + 6 * bvec[174] - 6 * bvec[175] - 12 * bvec[176] - 6 * bvec[177] - 12 * bvec[178] - 6 * bvec[179] - 6 * bvec[180] - 3 * bvec[181] - 6 * bvec[182] - 3 * bvec[183] + 12 * bvec[184] + 6 * bvec[185] + 12 * bvec[186] + 6 * bvec[187] + 6 * bvec[188] + 3 * bvec[189] + 6 * bvec[190] + 3 * bvec[191] - 12 * bvec[192] - 6 * bvec[193] - 12 * bvec[194] - 6 * bvec[195] + 12 * bvec[196] + 6 * bvec[197] + 12 * bvec[198] + 6 * bvec[199] - 6 * bvec[200] - 3 * bvec[201] - 6 * bvec[202] - 3 * bvec[203] + 6 * bvec[204] + 3 * bvec[205] + 6 * bvec[206] + 3 * bvec[207] - 16 * bvec[208] - 8 * bvec[209] + 16 * bvec[210] + 8 * bvec[211] - 8 * bvec[212] - 4 * bvec[213] + 8 * bvec[214] + 4 * bvec[215] - 8 * bvec[216] - 4 * bvec[217] + 8 * bvec[218] + 4 * bvec[219] - 4 * bvec[220] - 2 * bvec[221] + 4 * bvec[222] + 2 * bvec[223] - 12 * bvec[224] + 12 * bvec[225] - 12 * bvec[226] + 12 * bvec[227] - 6 * bvec[228] + 6 * bvec[229] - 6 * bvec[230] + 6 * bvec[231] - 6 * bvec[232] + 6 * bvec[233] - 6 * bvec[234] + 6 * bvec[235] - 3 * bvec[236] + 3 * bvec[237] - 3 * bvec[238] + 3 * bvec[239] - 8 * bvec[240] - 4 * bvec[241] - 8 * bvec[242] - 4 * bvec[243] - 4 * bvec[244] - 2 * bvec[245] - 4 * bvec[246] - 2 * bvec[247] - 4 * bvec[248] - 2 * bvec[249] - 4 * bvec[250] - 2 * bvec[251] - 2 * bvec[252] - 1 * bvec[253] - 2 * bvec[254] - 1 * bvec[255];
       alphavec[175] = +36 * bvec[0] - 36 * bvec[1] - 36 * bvec[2] + 36 * bvec[3] - 36 * bvec[4] + 36 * bvec[5] + 36 * bvec[6] - 36 * bvec[7] - 36 * bvec[8] + 36 * bvec[9] + 36 * bvec[10] - 36 * bvec[11] + 36 * bvec[12] - 36 * bvec[13] - 36 * bvec[14] + 36 * bvec[15] + 18 * bvec[16] + 18 * bvec[17] - 18 * bvec[18] - 18 * bvec[19] - 18 * bvec[20] - 18 * bvec[21] + 18 * bvec[22] + 18 * bvec[23] - 18 * bvec[24] - 18 * bvec[25] + 18 * bvec[26] + 18 * bvec[27] + 18 * bvec[28] + 18 * bvec[29] - 18 * bvec[30] - 18 * bvec[31] + 18 * bvec[32] - 18 * bvec[33] + 18 * bvec[34] - 18 * bvec[35] - 18 * bvec[36] + 18 * bvec[37] - 18 * bvec[38] + 18 * bvec[39] - 18 * bvec[40] + 18 * bvec[41] - 18 * bvec[42] + 18 * bvec[43] + 18 * bvec[44] - 18 * bvec[45] + 18 * bvec[46] - 18 * bvec[47] + 24 * bvec[48] - 24 * bvec[49] - 24 * bvec[50] + 24 * bvec[51] + 12 * bvec[52] - 12 * bvec[53] - 12 * bvec[54] + 12 * bvec[55] - 24 * bvec[56] + 24 * bvec[57] + 24 * bvec[58] - 24 * bvec[59] - 12 * bvec[60] + 12 * bvec[61] + 12 * bvec[62] - 12 * bvec[63] + 24 * bvec[64] - 24 * bvec[65] - 24 * bvec[66] + 24 * bvec[67] - 24 * bvec[68] + 24 * bvec[69] + 24 * bvec[70] - 24 * bvec[71] + 12 * bvec[72] - 12 * bvec[73] - 12 * bvec[74] + 12 * bvec[75] - 12 * bvec[76] + 12 * bvec[77] + 12 * bvec[78] - 12 * bvec[79] + 9 * bvec[80] + 9 * bvec[81] + 9 * bvec[82] + 9 * bvec[83] - 9 * bvec[84] - 9 * bvec[85] - 9 * bvec[86] - 9 * bvec[87] - 9 * bvec[88] - 9 * bvec[89] - 9 * bvec[90] - 9 * bvec[91] + 9 * bvec[92] + 9 * bvec[93] + 9 * bvec[94] + 9 * bvec[95] + 12 * bvec[96] + 12 * bvec[97] - 12 * bvec[98] - 12 * bvec[99] + 6 * bvec[100] + 6 * bvec[101] - 6 * bvec[102] - 6 * bvec[103] - 12 * bvec[104] - 12 * bvec[105] + 12 * bvec[106] + 12 * bvec[107] - 6 * bvec[108] - 6 * bvec[109] + 6 * bvec[110] + 6 * bvec[111] + 12 * bvec[112] + 12 * bvec[113] - 12 * bvec[114] - 12 * bvec[115] - 12 * bvec[116] - 12 * bvec[117] + 12 * bvec[118] + 12 * bvec[119] + 6 * bvec[120] + 6 * bvec[121] - 6 * bvec[122] - 6 * bvec[123] - 6 * bvec[124] - 6 * bvec[125] + 6 * bvec[126] + 6 * bvec[127] + 12 * bvec[128] - 12 * bvec[129] + 12 * bvec[130] - 12 * bvec[131] + 6 * bvec[132] - 6 * bvec[133] + 6 * bvec[134] - 6 * bvec[135] - 12 * bvec[136] + 12 * bvec[137] - 12 * bvec[138] + 12 * bvec[139] - 6 * bvec[140] + 6 * bvec[141] - 6 * bvec[142] + 6 * bvec[143] + 12 * bvec[144] - 12 * bvec[145] + 12 * bvec[146] - 12 * bvec[147] - 12 * bvec[148] + 12 * bvec[149] - 12 * bvec[150] + 12 * bvec[151] + 6 * bvec[152] - 6 * bvec[153] + 6 * bvec[154] - 6 * bvec[155] - 6 * bvec[156] + 6 * bvec[157] - 6 * bvec[158] + 6 * bvec[159] + 16 * bvec[160] - 16 * bvec[161] - 16 * bvec[162] + 16 * bvec[163] + 8 * bvec[164] - 8 * bvec[165] - 8 * bvec[166] + 8 * bvec[167] + 8 * bvec[168] - 8 * bvec[169] - 8 * bvec[170] + 8 * bvec[171] + 4 * bvec[172] - 4 * bvec[173] - 4 * bvec[174] + 4 * bvec[175] + 6 * bvec[176] + 6 * bvec[177] + 6 * bvec[178] + 6 * bvec[179] + 3 * bvec[180] + 3 * bvec[181] + 3 * bvec[182] + 3 * bvec[183] - 6 * bvec[184] - 6 * bvec[185] - 6 * bvec[186] - 6 * bvec[187] - 3 * bvec[188] - 3 * bvec[189] - 3 * bvec[190] - 3 * bvec[191] + 6 * bvec[192] + 6 * bvec[193] + 6 * bvec[194] + 6 * bvec[195] - 6 * bvec[196] - 6 * bvec[197] - 6 * bvec[198] - 6 * bvec[199] + 3 * bvec[200] + 3 * bvec[201] + 3 * bvec[202] + 3 * bvec[203] - 3 * bvec[204] - 3 * bvec[205] - 3 * bvec[206] - 3 * bvec[207] + 8 * bvec[208] + 8 * bvec[209] - 8 * bvec[210] - 8 * bvec[211] + 4 * bvec[212] + 4 * bvec[213] - 4 * bvec[214] - 4 * bvec[215] + 4 * bvec[216] + 4 * bvec[217] - 4 * bvec[218] - 4 * bvec[219] + 2 * bvec[220] + 2 * bvec[221] - 2 * bvec[222] - 2 * bvec[223] + 8 * bvec[224] - 8 * bvec[225] + 8 * bvec[226] - 8 * bvec[227] + 4 * bvec[228] - 4 * bvec[229] + 4 * bvec[230] - 4 * bvec[231] + 4 * bvec[232] - 4 * bvec[233] + 4 * bvec[234] - 4 * bvec[235] + 2 * bvec[236] - 2 * bvec[237] + 2 * bvec[238] - 2 * bvec[239] + 4 * bvec[240] + 4 * bvec[241] + 4 * bvec[242] + 4 * bvec[243] + 2 * bvec[244] + 2 * bvec[245] + 2 * bvec[246] + 2 * bvec[247] + 2 * bvec[248] + 2 * bvec[249] + 2 * bvec[250] + 2 * bvec[251] + 1 * bvec[252] + 1 * bvec[253] + 1 * bvec[254] + 1 * bvec[255];
       alphavec[176] = -6 * bvec[0] + 6 * bvec[4] + 6 * bvec[8] - 6 * bvec[12] - 3 * bvec[48] - 3 * bvec[52] + 3 * bvec[56] + 3 * bvec[60] - 4 * bvec[64] + 4 * bvec[68] - 2 * bvec[72] + 2 * bvec[76] - 2 * bvec[160] - 2 * bvec[164] - 1 * bvec[168] - 1 * bvec[172];
       alphavec[177] = -6 * bvec[16] + 6 * bvec[20] + 6 * bvec[24] - 6 * bvec[28] - 3 * bvec[96] - 3 * bvec[100] + 3 * bvec[104] + 3 * bvec[108] - 4 * bvec[112] + 4 * bvec[116] - 2 * bvec[120] + 2 * bvec[124] - 2 * bvec[208] - 2 * bvec[212] - 1 * bvec[216] - 1 * bvec[220];
       alphavec[178] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[4] + 18 * bvec[5] - 18 * bvec[8] + 18 * bvec[9] + 18 * bvec[12] - 18 * bvec[13] + 12 * bvec[16] + 6 * bvec[17] - 12 * bvec[20] - 6 * bvec[21] - 12 * bvec[24] - 6 * bvec[25] + 12 * bvec[28] + 6 * bvec[29] + 9 * bvec[48] - 9 * bvec[49] + 9 * bvec[52] - 9 * bvec[53] - 9 * bvec[56] + 9 * bvec[57] - 9 * bvec[60] + 9 * bvec[61] + 12 * bvec[64] - 12 * bvec[65] - 12 * bvec[68] + 12 * bvec[69] + 6 * bvec[72] - 6 * bvec[73] - 6 * bvec[76] + 6 * bvec[77] + 6 * bvec[96] + 3 * bvec[97] + 6 * bvec[100] + 3 * bvec[101] - 6 * bvec[104] - 3 * bvec[105] - 6 * bvec[108] - 3 * bvec[109] + 8 * bvec[112] + 4 * bvec[113] - 8 * bvec[116] - 4 * bvec[117] + 4 * bvec[120] + 2 * bvec[121] - 4 * bvec[124] - 2 * bvec[125] + 6 * bvec[160] - 6 * bvec[161] + 6 * bvec[164] - 6 * bvec[165] + 3 * bvec[168] - 3 * bvec[169] + 3 * bvec[172] - 3 * bvec[173] + 4 * bvec[208] + 2 * bvec[209] + 4 * bvec[212] + 2 * bvec[213] + 2 * bvec[216] + 1 * bvec[217] + 2 * bvec[220] + 1 * bvec[221];
       alphavec[179] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[4] - 12 * bvec[5] + 12 * bvec[8] - 12 * bvec[9] - 12 * bvec[12] + 12 * bvec[13] - 6 * bvec[16] - 6 * bvec[17] + 6 * bvec[20] + 6 * bvec[21] + 6 * bvec[24] + 6 * bvec[25] - 6 * bvec[28] - 6 * bvec[29] - 6 * bvec[48] + 6 * bvec[49] - 6 * bvec[52] + 6 * bvec[53] + 6 * bvec[56] - 6 * bvec[57] + 6 * bvec[60] - 6 * bvec[61] - 8 * bvec[64] + 8 * bvec[65] + 8 * bvec[68] - 8 * bvec[69] - 4 * bvec[72] + 4 * bvec[73] + 4 * bvec[76] - 4 * bvec[77] - 3 * bvec[96] - 3 * bvec[97] - 3 * bvec[100] - 3 * bvec[101] + 3 * bvec[104] + 3 * bvec[105] + 3 * bvec[108] + 3 * bvec[109] - 4 * bvec[112] - 4 * bvec[113] + 4 * bvec[116] + 4 * bvec[117] - 2 * bvec[120] - 2 * bvec[121] + 2 * bvec[124] + 2 * bvec[125] - 4 * bvec[160] + 4 * bvec[161] - 4 * bvec[164] + 4 * bvec[165] - 2 * bvec[168] + 2 * bvec[169] - 2 * bvec[172] + 2 * bvec[173] - 2 * bvec[208] - 2 * bvec[209] - 2 * bvec[212] - 2 * bvec[213] - 1 * bvec[216] - 1 * bvec[217] - 1 * bvec[220] - 1 * bvec[221];
       alphavec[180] = -6 * bvec[32] + 6 * bvec[36] + 6 * bvec[40] - 6 * bvec[44] - 3 * bvec[128] - 3 * bvec[132] + 3 * bvec[136] + 3 * bvec[140] - 4 * bvec[144] + 4 * bvec[148] - 2 * bvec[152] + 2 * bvec[156] - 2 * bvec[224] - 2 * bvec[228] - 1 * bvec[232] - 1 * bvec[236];
       alphavec[181] = -6 * bvec[80] + 6 * bvec[84] + 6 * bvec[88] - 6 * bvec[92] - 3 * bvec[176] - 3 * bvec[180] + 3 * bvec[184] + 3 * bvec[188] - 4 * bvec[192] + 4 * bvec[196] - 2 * bvec[200] + 2 * bvec[204] - 2 * bvec[240] - 2 * bvec[244] - 1 * bvec[248] - 1 * bvec[252];
       alphavec[182] = +18 * bvec[32] - 18 * bvec[33] - 18 * bvec[36] + 18 * bvec[37] - 18 * bvec[40] + 18 * bvec[41] + 18 * bvec[44] - 18 * bvec[45] + 12 * bvec[80] + 6 * bvec[81] - 12 * bvec[84] - 6 * bvec[85] - 12 * bvec[88] - 6 * bvec[89] + 12 * bvec[92] + 6 * bvec[93] + 9 * bvec[128] - 9 * bvec[129] + 9 * bvec[132] - 9 * bvec[133] - 9 * bvec[136] + 9 * bvec[137] - 9 * bvec[140] + 9 * bvec[141] + 12 * bvec[144] - 12 * bvec[145] - 12 * bvec[148] + 12 * bvec[149] + 6 * bvec[152] - 6 * bvec[153] - 6 * bvec[156] + 6 * bvec[157] + 6 * bvec[176] + 3 * bvec[177] + 6 * bvec[180] + 3 * bvec[181] - 6 * bvec[184] - 3 * bvec[185] - 6 * bvec[188] - 3 * bvec[189] + 8 * bvec[192] + 4 * bvec[193] - 8 * bvec[196] - 4 * bvec[197] + 4 * bvec[200] + 2 * bvec[201] - 4 * bvec[204] - 2 * bvec[205] + 6 * bvec[224] - 6 * bvec[225] + 6 * bvec[228] - 6 * bvec[229] + 3 * bvec[232] - 3 * bvec[233] + 3 * bvec[236] - 3 * bvec[237] + 4 * bvec[240] + 2 * bvec[241] + 4 * bvec[244] + 2 * bvec[245] + 2 * bvec[248] + 1 * bvec[249] + 2 * bvec[252] + 1 * bvec[253];
       alphavec[183] = -12 * bvec[32] + 12 * bvec[33] + 12 * bvec[36] - 12 * bvec[37] + 12 * bvec[40] - 12 * bvec[41] - 12 * bvec[44] + 12 * bvec[45] - 6 * bvec[80] - 6 * bvec[81] + 6 * bvec[84] + 6 * bvec[85] + 6 * bvec[88] + 6 * bvec[89] - 6 * bvec[92] - 6 * bvec[93] - 6 * bvec[128] + 6 * bvec[129] - 6 * bvec[132] + 6 * bvec[133] + 6 * bvec[136] - 6 * bvec[137] + 6 * bvec[140] - 6 * bvec[141] - 8 * bvec[144] + 8 * bvec[145] + 8 * bvec[148] - 8 * bvec[149] - 4 * bvec[152] + 4 * bvec[153] + 4 * bvec[156] - 4 * bvec[157] - 3 * bvec[176] - 3 * bvec[177] - 3 * bvec[180] - 3 * bvec[181] + 3 * bvec[184] + 3 * bvec[185] + 3 * bvec[188] + 3 * bvec[189] - 4 * bvec[192] - 4 * bvec[193] + 4 * bvec[196] + 4 * bvec[197] - 2 * bvec[200] - 2 * bvec[201] + 2 * bvec[204] + 2 * bvec[205] - 4 * bvec[224] + 4 * bvec[225] - 4 * bvec[228] + 4 * bvec[229] - 2 * bvec[232] + 2 * bvec[233] - 2 * bvec[236] + 2 * bvec[237] - 2 * bvec[240] - 2 * bvec[241] - 2 * bvec[244] - 2 * bvec[245] - 1 * bvec[248] - 1 * bvec[249] - 1 * bvec[252] - 1 * bvec[253];
       alphavec[184] = +18 * bvec[0] - 18 * bvec[2] - 18 * bvec[4] + 18 * bvec[6] - 18 * bvec[8] + 18 * bvec[10] + 18 * bvec[12] - 18 * bvec[14] + 12 * bvec[32] + 6 * bvec[34] - 12 * bvec[36] - 6 * bvec[38] - 12 * bvec[40] - 6 * bvec[42] + 12 * bvec[44] + 6 * bvec[46] + 9 * bvec[48] - 9 * bvec[50] + 9 * bvec[52] - 9 * bvec[54] - 9 * bvec[56] + 9 * bvec[58] - 9 * bvec[60] + 9 * bvec[62] + 12 * bvec[64] - 12 * bvec[66] - 12 * bvec[68] + 12 * bvec[70] + 6 * bvec[72] - 6 * bvec[74] - 6 * bvec[76] + 6 * bvec[78] + 6 * bvec[128] + 3 * bvec[130] + 6 * bvec[132] + 3 * bvec[134] - 6 * bvec[136] - 3 * bvec[138] - 6 * bvec[140] - 3 * bvec[142] + 8 * bvec[144] + 4 * bvec[146] - 8 * bvec[148] - 4 * bvec[150] + 4 * bvec[152] + 2 * bvec[154] - 4 * bvec[156] - 2 * bvec[158] + 6 * bvec[160] - 6 * bvec[162] + 6 * bvec[164] - 6 * bvec[166] + 3 * bvec[168] - 3 * bvec[170] + 3 * bvec[172] - 3 * bvec[174] + 4 * bvec[224] + 2 * bvec[226] + 4 * bvec[228] + 2 * bvec[230] + 2 * bvec[232] + 1 * bvec[234] + 2 * bvec[236] + 1 * bvec[238];
       alphavec[185] = +18 * bvec[16] - 18 * bvec[18] - 18 * bvec[20] + 18 * bvec[22] - 18 * bvec[24] + 18 * bvec[26] + 18 * bvec[28] - 18 * bvec[30] + 12 * bvec[80] + 6 * bvec[82] - 12 * bvec[84] - 6 * bvec[86] - 12 * bvec[88] - 6 * bvec[90] + 12 * bvec[92] + 6 * bvec[94] + 9 * bvec[96] - 9 * bvec[98] + 9 * bvec[100] - 9 * bvec[102] - 9 * bvec[104] + 9 * bvec[106] - 9 * bvec[108] + 9 * bvec[110] + 12 * bvec[112] - 12 * bvec[114] - 12 * bvec[116] + 12 * bvec[118] + 6 * bvec[120] - 6 * bvec[122] - 6 * bvec[124] + 6 * bvec[126] + 6 * bvec[176] + 3 * bvec[178] + 6 * bvec[180] + 3 * bvec[182] - 6 * bvec[184] - 3 * bvec[186] - 6 * bvec[188] - 3 * bvec[190] + 8 * bvec[192] + 4 * bvec[194] - 8 * bvec[196] - 4 * bvec[198] + 4 * bvec[200] + 2 * bvec[202] - 4 * bvec[204] - 2 * bvec[206] + 6 * bvec[208] - 6 * bvec[210] + 6 * bvec[212] - 6 * bvec[214] + 3 * bvec[216] - 3 * bvec[218] + 3 * bvec[220] - 3 * bvec[222] + 4 * bvec[240] + 2 * bvec[242] + 4 * bvec[244] + 2 * bvec[246] + 2 * bvec[248] + 1 * bvec[250] + 2 * bvec[252] + 1 * bvec[254];
       alphavec[186] = -54 * bvec[0] + 54 * bvec[1] + 54 * bvec[2] - 54 * bvec[3] + 54 * bvec[4] - 54 * bvec[5] - 54 * bvec[6] + 54 * bvec[7] + 54 * bvec[8] - 54 * bvec[9] - 54 * bvec[10] + 54 * bvec[11] - 54 * bvec[12] + 54 * bvec[13] + 54 * bvec[14] - 54 * bvec[15] - 36 * bvec[16] - 18 * bvec[17] + 36 * bvec[18] + 18 * bvec[19] + 36 * bvec[20] + 18 * bvec[21] - 36 * bvec[22] - 18 * bvec[23] + 36 * bvec[24] + 18 * bvec[25] - 36 * bvec[26] - 18 * bvec[27] - 36 * bvec[28] - 18 * bvec[29] + 36 * bvec[30] + 18 * bvec[31] - 36 * bvec[32] + 36 * bvec[33] - 18 * bvec[34] + 18 * bvec[35] + 36 * bvec[36] - 36 * bvec[37] + 18 * bvec[38] - 18 * bvec[39] + 36 * bvec[40] - 36 * bvec[41] + 18 * bvec[42] - 18 * bvec[43] - 36 * bvec[44] + 36 * bvec[45] - 18 * bvec[46] + 18 * bvec[47] - 27 * bvec[48] + 27 * bvec[49] + 27 * bvec[50] - 27 * bvec[51] - 27 * bvec[52] + 27 * bvec[53] + 27 * bvec[54] - 27 * bvec[55] + 27 * bvec[56] - 27 * bvec[57] - 27 * bvec[58] + 27 * bvec[59] + 27 * bvec[60] - 27 * bvec[61] - 27 * bvec[62] + 27 * bvec[63] - 36 * bvec[64] + 36 * bvec[65] + 36 * bvec[66] - 36 * bvec[67] + 36 * bvec[68] - 36 * bvec[69] - 36 * bvec[70] + 36 * bvec[71] - 18 * bvec[72] + 18 * bvec[73] + 18 * bvec[74] - 18 * bvec[75] + 18 * bvec[76] - 18 * bvec[77] - 18 * bvec[78] + 18 * bvec[79] - 24 * bvec[80] - 12 * bvec[81] - 12 * bvec[82] - 6 * bvec[83] + 24 * bvec[84] + 12 * bvec[85] + 12 * bvec[86] + 6 * bvec[87] + 24 * bvec[88] + 12 * bvec[89] + 12 * bvec[90] + 6 * bvec[91] - 24 * bvec[92] - 12 * bvec[93] - 12 * bvec[94] - 6 * bvec[95] - 18 * bvec[96] - 9 * bvec[97] + 18 * bvec[98] + 9 * bvec[99] - 18 * bvec[100] - 9 * bvec[101] + 18 * bvec[102] + 9 * bvec[103] + 18 * bvec[104] + 9 * bvec[105] - 18 * bvec[106] - 9 * bvec[107] + 18 * bvec[108] + 9 * bvec[109] - 18 * bvec[110] - 9 * bvec[111] - 24 * bvec[112] - 12 * bvec[113] + 24 * bvec[114] + 12 * bvec[115] + 24 * bvec[116] + 12 * bvec[117] - 24 * bvec[118] - 12 * bvec[119] - 12 * bvec[120] - 6 * bvec[121] + 12 * bvec[122] + 6 * bvec[123] + 12 * bvec[124] + 6 * bvec[125] - 12 * bvec[126] - 6 * bvec[127] - 18 * bvec[128] + 18 * bvec[129] - 9 * bvec[130] + 9 * bvec[131] - 18 * bvec[132] + 18 * bvec[133] - 9 * bvec[134] + 9 * bvec[135] + 18 * bvec[136] - 18 * bvec[137] + 9 * bvec[138] - 9 * bvec[139] + 18 * bvec[140] - 18 * bvec[141] + 9 * bvec[142] - 9 * bvec[143] - 24 * bvec[144] + 24 * bvec[145] - 12 * bvec[146] + 12 * bvec[147] + 24 * bvec[148] - 24 * bvec[149] + 12 * bvec[150] - 12 * bvec[151] - 12 * bvec[152] + 12 * bvec[153] - 6 * bvec[154] + 6 * bvec[155] + 12 * bvec[156] - 12 * bvec[157] + 6 * bvec[158] - 6 * bvec[159] - 18 * bvec[160] + 18 * bvec[161] + 18 * bvec[162] - 18 * bvec[163] - 18 * bvec[164] + 18 * bvec[165] + 18 * bvec[166] - 18 * bvec[167] - 9 * bvec[168] + 9 * bvec[169] + 9 * bvec[170] - 9 * bvec[171] - 9 * bvec[172] + 9 * bvec[173] + 9 * bvec[174] - 9 * bvec[175] - 12 * bvec[176] - 6 * bvec[177] - 6 * bvec[178] - 3 * bvec[179] - 12 * bvec[180] - 6 * bvec[181] - 6 * bvec[182] - 3 * bvec[183] + 12 * bvec[184] + 6 * bvec[185] + 6 * bvec[186] + 3 * bvec[187] + 12 * bvec[188] + 6 * bvec[189] + 6 * bvec[190] + 3 * bvec[191] - 16 * bvec[192] - 8 * bvec[193] - 8 * bvec[194] - 4 * bvec[195] + 16 * bvec[196] + 8 * bvec[197] + 8 * bvec[198] + 4 * bvec[199] - 8 * bvec[200] - 4 * bvec[201] - 4 * bvec[202] - 2 * bvec[203] + 8 * bvec[204] + 4 * bvec[205] + 4 * bvec[206] + 2 * bvec[207] - 12 * bvec[208] - 6 * bvec[209] + 12 * bvec[210] + 6 * bvec[211] - 12 * bvec[212] - 6 * bvec[213] + 12 * bvec[214] + 6 * bvec[215] - 6 * bvec[216] - 3 * bvec[217] + 6 * bvec[218] + 3 * bvec[219] - 6 * bvec[220] - 3 * bvec[221] + 6 * bvec[222] + 3 * bvec[223] - 12 * bvec[224] + 12 * bvec[225] - 6 * bvec[226] + 6 * bvec[227] - 12 * bvec[228] + 12 * bvec[229] - 6 * bvec[230] + 6 * bvec[231] - 6 * bvec[232] + 6 * bvec[233] - 3 * bvec[234] + 3 * bvec[235] - 6 * bvec[236] + 6 * bvec[237] - 3 * bvec[238] + 3 * bvec[239] - 8 * bvec[240] - 4 * bvec[241] - 4 * bvec[242] - 2 * bvec[243] - 8 * bvec[244] - 4 * bvec[245] - 4 * bvec[246] - 2 * bvec[247] - 4 * bvec[248] - 2 * bvec[249] - 2 * bvec[250] - 1 * bvec[251] - 4 * bvec[252] - 2 * bvec[253] - 2 * bvec[254] - 1 * bvec[255];
       alphavec[187] = +36 * bvec[0] - 36 * bvec[1] - 36 * bvec[2] + 36 * bvec[3] - 36 * bvec[4] + 36 * bvec[5] + 36 * bvec[6] - 36 * bvec[7] - 36 * bvec[8] + 36 * bvec[9] + 36 * bvec[10] - 36 * bvec[11] + 36 * bvec[12] - 36 * bvec[13] - 36 * bvec[14] + 36 * bvec[15] + 18 * bvec[16] + 18 * bvec[17] - 18 * bvec[18] - 18 * bvec[19] - 18 * bvec[20] - 18 * bvec[21] + 18 * bvec[22] + 18 * bvec[23] - 18 * bvec[24] - 18 * bvec[25] + 18 * bvec[26] + 18 * bvec[27] + 18 * bvec[28] + 18 * bvec[29] - 18 * bvec[30] - 18 * bvec[31] + 24 * bvec[32] - 24 * bvec[33] + 12 * bvec[34] - 12 * bvec[35] - 24 * bvec[36] + 24 * bvec[37] - 12 * bvec[38] + 12 * bvec[39] - 24 * bvec[40] + 24 * bvec[41] - 12 * bvec[42] + 12 * bvec[43] + 24 * bvec[44] - 24 * bvec[45] + 12 * bvec[46] - 12 * bvec[47] + 18 * bvec[48] - 18 * bvec[49] - 18 * bvec[50] + 18 * bvec[51] + 18 * bvec[52] - 18 * bvec[53] - 18 * bvec[54] + 18 * bvec[55] - 18 * bvec[56] + 18 * bvec[57] + 18 * bvec[58] - 18 * bvec[59] - 18 * bvec[60] + 18 * bvec[61] + 18 * bvec[62] - 18 * bvec[63] + 24 * bvec[64] - 24 * bvec[65] - 24 * bvec[66] + 24 * bvec[67] - 24 * bvec[68] + 24 * bvec[69] + 24 * bvec[70] - 24 * bvec[71] + 12 * bvec[72] - 12 * bvec[73] - 12 * bvec[74] + 12 * bvec[75] - 12 * bvec[76] + 12 * bvec[77] + 12 * bvec[78] - 12 * bvec[79] + 12 * bvec[80] + 12 * bvec[81] + 6 * bvec[82] + 6 * bvec[83] - 12 * bvec[84] - 12 * bvec[85] - 6 * bvec[86] - 6 * bvec[87] - 12 * bvec[88] - 12 * bvec[89] - 6 * bvec[90] - 6 * bvec[91] + 12 * bvec[92] + 12 * bvec[93] + 6 * bvec[94] + 6 * bvec[95] + 9 * bvec[96] + 9 * bvec[97] - 9 * bvec[98] - 9 * bvec[99] + 9 * bvec[100] + 9 * bvec[101] - 9 * bvec[102] - 9 * bvec[103] - 9 * bvec[104] - 9 * bvec[105] + 9 * bvec[106] + 9 * bvec[107] - 9 * bvec[108] - 9 * bvec[109] + 9 * bvec[110] + 9 * bvec[111] + 12 * bvec[112] + 12 * bvec[113] - 12 * bvec[114] - 12 * bvec[115] - 12 * bvec[116] - 12 * bvec[117] + 12 * bvec[118] + 12 * bvec[119] + 6 * bvec[120] + 6 * bvec[121] - 6 * bvec[122] - 6 * bvec[123] - 6 * bvec[124] - 6 * bvec[125] + 6 * bvec[126] + 6 * bvec[127] + 12 * bvec[128] - 12 * bvec[129] + 6 * bvec[130] - 6 * bvec[131] + 12 * bvec[132] - 12 * bvec[133] + 6 * bvec[134] - 6 * bvec[135] - 12 * bvec[136] + 12 * bvec[137] - 6 * bvec[138] + 6 * bvec[139] - 12 * bvec[140] + 12 * bvec[141] - 6 * bvec[142] + 6 * bvec[143] + 16 * bvec[144] - 16 * bvec[145] + 8 * bvec[146] - 8 * bvec[147] - 16 * bvec[148] + 16 * bvec[149] - 8 * bvec[150] + 8 * bvec[151] + 8 * bvec[152] - 8 * bvec[153] + 4 * bvec[154] - 4 * bvec[155] - 8 * bvec[156] + 8 * bvec[157] - 4 * bvec[158] + 4 * bvec[159] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 12 * bvec[164] - 12 * bvec[165] - 12 * bvec[166] + 12 * bvec[167] + 6 * bvec[168] - 6 * bvec[169] - 6 * bvec[170] + 6 * bvec[171] + 6 * bvec[172] - 6 * bvec[173] - 6 * bvec[174] + 6 * bvec[175] + 6 * bvec[176] + 6 * bvec[177] + 3 * bvec[178] + 3 * bvec[179] + 6 * bvec[180] + 6 * bvec[181] + 3 * bvec[182] + 3 * bvec[183] - 6 * bvec[184] - 6 * bvec[185] - 3 * bvec[186] - 3 * bvec[187] - 6 * bvec[188] - 6 * bvec[189] - 3 * bvec[190] - 3 * bvec[191] + 8 * bvec[192] + 8 * bvec[193] + 4 * bvec[194] + 4 * bvec[195] - 8 * bvec[196] - 8 * bvec[197] - 4 * bvec[198] - 4 * bvec[199] + 4 * bvec[200] + 4 * bvec[201] + 2 * bvec[202] + 2 * bvec[203] - 4 * bvec[204] - 4 * bvec[205] - 2 * bvec[206] - 2 * bvec[207] + 6 * bvec[208] + 6 * bvec[209] - 6 * bvec[210] - 6 * bvec[211] + 6 * bvec[212] + 6 * bvec[213] - 6 * bvec[214] - 6 * bvec[215] + 3 * bvec[216] + 3 * bvec[217] - 3 * bvec[218] - 3 * bvec[219] + 3 * bvec[220] + 3 * bvec[221] - 3 * bvec[222] - 3 * bvec[223] + 8 * bvec[224] - 8 * bvec[225] + 4 * bvec[226] - 4 * bvec[227] + 8 * bvec[228] - 8 * bvec[229] + 4 * bvec[230] - 4 * bvec[231] + 4 * bvec[232] - 4 * bvec[233] + 2 * bvec[234] - 2 * bvec[235] + 4 * bvec[236] - 4 * bvec[237] + 2 * bvec[238] - 2 * bvec[239] + 4 * bvec[240] + 4 * bvec[241] + 2 * bvec[242] + 2 * bvec[243] + 4 * bvec[244] + 4 * bvec[245] + 2 * bvec[246] + 2 * bvec[247] + 2 * bvec[248] + 2 * bvec[249] + 1 * bvec[250] + 1 * bvec[251] + 2 * bvec[252] + 2 * bvec[253] + 1 * bvec[254] + 1 * bvec[255];
       alphavec[188] = -12 * bvec[0] + 12 * bvec[2] + 12 * bvec[4] - 12 * bvec[6] + 12 * bvec[8] - 12 * bvec[10] - 12 * bvec[12] + 12 * bvec[14] - 6 * bvec[32] - 6 * bvec[34] + 6 * bvec[36] + 6 * bvec[38] + 6 * bvec[40] + 6 * bvec[42] - 6 * bvec[44] - 6 * bvec[46] - 6 * bvec[48] + 6 * bvec[50] - 6 * bvec[52] + 6 * bvec[54] + 6 * bvec[56] - 6 * bvec[58] + 6 * bvec[60] - 6 * bvec[62] - 8 * bvec[64] + 8 * bvec[66] + 8 * bvec[68] - 8 * bvec[70] - 4 * bvec[72] + 4 * bvec[74] + 4 * bvec[76] - 4 * bvec[78] - 3 * bvec[128] - 3 * bvec[130] - 3 * bvec[132] - 3 * bvec[134] + 3 * bvec[136] + 3 * bvec[138] + 3 * bvec[140] + 3 * bvec[142] - 4 * bvec[144] - 4 * bvec[146] + 4 * bvec[148] + 4 * bvec[150] - 2 * bvec[152] - 2 * bvec[154] + 2 * bvec[156] + 2 * bvec[158] - 4 * bvec[160] + 4 * bvec[162] - 4 * bvec[164] + 4 * bvec[166] - 2 * bvec[168] + 2 * bvec[170] - 2 * bvec[172] + 2 * bvec[174] - 2 * bvec[224] - 2 * bvec[226] - 2 * bvec[228] - 2 * bvec[230] - 1 * bvec[232] - 1 * bvec[234] - 1 * bvec[236] - 1 * bvec[238];
       alphavec[189] = -12 * bvec[16] + 12 * bvec[18] + 12 * bvec[20] - 12 * bvec[22] + 12 * bvec[24] - 12 * bvec[26] - 12 * bvec[28] + 12 * bvec[30] - 6 * bvec[80] - 6 * bvec[82] + 6 * bvec[84] + 6 * bvec[86] + 6 * bvec[88] + 6 * bvec[90] - 6 * bvec[92] - 6 * bvec[94] - 6 * bvec[96] + 6 * bvec[98] - 6 * bvec[100] + 6 * bvec[102] + 6 * bvec[104] - 6 * bvec[106] + 6 * bvec[108] - 6 * bvec[110] - 8 * bvec[112] + 8 * bvec[114] + 8 * bvec[116] - 8 * bvec[118] - 4 * bvec[120] + 4 * bvec[122] + 4 * bvec[124] - 4 * bvec[126] - 3 * bvec[176] - 3 * bvec[178] - 3 * bvec[180] - 3 * bvec[182] + 3 * bvec[184] + 3 * bvec[186] + 3 * bvec[188] + 3 * bvec[190] - 4 * bvec[192] - 4 * bvec[194] + 4 * bvec[196] + 4 * bvec[198] - 2 * bvec[200] - 2 * bvec[202] + 2 * bvec[204] + 2 * bvec[206] - 4 * bvec[208] + 4 * bvec[210] - 4 * bvec[212] + 4 * bvec[214] - 2 * bvec[216] + 2 * bvec[218] - 2 * bvec[220] + 2 * bvec[222] - 2 * bvec[240] - 2 * bvec[242] - 2 * bvec[244] - 2 * bvec[246] - 1 * bvec[248] - 1 * bvec[250] - 1 * bvec[252] - 1 * bvec[254];
       alphavec[190] = +36 * bvec[0] - 36 * bvec[1] - 36 * bvec[2] + 36 * bvec[3] - 36 * bvec[4] + 36 * bvec[5] + 36 * bvec[6] - 36 * bvec[7] - 36 * bvec[8] + 36 * bvec[9] + 36 * bvec[10] - 36 * bvec[11] + 36 * bvec[12] - 36 * bvec[13] - 36 * bvec[14] + 36 * bvec[15] + 24 * bvec[16] + 12 * bvec[17] - 24 * bvec[18] - 12 * bvec[19] - 24 * bvec[20] - 12 * bvec[21] + 24 * bvec[22] + 12 * bvec[23] - 24 * bvec[24] - 12 * bvec[25] + 24 * bvec[26] + 12 * bvec[27] + 24 * bvec[28] + 12 * bvec[29] - 24 * bvec[30] - 12 * bvec[31] + 18 * bvec[32] - 18 * bvec[33] + 18 * bvec[34] - 18 * bvec[35] - 18 * bvec[36] + 18 * bvec[37] - 18 * bvec[38] + 18 * bvec[39] - 18 * bvec[40] + 18 * bvec[41] - 18 * bvec[42] + 18 * bvec[43] + 18 * bvec[44] - 18 * bvec[45] + 18 * bvec[46] - 18 * bvec[47] + 18 * bvec[48] - 18 * bvec[49] - 18 * bvec[50] + 18 * bvec[51] + 18 * bvec[52] - 18 * bvec[53] - 18 * bvec[54] + 18 * bvec[55] - 18 * bvec[56] + 18 * bvec[57] + 18 * bvec[58] - 18 * bvec[59] - 18 * bvec[60] + 18 * bvec[61] + 18 * bvec[62] - 18 * bvec[63] + 24 * bvec[64] - 24 * bvec[65] - 24 * bvec[66] + 24 * bvec[67] - 24 * bvec[68] + 24 * bvec[69] + 24 * bvec[70] - 24 * bvec[71] + 12 * bvec[72] - 12 * bvec[73] - 12 * bvec[74] + 12 * bvec[75] - 12 * bvec[76] + 12 * bvec[77] + 12 * bvec[78] - 12 * bvec[79] + 12 * bvec[80] + 6 * bvec[81] + 12 * bvec[82] + 6 * bvec[83] - 12 * bvec[84] - 6 * bvec[85] - 12 * bvec[86] - 6 * bvec[87] - 12 * bvec[88] - 6 * bvec[89] - 12 * bvec[90] - 6 * bvec[91] + 12 * bvec[92] + 6 * bvec[93] + 12 * bvec[94] + 6 * bvec[95] + 12 * bvec[96] + 6 * bvec[97] - 12 * bvec[98] - 6 * bvec[99] + 12 * bvec[100] + 6 * bvec[101] - 12 * bvec[102] - 6 * bvec[103] - 12 * bvec[104] - 6 * bvec[105] + 12 * bvec[106] + 6 * bvec[107] - 12 * bvec[108] - 6 * bvec[109] + 12 * bvec[110] + 6 * bvec[111] + 16 * bvec[112] + 8 * bvec[113] - 16 * bvec[114] - 8 * bvec[115] - 16 * bvec[116] - 8 * bvec[117] + 16 * bvec[118] + 8 * bvec[119] + 8 * bvec[120] + 4 * bvec[121] - 8 * bvec[122] - 4 * bvec[123] - 8 * bvec[124] - 4 * bvec[125] + 8 * bvec[126] + 4 * bvec[127] + 9 * bvec[128] - 9 * bvec[129] + 9 * bvec[130] - 9 * bvec[131] + 9 * bvec[132] - 9 * bvec[133] + 9 * bvec[134] - 9 * bvec[135] - 9 * bvec[136] + 9 * bvec[137] - 9 * bvec[138] + 9 * bvec[139] - 9 * bvec[140] + 9 * bvec[141] - 9 * bvec[142] + 9 * bvec[143] + 12 * bvec[144] - 12 * bvec[145] + 12 * bvec[146] - 12 * bvec[147] - 12 * bvec[148] + 12 * bvec[149] - 12 * bvec[150] + 12 * bvec[151] + 6 * bvec[152] - 6 * bvec[153] + 6 * bvec[154] - 6 * bvec[155] - 6 * bvec[156] + 6 * bvec[157] - 6 * bvec[158] + 6 * bvec[159] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 12 * bvec[164] - 12 * bvec[165] - 12 * bvec[166] + 12 * bvec[167] + 6 * bvec[168] - 6 * bvec[169] - 6 * bvec[170] + 6 * bvec[171] + 6 * bvec[172] - 6 * bvec[173] - 6 * bvec[174] + 6 * bvec[175] + 6 * bvec[176] + 3 * bvec[177] + 6 * bvec[178] + 3 * bvec[179] + 6 * bvec[180] + 3 * bvec[181] + 6 * bvec[182] + 3 * bvec[183] - 6 * bvec[184] - 3 * bvec[185] - 6 * bvec[186] - 3 * bvec[187] - 6 * bvec[188] - 3 * bvec[189] - 6 * bvec[190] - 3 * bvec[191] + 8 * bvec[192] + 4 * bvec[193] + 8 * bvec[194] + 4 * bvec[195] - 8 * bvec[196] - 4 * bvec[197] - 8 * bvec[198] - 4 * bvec[199] + 4 * bvec[200] + 2 * bvec[201] + 4 * bvec[202] + 2 * bvec[203] - 4 * bvec[204] - 2 * bvec[205] - 4 * bvec[206] - 2 * bvec[207] + 8 * bvec[208] + 4 * bvec[209] - 8 * bvec[210] - 4 * bvec[211] + 8 * bvec[212] + 4 * bvec[213] - 8 * bvec[214] - 4 * bvec[215] + 4 * bvec[216] + 2 * bvec[217] - 4 * bvec[218] - 2 * bvec[219] + 4 * bvec[220] + 2 * bvec[221] - 4 * bvec[222] - 2 * bvec[223] + 6 * bvec[224] - 6 * bvec[225] + 6 * bvec[226] - 6 * bvec[227] + 6 * bvec[228] - 6 * bvec[229] + 6 * bvec[230] - 6 * bvec[231] + 3 * bvec[232] - 3 * bvec[233] + 3 * bvec[234] - 3 * bvec[235] + 3 * bvec[236] - 3 * bvec[237] + 3 * bvec[238] - 3 * bvec[239] + 4 * bvec[240] + 2 * bvec[241] + 4 * bvec[242] + 2 * bvec[243] + 4 * bvec[244] + 2 * bvec[245] + 4 * bvec[246] + 2 * bvec[247] + 2 * bvec[248] + 1 * bvec[249] + 2 * bvec[250] + 1 * bvec[251] + 2 * bvec[252] + 1 * bvec[253] + 2 * bvec[254] + 1 * bvec[255];
       alphavec[191] = -24 * bvec[0] + 24 * bvec[1] + 24 * bvec[2] - 24 * bvec[3] + 24 * bvec[4] - 24 * bvec[5] - 24 * bvec[6] + 24 * bvec[7] + 24 * bvec[8] - 24 * bvec[9] - 24 * bvec[10] + 24 * bvec[11] - 24 * bvec[12] + 24 * bvec[13] + 24 * bvec[14] - 24 * bvec[15] - 12 * bvec[16] - 12 * bvec[17] + 12 * bvec[18] + 12 * bvec[19] + 12 * bvec[20] + 12 * bvec[21] - 12 * bvec[22] - 12 * bvec[23] + 12 * bvec[24] + 12 * bvec[25] - 12 * bvec[26] - 12 * bvec[27] - 12 * bvec[28] - 12 * bvec[29] + 12 * bvec[30] + 12 * bvec[31] - 12 * bvec[32] + 12 * bvec[33] - 12 * bvec[34] + 12 * bvec[35] + 12 * bvec[36] - 12 * bvec[37] + 12 * bvec[38] - 12 * bvec[39] + 12 * bvec[40] - 12 * bvec[41] + 12 * bvec[42] - 12 * bvec[43] - 12 * bvec[44] + 12 * bvec[45] - 12 * bvec[46] + 12 * bvec[47] - 12 * bvec[48] + 12 * bvec[49] + 12 * bvec[50] - 12 * bvec[51] - 12 * bvec[52] + 12 * bvec[53] + 12 * bvec[54] - 12 * bvec[55] + 12 * bvec[56] - 12 * bvec[57] - 12 * bvec[58] + 12 * bvec[59] + 12 * bvec[60] - 12 * bvec[61] - 12 * bvec[62] + 12 * bvec[63] - 16 * bvec[64] + 16 * bvec[65] + 16 * bvec[66] - 16 * bvec[67] + 16 * bvec[68] - 16 * bvec[69] - 16 * bvec[70] + 16 * bvec[71] - 8 * bvec[72] + 8 * bvec[73] + 8 * bvec[74] - 8 * bvec[75] + 8 * bvec[76] - 8 * bvec[77] - 8 * bvec[78] + 8 * bvec[79] - 6 * bvec[80] - 6 * bvec[81] - 6 * bvec[82] - 6 * bvec[83] + 6 * bvec[84] + 6 * bvec[85] + 6 * bvec[86] + 6 * bvec[87] + 6 * bvec[88] + 6 * bvec[89] + 6 * bvec[90] + 6 * bvec[91] - 6 * bvec[92] - 6 * bvec[93] - 6 * bvec[94] - 6 * bvec[95] - 6 * bvec[96] - 6 * bvec[97] + 6 * bvec[98] + 6 * bvec[99] - 6 * bvec[100] - 6 * bvec[101] + 6 * bvec[102] + 6 * bvec[103] + 6 * bvec[104] + 6 * bvec[105] - 6 * bvec[106] - 6 * bvec[107] + 6 * bvec[108] + 6 * bvec[109] - 6 * bvec[110] - 6 * bvec[111] - 8 * bvec[112] - 8 * bvec[113] + 8 * bvec[114] + 8 * bvec[115] + 8 * bvec[116] + 8 * bvec[117] - 8 * bvec[118] - 8 * bvec[119] - 4 * bvec[120] - 4 * bvec[121] + 4 * bvec[122] + 4 * bvec[123] + 4 * bvec[124] + 4 * bvec[125] - 4 * bvec[126] - 4 * bvec[127] - 6 * bvec[128] + 6 * bvec[129] - 6 * bvec[130] + 6 * bvec[131] - 6 * bvec[132] + 6 * bvec[133] - 6 * bvec[134] + 6 * bvec[135] + 6 * bvec[136] - 6 * bvec[137] + 6 * bvec[138] - 6 * bvec[139] + 6 * bvec[140] - 6 * bvec[141] + 6 * bvec[142] - 6 * bvec[143] - 8 * bvec[144] + 8 * bvec[145] - 8 * bvec[146] + 8 * bvec[147] + 8 * bvec[148] - 8 * bvec[149] + 8 * bvec[150] - 8 * bvec[151] - 4 * bvec[152] + 4 * bvec[153] - 4 * bvec[154] + 4 * bvec[155] + 4 * bvec[156] - 4 * bvec[157] + 4 * bvec[158] - 4 * bvec[159] - 8 * bvec[160] + 8 * bvec[161] + 8 * bvec[162] - 8 * bvec[163] - 8 * bvec[164] + 8 * bvec[165] + 8 * bvec[166] - 8 * bvec[167] - 4 * bvec[168] + 4 * bvec[169] + 4 * bvec[170] - 4 * bvec[171] - 4 * bvec[172] + 4 * bvec[173] + 4 * bvec[174] - 4 * bvec[175] - 3 * bvec[176] - 3 * bvec[177] - 3 * bvec[178] - 3 * bvec[179] - 3 * bvec[180] - 3 * bvec[181] - 3 * bvec[182] - 3 * bvec[183] + 3 * bvec[184] + 3 * bvec[185] + 3 * bvec[186] + 3 * bvec[187] + 3 * bvec[188] + 3 * bvec[189] + 3 * bvec[190] + 3 * bvec[191] - 4 * bvec[192] - 4 * bvec[193] - 4 * bvec[194] - 4 * bvec[195] + 4 * bvec[196] + 4 * bvec[197] + 4 * bvec[198] + 4 * bvec[199] - 2 * bvec[200] - 2 * bvec[201] - 2 * bvec[202] - 2 * bvec[203] + 2 * bvec[204] + 2 * bvec[205] + 2 * bvec[206] + 2 * bvec[207] - 4 * bvec[208] - 4 * bvec[209] + 4 * bvec[210] + 4 * bvec[211] - 4 * bvec[212] - 4 * bvec[213] + 4 * bvec[214] + 4 * bvec[215] - 2 * bvec[216] - 2 * bvec[217] + 2 * bvec[218] + 2 * bvec[219] - 2 * bvec[220] - 2 * bvec[221] + 2 * bvec[222] + 2 * bvec[223] - 4 * bvec[224] + 4 * bvec[225] - 4 * bvec[226] + 4 * bvec[227] - 4 * bvec[228] + 4 * bvec[229] - 4 * bvec[230] + 4 * bvec[231] - 2 * bvec[232] + 2 * bvec[233] - 2 * bvec[234] + 2 * bvec[235] - 2 * bvec[236] + 2 * bvec[237] - 2 * bvec[238] + 2 * bvec[239] - 2 * bvec[240] - 2 * bvec[241] - 2 * bvec[242] - 2 * bvec[243] - 2 * bvec[244] - 2 * bvec[245] - 2 * bvec[246] - 2 * bvec[247] - 1 * bvec[248] - 1 * bvec[249] - 1 * bvec[250] - 1 * bvec[251] - 1 * bvec[252] - 1 * bvec[253] - 1 * bvec[254] - 1 * bvec[255];
       alphavec[192] = +2 * bvec[0] - 2 * bvec[8] + 1 * bvec[64] + 1 * bvec[72];
       alphavec[193] = +2 * bvec[16] - 2 * bvec[24] + 1 * bvec[112] + 1 * bvec[120];
       alphavec[194] = -6 * bvec[0] + 6 * bvec[1] + 6 * bvec[8] - 6 * bvec[9] - 4 * bvec[16] - 2 * bvec[17] + 4 * bvec[24] + 2 * bvec[25] - 3 * bvec[64] + 3 * bvec[65] - 3 * bvec[72] + 3 * bvec[73] - 2 * bvec[112] - 1 * bvec[113] - 2 * bvec[120] - 1 * bvec[121];
       alphavec[195] = +4 * bvec[0] - 4 * bvec[1] - 4 * bvec[8] + 4 * bvec[9] + 2 * bvec[16] + 2 * bvec[17] - 2 * bvec[24] - 2 * bvec[25] + 2 * bvec[64] - 2 * bvec[65] + 2 * bvec[72] - 2 * bvec[73] + 1 * bvec[112] + 1 * bvec[113] + 1 * bvec[120] + 1 * bvec[121];
       alphavec[196] = +2 * bvec[32] - 2 * bvec[40] + 1 * bvec[144] + 1 * bvec[152];
       alphavec[197] = +2 * bvec[80] - 2 * bvec[88] + 1 * bvec[192] + 1 * bvec[200];
       alphavec[198] = -6 * bvec[32] + 6 * bvec[33] + 6 * bvec[40] - 6 * bvec[41] - 4 * bvec[80] - 2 * bvec[81] + 4 * bvec[88] + 2 * bvec[89] - 3 * bvec[144] + 3 * bvec[145] - 3 * bvec[152] + 3 * bvec[153] - 2 * bvec[192] - 1 * bvec[193] - 2 * bvec[200] - 1 * bvec[201];
       alphavec[199] = +4 * bvec[32] - 4 * bvec[33] - 4 * bvec[40] + 4 * bvec[41] + 2 * bvec[80] + 2 * bvec[81] - 2 * bvec[88] - 2 * bvec[89] + 2 * bvec[144] - 2 * bvec[145] + 2 * bvec[152] - 2 * bvec[153] + 1 * bvec[192] + 1 * bvec[193] + 1 * bvec[200] + 1 * bvec[201];
       alphavec[200] = -6 * bvec[0] + 6 * bvec[2] + 6 * bvec[8] - 6 * bvec[10] - 4 * bvec[32] - 2 * bvec[34] + 4 * bvec[40] + 2 * bvec[42] - 3 * bvec[64] + 3 * bvec[66] - 3 * bvec[72] + 3 * bvec[74] - 2 * bvec[144] - 1 * bvec[146] - 2 * bvec[152] - 1 * bvec[154];
       alphavec[201] = -6 * bvec[16] + 6 * bvec[18] + 6 * bvec[24] - 6 * bvec[26] - 4 * bvec[80] - 2 * bvec[82] + 4 * bvec[88] + 2 * bvec[90] - 3 * bvec[112] + 3 * bvec[114] - 3 * bvec[120] + 3 * bvec[122] - 2 * bvec[192] - 1 * bvec[194] - 2 * bvec[200] - 1 * bvec[202];
       alphavec[202] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[2] + 18 * bvec[3] - 18 * bvec[8] + 18 * bvec[9] + 18 * bvec[10] - 18 * bvec[11] + 12 * bvec[16] + 6 * bvec[17] - 12 * bvec[18] - 6 * bvec[19] - 12 * bvec[24] - 6 * bvec[25] + 12 * bvec[26] + 6 * bvec[27] + 12 * bvec[32] - 12 * bvec[33] + 6 * bvec[34] - 6 * bvec[35] - 12 * bvec[40] + 12 * bvec[41] - 6 * bvec[42] + 6 * bvec[43] + 9 * bvec[64] - 9 * bvec[65] - 9 * bvec[66] + 9 * bvec[67] + 9 * bvec[72] - 9 * bvec[73] - 9 * bvec[74] + 9 * bvec[75] + 8 * bvec[80] + 4 * bvec[81] + 4 * bvec[82] + 2 * bvec[83] - 8 * bvec[88] - 4 * bvec[89] - 4 * bvec[90] - 2 * bvec[91] + 6 * bvec[112] + 3 * bvec[113] - 6 * bvec[114] - 3 * bvec[115] + 6 * bvec[120] + 3 * bvec[121] - 6 * bvec[122] - 3 * bvec[123] + 6 * bvec[144] - 6 * bvec[145] + 3 * bvec[146] - 3 * bvec[147] + 6 * bvec[152] - 6 * bvec[153] + 3 * bvec[154] - 3 * bvec[155] + 4 * bvec[192] + 2 * bvec[193] + 2 * bvec[194] + 1 * bvec[195] + 4 * bvec[200] + 2 * bvec[201] + 2 * bvec[202] + 1 * bvec[203];
       alphavec[203] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[8] - 12 * bvec[9] - 12 * bvec[10] + 12 * bvec[11] - 6 * bvec[16] - 6 * bvec[17] + 6 * bvec[18] + 6 * bvec[19] + 6 * bvec[24] + 6 * bvec[25] - 6 * bvec[26] - 6 * bvec[27] - 8 * bvec[32] + 8 * bvec[33] - 4 * bvec[34] + 4 * bvec[35] + 8 * bvec[40] - 8 * bvec[41] + 4 * bvec[42] - 4 * bvec[43] - 6 * bvec[64] + 6 * bvec[65] + 6 * bvec[66] - 6 * bvec[67] - 6 * bvec[72] + 6 * bvec[73] + 6 * bvec[74] - 6 * bvec[75] - 4 * bvec[80] - 4 * bvec[81] - 2 * bvec[82] - 2 * bvec[83] + 4 * bvec[88] + 4 * bvec[89] + 2 * bvec[90] + 2 * bvec[91] - 3 * bvec[112] - 3 * bvec[113] + 3 * bvec[114] + 3 * bvec[115] - 3 * bvec[120] - 3 * bvec[121] + 3 * bvec[122] + 3 * bvec[123] - 4 * bvec[144] + 4 * bvec[145] - 2 * bvec[146] + 2 * bvec[147] - 4 * bvec[152] + 4 * bvec[153] - 2 * bvec[154] + 2 * bvec[155] - 2 * bvec[192] - 2 * bvec[193] - 1 * bvec[194] - 1 * bvec[195] - 2 * bvec[200] - 2 * bvec[201] - 1 * bvec[202] - 1 * bvec[203];
       alphavec[204] = +4 * bvec[0] - 4 * bvec[2] - 4 * bvec[8] + 4 * bvec[10] + 2 * bvec[32] + 2 * bvec[34] - 2 * bvec[40] - 2 * bvec[42] + 2 * bvec[64] - 2 * bvec[66] + 2 * bvec[72] - 2 * bvec[74] + 1 * bvec[144] + 1 * bvec[146] + 1 * bvec[152] + 1 * bvec[154];
       alphavec[205] = +4 * bvec[16] - 4 * bvec[18] - 4 * bvec[24] + 4 * bvec[26] + 2 * bvec[80] + 2 * bvec[82] - 2 * bvec[88] - 2 * bvec[90] + 2 * bvec[112] - 2 * bvec[114] + 2 * bvec[120] - 2 * bvec[122] + 1 * bvec[192] + 1 * bvec[194] + 1 * bvec[200] + 1 * bvec[202];
       alphavec[206] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[2] - 12 * bvec[3] + 12 * bvec[8] - 12 * bvec[9] - 12 * bvec[10] + 12 * bvec[11] - 8 * bvec[16] - 4 * bvec[17] + 8 * bvec[18] + 4 * bvec[19] + 8 * bvec[24] + 4 * bvec[25] - 8 * bvec[26] - 4 * bvec[27] - 6 * bvec[32] + 6 * bvec[33] - 6 * bvec[34] + 6 * bvec[35] + 6 * bvec[40] - 6 * bvec[41] + 6 * bvec[42] - 6 * bvec[43] - 6 * bvec[64] + 6 * bvec[65] + 6 * bvec[66] - 6 * bvec[67] - 6 * bvec[72] + 6 * bvec[73] + 6 * bvec[74] - 6 * bvec[75] - 4 * bvec[80] - 2 * bvec[81] - 4 * bvec[82] - 2 * bvec[83] + 4 * bvec[88] + 2 * bvec[89] + 4 * bvec[90] + 2 * bvec[91] - 4 * bvec[112] - 2 * bvec[113] + 4 * bvec[114] + 2 * bvec[115] - 4 * bvec[120] - 2 * bvec[121] + 4 * bvec[122] + 2 * bvec[123] - 3 * bvec[144] + 3 * bvec[145] - 3 * bvec[146] + 3 * bvec[147] - 3 * bvec[152] + 3 * bvec[153] - 3 * bvec[154] + 3 * bvec[155] - 2 * bvec[192] - 1 * bvec[193] - 2 * bvec[194] - 1 * bvec[195] - 2 * bvec[200] - 1 * bvec[201] - 2 * bvec[202] - 1 * bvec[203];
       alphavec[207] = +8 * bvec[0] - 8 * bvec[1] - 8 * bvec[2] + 8 * bvec[3] - 8 * bvec[8] + 8 * bvec[9] + 8 * bvec[10] - 8 * bvec[11] + 4 * bvec[16] + 4 * bvec[17] - 4 * bvec[18] - 4 * bvec[19] - 4 * bvec[24] - 4 * bvec[25] + 4 * bvec[26] + 4 * bvec[27] + 4 * bvec[32] - 4 * bvec[33] + 4 * bvec[34] - 4 * bvec[35] - 4 * bvec[40] + 4 * bvec[41] - 4 * bvec[42] + 4 * bvec[43] + 4 * bvec[64] - 4 * bvec[65] - 4 * bvec[66] + 4 * bvec[67] + 4 * bvec[72] - 4 * bvec[73] - 4 * bvec[74] + 4 * bvec[75] + 2 * bvec[80] + 2 * bvec[81] + 2 * bvec[82] + 2 * bvec[83] - 2 * bvec[88] - 2 * bvec[89] - 2 * bvec[90] - 2 * bvec[91] + 2 * bvec[112] + 2 * bvec[113] - 2 * bvec[114] - 2 * bvec[115] + 2 * bvec[120] + 2 * bvec[121] - 2 * bvec[122] - 2 * bvec[123] + 2 * bvec[144] - 2 * bvec[145] + 2 * bvec[146] - 2 * bvec[147] + 2 * bvec[152] - 2 * bvec[153] + 2 * bvec[154] - 2 * bvec[155] + 1 * bvec[192] + 1 * bvec[193] + 1 * bvec[194] + 1 * bvec[195] + 1 * bvec[200] + 1 * bvec[201] + 1 * bvec[202] + 1 * bvec[203];
       alphavec[208] = +2 * bvec[48] - 2 * bvec[56] + 1 * bvec[160] + 1 * bvec[168];
       alphavec[209] = +2 * bvec[96] - 2 * bvec[104] + 1 * bvec[208] + 1 * bvec[216];
       alphavec[210] = -6 * bvec[48] + 6 * bvec[49] + 6 * bvec[56] - 6 * bvec[57] - 4 * bvec[96] - 2 * bvec[97] + 4 * bvec[104] + 2 * bvec[105] - 3 * bvec[160] + 3 * bvec[161] - 3 * bvec[168] + 3 * bvec[169] - 2 * bvec[208] - 1 * bvec[209] - 2 * bvec[216] - 1 * bvec[217];
       alphavec[211] = +4 * bvec[48] - 4 * bvec[49] - 4 * bvec[56] + 4 * bvec[57] + 2 * bvec[96] + 2 * bvec[97] - 2 * bvec[104] - 2 * bvec[105] + 2 * bvec[160] - 2 * bvec[161] + 2 * bvec[168] - 2 * bvec[169] + 1 * bvec[208] + 1 * bvec[209] + 1 * bvec[216] + 1 * bvec[217];
       alphavec[212] = +2 * bvec[128] - 2 * bvec[136] + 1 * bvec[224] + 1 * bvec[232];
       alphavec[213] = +2 * bvec[176] - 2 * bvec[184] + 1 * bvec[240] + 1 * bvec[248];
       alphavec[214] = -6 * bvec[128] + 6 * bvec[129] + 6 * bvec[136] - 6 * bvec[137] - 4 * bvec[176] - 2 * bvec[177] + 4 * bvec[184] + 2 * bvec[185] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[232] + 3 * bvec[233] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[248] - 1 * bvec[249];
       alphavec[215] = +4 * bvec[128] - 4 * bvec[129] - 4 * bvec[136] + 4 * bvec[137] + 2 * bvec[176] + 2 * bvec[177] - 2 * bvec[184] - 2 * bvec[185] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[232] - 2 * bvec[233] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[248] + 1 * bvec[249];
       alphavec[216] = -6 * bvec[48] + 6 * bvec[50] + 6 * bvec[56] - 6 * bvec[58] - 4 * bvec[128] - 2 * bvec[130] + 4 * bvec[136] + 2 * bvec[138] - 3 * bvec[160] + 3 * bvec[162] - 3 * bvec[168] + 3 * bvec[170] - 2 * bvec[224] - 1 * bvec[226] - 2 * bvec[232] - 1 * bvec[234];
       alphavec[217] = -6 * bvec[96] + 6 * bvec[98] + 6 * bvec[104] - 6 * bvec[106] - 4 * bvec[176] - 2 * bvec[178] + 4 * bvec[184] + 2 * bvec[186] - 3 * bvec[208] + 3 * bvec[210] - 3 * bvec[216] + 3 * bvec[218] - 2 * bvec[240] - 1 * bvec[242] - 2 * bvec[248] - 1 * bvec[250];
       alphavec[218] = +18 * bvec[48] - 18 * bvec[49] - 18 * bvec[50] + 18 * bvec[51] - 18 * bvec[56] + 18 * bvec[57] + 18 * bvec[58] - 18 * bvec[59] + 12 * bvec[96] + 6 * bvec[97] - 12 * bvec[98] - 6 * bvec[99] - 12 * bvec[104] - 6 * bvec[105] + 12 * bvec[106] + 6 * bvec[107] + 12 * bvec[128] - 12 * bvec[129] + 6 * bvec[130] - 6 * bvec[131] - 12 * bvec[136] + 12 * bvec[137] - 6 * bvec[138] + 6 * bvec[139] + 9 * bvec[160] - 9 * bvec[161] - 9 * bvec[162] + 9 * bvec[163] + 9 * bvec[168] - 9 * bvec[169] - 9 * bvec[170] + 9 * bvec[171] + 8 * bvec[176] + 4 * bvec[177] + 4 * bvec[178] + 2 * bvec[179] - 8 * bvec[184] - 4 * bvec[185] - 4 * bvec[186] - 2 * bvec[187] + 6 * bvec[208] + 3 * bvec[209] - 6 * bvec[210] - 3 * bvec[211] + 6 * bvec[216] + 3 * bvec[217] - 6 * bvec[218] - 3 * bvec[219] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[226] - 3 * bvec[227] + 6 * bvec[232] - 6 * bvec[233] + 3 * bvec[234] - 3 * bvec[235] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[242] + 1 * bvec[243] + 4 * bvec[248] + 2 * bvec[249] + 2 * bvec[250] + 1 * bvec[251];
       alphavec[219] = -12 * bvec[48] + 12 * bvec[49] + 12 * bvec[50] - 12 * bvec[51] + 12 * bvec[56] - 12 * bvec[57] - 12 * bvec[58] + 12 * bvec[59] - 6 * bvec[96] - 6 * bvec[97] + 6 * bvec[98] + 6 * bvec[99] + 6 * bvec[104] + 6 * bvec[105] - 6 * bvec[106] - 6 * bvec[107] - 8 * bvec[128] + 8 * bvec[129] - 4 * bvec[130] + 4 * bvec[131] + 8 * bvec[136] - 8 * bvec[137] + 4 * bvec[138] - 4 * bvec[139] - 6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 6 * bvec[168] + 6 * bvec[169] + 6 * bvec[170] - 6 * bvec[171] - 4 * bvec[176] - 4 * bvec[177] - 2 * bvec[178] - 2 * bvec[179] + 4 * bvec[184] + 4 * bvec[185] + 2 * bvec[186] + 2 * bvec[187] - 3 * bvec[208] - 3 * bvec[209] + 3 * bvec[210] + 3 * bvec[211] - 3 * bvec[216] - 3 * bvec[217] + 3 * bvec[218] + 3 * bvec[219] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[226] + 2 * bvec[227] - 4 * bvec[232] + 4 * bvec[233] - 2 * bvec[234] + 2 * bvec[235] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[242] - 1 * bvec[243] - 2 * bvec[248] - 2 * bvec[249] - 1 * bvec[250] - 1 * bvec[251];
       alphavec[220] = +4 * bvec[48] - 4 * bvec[50] - 4 * bvec[56] + 4 * bvec[58] + 2 * bvec[128] + 2 * bvec[130] - 2 * bvec[136] - 2 * bvec[138] + 2 * bvec[160] - 2 * bvec[162] + 2 * bvec[168] - 2 * bvec[170] + 1 * bvec[224] + 1 * bvec[226] + 1 * bvec[232] + 1 * bvec[234];
       alphavec[221] = +4 * bvec[96] - 4 * bvec[98] - 4 * bvec[104] + 4 * bvec[106] + 2 * bvec[176] + 2 * bvec[178] - 2 * bvec[184] - 2 * bvec[186] + 2 * bvec[208] - 2 * bvec[210] + 2 * bvec[216] - 2 * bvec[218] + 1 * bvec[240] + 1 * bvec[242] + 1 * bvec[248] + 1 * bvec[250];
       alphavec[222] = -12 * bvec[48] + 12 * bvec[49] + 12 * bvec[50] - 12 * bvec[51] + 12 * bvec[56] - 12 * bvec[57] - 12 * bvec[58] + 12 * bvec[59] - 8 * bvec[96] - 4 * bvec[97] + 8 * bvec[98] + 4 * bvec[99] + 8 * bvec[104] + 4 * bvec[105] - 8 * bvec[106] - 4 * bvec[107] - 6 * bvec[128] + 6 * bvec[129] - 6 * bvec[130] + 6 * bvec[131] + 6 * bvec[136] - 6 * bvec[137] + 6 * bvec[138] - 6 * bvec[139] - 6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 6 * bvec[168] + 6 * bvec[169] + 6 * bvec[170] - 6 * bvec[171] - 4 * bvec[176] - 2 * bvec[177] - 4 * bvec[178] - 2 * bvec[179] + 4 * bvec[184] + 2 * bvec[185] + 4 * bvec[186] + 2 * bvec[187] - 4 * bvec[208] - 2 * bvec[209] + 4 * bvec[210] + 2 * bvec[211] - 4 * bvec[216] - 2 * bvec[217] + 4 * bvec[218] + 2 * bvec[219] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[226] + 3 * bvec[227] - 3 * bvec[232] + 3 * bvec[233] - 3 * bvec[234] + 3 * bvec[235] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[242] - 1 * bvec[243] - 2 * bvec[248] - 1 * bvec[249] - 2 * bvec[250] - 1 * bvec[251];
       alphavec[223] = +8 * bvec[48] - 8 * bvec[49] - 8 * bvec[50] + 8 * bvec[51] - 8 * bvec[56] + 8 * bvec[57] + 8 * bvec[58] - 8 * bvec[59] + 4 * bvec[96] + 4 * bvec[97] - 4 * bvec[98] - 4 * bvec[99] - 4 * bvec[104] - 4 * bvec[105] + 4 * bvec[106] + 4 * bvec[107] + 4 * bvec[128] - 4 * bvec[129] + 4 * bvec[130] - 4 * bvec[131] - 4 * bvec[136] + 4 * bvec[137] - 4 * bvec[138] + 4 * bvec[139] + 4 * bvec[160] - 4 * bvec[161] - 4 * bvec[162] + 4 * bvec[163] + 4 * bvec[168] - 4 * bvec[169] - 4 * bvec[170] + 4 * bvec[171] + 2 * bvec[176] + 2 * bvec[177] + 2 * bvec[178] + 2 * bvec[179] - 2 * bvec[184] - 2 * bvec[185] - 2 * bvec[186] - 2 * bvec[187] + 2 * bvec[208] + 2 * bvec[209] - 2 * bvec[210] - 2 * bvec[211] + 2 * bvec[216] + 2 * bvec[217] - 2 * bvec[218] - 2 * bvec[219] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[226] - 2 * bvec[227] + 2 * bvec[232] - 2 * bvec[233] + 2 * bvec[234] - 2 * bvec[235] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[242] + 1 * bvec[243] + 1 * bvec[248] + 1 * bvec[249] + 1 * bvec[250] + 1 * bvec[251];
       alphavec[224] = -6 * bvec[0] + 6 * bvec[4] + 6 * bvec[8] - 6 * bvec[12] - 4 * bvec[48] - 2 * bvec[52] + 4 * bvec[56] + 2 * bvec[60] - 3 * bvec[64] + 3 * bvec[68] - 3 * bvec[72] + 3 * bvec[76] - 2 * bvec[160] - 1 * bvec[164] - 2 * bvec[168] - 1 * bvec[172];
       alphavec[225] = -6 * bvec[16] + 6 * bvec[20] + 6 * bvec[24] - 6 * bvec[28] - 4 * bvec[96] - 2 * bvec[100] + 4 * bvec[104] + 2 * bvec[108] - 3 * bvec[112] + 3 * bvec[116] - 3 * bvec[120] + 3 * bvec[124] - 2 * bvec[208] - 1 * bvec[212] - 2 * bvec[216] - 1 * bvec[220];
       alphavec[226] = +18 * bvec[0] - 18 * bvec[1] - 18 * bvec[4] + 18 * bvec[5] - 18 * bvec[8] + 18 * bvec[9] + 18 * bvec[12] - 18 * bvec[13] + 12 * bvec[16] + 6 * bvec[17] - 12 * bvec[20] - 6 * bvec[21] - 12 * bvec[24] - 6 * bvec[25] + 12 * bvec[28] + 6 * bvec[29] + 12 * bvec[48] - 12 * bvec[49] + 6 * bvec[52] - 6 * bvec[53] - 12 * bvec[56] + 12 * bvec[57] - 6 * bvec[60] + 6 * bvec[61] + 9 * bvec[64] - 9 * bvec[65] - 9 * bvec[68] + 9 * bvec[69] + 9 * bvec[72] - 9 * bvec[73] - 9 * bvec[76] + 9 * bvec[77] + 8 * bvec[96] + 4 * bvec[97] + 4 * bvec[100] + 2 * bvec[101] - 8 * bvec[104] - 4 * bvec[105] - 4 * bvec[108] - 2 * bvec[109] + 6 * bvec[112] + 3 * bvec[113] - 6 * bvec[116] - 3 * bvec[117] + 6 * bvec[120] + 3 * bvec[121] - 6 * bvec[124] - 3 * bvec[125] + 6 * bvec[160] - 6 * bvec[161] + 3 * bvec[164] - 3 * bvec[165] + 6 * bvec[168] - 6 * bvec[169] + 3 * bvec[172] - 3 * bvec[173] + 4 * bvec[208] + 2 * bvec[209] + 2 * bvec[212] + 1 * bvec[213] + 4 * bvec[216] + 2 * bvec[217] + 2 * bvec[220] + 1 * bvec[221];
       alphavec[227] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[4] - 12 * bvec[5] + 12 * bvec[8] - 12 * bvec[9] - 12 * bvec[12] + 12 * bvec[13] - 6 * bvec[16] - 6 * bvec[17] + 6 * bvec[20] + 6 * bvec[21] + 6 * bvec[24] + 6 * bvec[25] - 6 * bvec[28] - 6 * bvec[29] - 8 * bvec[48] + 8 * bvec[49] - 4 * bvec[52] + 4 * bvec[53] + 8 * bvec[56] - 8 * bvec[57] + 4 * bvec[60] - 4 * bvec[61] - 6 * bvec[64] + 6 * bvec[65] + 6 * bvec[68] - 6 * bvec[69] - 6 * bvec[72] + 6 * bvec[73] + 6 * bvec[76] - 6 * bvec[77] - 4 * bvec[96] - 4 * bvec[97] - 2 * bvec[100] - 2 * bvec[101] + 4 * bvec[104] + 4 * bvec[105] + 2 * bvec[108] + 2 * bvec[109] - 3 * bvec[112] - 3 * bvec[113] + 3 * bvec[116] + 3 * bvec[117] - 3 * bvec[120] - 3 * bvec[121] + 3 * bvec[124] + 3 * bvec[125] - 4 * bvec[160] + 4 * bvec[161] - 2 * bvec[164] + 2 * bvec[165] - 4 * bvec[168] + 4 * bvec[169] - 2 * bvec[172] + 2 * bvec[173] - 2 * bvec[208] - 2 * bvec[209] - 1 * bvec[212] - 1 * bvec[213] - 2 * bvec[216] - 2 * bvec[217] - 1 * bvec[220] - 1 * bvec[221];
       alphavec[228] = -6 * bvec[32] + 6 * bvec[36] + 6 * bvec[40] - 6 * bvec[44] - 4 * bvec[128] - 2 * bvec[132] + 4 * bvec[136] + 2 * bvec[140] - 3 * bvec[144] + 3 * bvec[148] - 3 * bvec[152] + 3 * bvec[156] - 2 * bvec[224] - 1 * bvec[228] - 2 * bvec[232] - 1 * bvec[236];
       alphavec[229] = -6 * bvec[80] + 6 * bvec[84] + 6 * bvec[88] - 6 * bvec[92] - 4 * bvec[176] - 2 * bvec[180] + 4 * bvec[184] + 2 * bvec[188] - 3 * bvec[192] + 3 * bvec[196] - 3 * bvec[200] + 3 * bvec[204] - 2 * bvec[240] - 1 * bvec[244] - 2 * bvec[248] - 1 * bvec[252];
       alphavec[230] = +18 * bvec[32] - 18 * bvec[33] - 18 * bvec[36] + 18 * bvec[37] - 18 * bvec[40] + 18 * bvec[41] + 18 * bvec[44] - 18 * bvec[45] + 12 * bvec[80] + 6 * bvec[81] - 12 * bvec[84] - 6 * bvec[85] - 12 * bvec[88] - 6 * bvec[89] + 12 * bvec[92] + 6 * bvec[93] + 12 * bvec[128] - 12 * bvec[129] + 6 * bvec[132] - 6 * bvec[133] - 12 * bvec[136] + 12 * bvec[137] - 6 * bvec[140] + 6 * bvec[141] + 9 * bvec[144] - 9 * bvec[145] - 9 * bvec[148] + 9 * bvec[149] + 9 * bvec[152] - 9 * bvec[153] - 9 * bvec[156] + 9 * bvec[157] + 8 * bvec[176] + 4 * bvec[177] + 4 * bvec[180] + 2 * bvec[181] - 8 * bvec[184] - 4 * bvec[185] - 4 * bvec[188] - 2 * bvec[189] + 6 * bvec[192] + 3 * bvec[193] - 6 * bvec[196] - 3 * bvec[197] + 6 * bvec[200] + 3 * bvec[201] - 6 * bvec[204] - 3 * bvec[205] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[228] - 3 * bvec[229] + 6 * bvec[232] - 6 * bvec[233] + 3 * bvec[236] - 3 * bvec[237] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[244] + 1 * bvec[245] + 4 * bvec[248] + 2 * bvec[249] + 2 * bvec[252] + 1 * bvec[253];
       alphavec[231] = -12 * bvec[32] + 12 * bvec[33] + 12 * bvec[36] - 12 * bvec[37] + 12 * bvec[40] - 12 * bvec[41] - 12 * bvec[44] + 12 * bvec[45] - 6 * bvec[80] - 6 * bvec[81] + 6 * bvec[84] + 6 * bvec[85] + 6 * bvec[88] + 6 * bvec[89] - 6 * bvec[92] - 6 * bvec[93] - 8 * bvec[128] + 8 * bvec[129] - 4 * bvec[132] + 4 * bvec[133] + 8 * bvec[136] - 8 * bvec[137] + 4 * bvec[140] - 4 * bvec[141] - 6 * bvec[144] + 6 * bvec[145] + 6 * bvec[148] - 6 * bvec[149] - 6 * bvec[152] + 6 * bvec[153] + 6 * bvec[156] - 6 * bvec[157] - 4 * bvec[176] - 4 * bvec[177] - 2 * bvec[180] - 2 * bvec[181] + 4 * bvec[184] + 4 * bvec[185] + 2 * bvec[188] + 2 * bvec[189] - 3 * bvec[192] - 3 * bvec[193] + 3 * bvec[196] + 3 * bvec[197] - 3 * bvec[200] - 3 * bvec[201] + 3 * bvec[204] + 3 * bvec[205] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[228] + 2 * bvec[229] - 4 * bvec[232] + 4 * bvec[233] - 2 * bvec[236] + 2 * bvec[237] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[244] - 1 * bvec[245] - 2 * bvec[248] - 2 * bvec[249] - 1 * bvec[252] - 1 * bvec[253];
       alphavec[232] = +18 * bvec[0] - 18 * bvec[2] - 18 * bvec[4] + 18 * bvec[6] - 18 * bvec[8] + 18 * bvec[10] + 18 * bvec[12] - 18 * bvec[14] + 12 * bvec[32] + 6 * bvec[34] - 12 * bvec[36] - 6 * bvec[38] - 12 * bvec[40] - 6 * bvec[42] + 12 * bvec[44] + 6 * bvec[46] + 12 * bvec[48] - 12 * bvec[50] + 6 * bvec[52] - 6 * bvec[54] - 12 * bvec[56] + 12 * bvec[58] - 6 * bvec[60] + 6 * bvec[62] + 9 * bvec[64] - 9 * bvec[66] - 9 * bvec[68] + 9 * bvec[70] + 9 * bvec[72] - 9 * bvec[74] - 9 * bvec[76] + 9 * bvec[78] + 8 * bvec[128] + 4 * bvec[130] + 4 * bvec[132] + 2 * bvec[134] - 8 * bvec[136] - 4 * bvec[138] - 4 * bvec[140] - 2 * bvec[142] + 6 * bvec[144] + 3 * bvec[146] - 6 * bvec[148] - 3 * bvec[150] + 6 * bvec[152] + 3 * bvec[154] - 6 * bvec[156] - 3 * bvec[158] + 6 * bvec[160] - 6 * bvec[162] + 3 * bvec[164] - 3 * bvec[166] + 6 * bvec[168] - 6 * bvec[170] + 3 * bvec[172] - 3 * bvec[174] + 4 * bvec[224] + 2 * bvec[226] + 2 * bvec[228] + 1 * bvec[230] + 4 * bvec[232] + 2 * bvec[234] + 2 * bvec[236] + 1 * bvec[238];
       alphavec[233] = +18 * bvec[16] - 18 * bvec[18] - 18 * bvec[20] + 18 * bvec[22] - 18 * bvec[24] + 18 * bvec[26] + 18 * bvec[28] - 18 * bvec[30] + 12 * bvec[80] + 6 * bvec[82] - 12 * bvec[84] - 6 * bvec[86] - 12 * bvec[88] - 6 * bvec[90] + 12 * bvec[92] + 6 * bvec[94] + 12 * bvec[96] - 12 * bvec[98] + 6 * bvec[100] - 6 * bvec[102] - 12 * bvec[104] + 12 * bvec[106] - 6 * bvec[108] + 6 * bvec[110] + 9 * bvec[112] - 9 * bvec[114] - 9 * bvec[116] + 9 * bvec[118] + 9 * bvec[120] - 9 * bvec[122] - 9 * bvec[124] + 9 * bvec[126] + 8 * bvec[176] + 4 * bvec[178] + 4 * bvec[180] + 2 * bvec[182] - 8 * bvec[184] - 4 * bvec[186] - 4 * bvec[188] - 2 * bvec[190] + 6 * bvec[192] + 3 * bvec[194] - 6 * bvec[196] - 3 * bvec[198] + 6 * bvec[200] + 3 * bvec[202] - 6 * bvec[204] - 3 * bvec[206] + 6 * bvec[208] - 6 * bvec[210] + 3 * bvec[212] - 3 * bvec[214] + 6 * bvec[216] - 6 * bvec[218] + 3 * bvec[220] - 3 * bvec[222] + 4 * bvec[240] + 2 * bvec[242] + 2 * bvec[244] + 1 * bvec[246] + 4 * bvec[248] + 2 * bvec[250] + 2 * bvec[252] + 1 * bvec[254];
       alphavec[234] = -54 * bvec[0] + 54 * bvec[1] + 54 * bvec[2] - 54 * bvec[3] + 54 * bvec[4] - 54 * bvec[5] - 54 * bvec[6] + 54 * bvec[7] + 54 * bvec[8] - 54 * bvec[9] - 54 * bvec[10] + 54 * bvec[11] - 54 * bvec[12] + 54 * bvec[13] + 54 * bvec[14] - 54 * bvec[15] - 36 * bvec[16] - 18 * bvec[17] + 36 * bvec[18] + 18 * bvec[19] + 36 * bvec[20] + 18 * bvec[21] - 36 * bvec[22] - 18 * bvec[23] + 36 * bvec[24] + 18 * bvec[25] - 36 * bvec[26] - 18 * bvec[27] - 36 * bvec[28] - 18 * bvec[29] + 36 * bvec[30] + 18 * bvec[31] - 36 * bvec[32] + 36 * bvec[33] - 18 * bvec[34] + 18 * bvec[35] + 36 * bvec[36] - 36 * bvec[37] + 18 * bvec[38] - 18 * bvec[39] + 36 * bvec[40] - 36 * bvec[41] + 18 * bvec[42] - 18 * bvec[43] - 36 * bvec[44] + 36 * bvec[45] - 18 * bvec[46] + 18 * bvec[47] - 36 * bvec[48] + 36 * bvec[49] + 36 * bvec[50] - 36 * bvec[51] - 18 * bvec[52] + 18 * bvec[53] + 18 * bvec[54] - 18 * bvec[55] + 36 * bvec[56] - 36 * bvec[57] - 36 * bvec[58] + 36 * bvec[59] + 18 * bvec[60] - 18 * bvec[61] - 18 * bvec[62] + 18 * bvec[63] - 27 * bvec[64] + 27 * bvec[65] + 27 * bvec[66] - 27 * bvec[67] + 27 * bvec[68] - 27 * bvec[69] - 27 * bvec[70] + 27 * bvec[71] - 27 * bvec[72] + 27 * bvec[73] + 27 * bvec[74] - 27 * bvec[75] + 27 * bvec[76] - 27 * bvec[77] - 27 * bvec[78] + 27 * bvec[79] - 24 * bvec[80] - 12 * bvec[81] - 12 * bvec[82] - 6 * bvec[83] + 24 * bvec[84] + 12 * bvec[85] + 12 * bvec[86] + 6 * bvec[87] + 24 * bvec[88] + 12 * bvec[89] + 12 * bvec[90] + 6 * bvec[91] - 24 * bvec[92] - 12 * bvec[93] - 12 * bvec[94] - 6 * bvec[95] - 24 * bvec[96] - 12 * bvec[97] + 24 * bvec[98] + 12 * bvec[99] - 12 * bvec[100] - 6 * bvec[101] + 12 * bvec[102] + 6 * bvec[103] + 24 * bvec[104] + 12 * bvec[105] - 24 * bvec[106] - 12 * bvec[107] + 12 * bvec[108] + 6 * bvec[109] - 12 * bvec[110] - 6 * bvec[111] - 18 * bvec[112] - 9 * bvec[113] + 18 * bvec[114] + 9 * bvec[115] + 18 * bvec[116] + 9 * bvec[117] - 18 * bvec[118] - 9 * bvec[119] - 18 * bvec[120] - 9 * bvec[121] + 18 * bvec[122] + 9 * bvec[123] + 18 * bvec[124] + 9 * bvec[125] - 18 * bvec[126] - 9 * bvec[127] - 24 * bvec[128] + 24 * bvec[129] - 12 * bvec[130] + 12 * bvec[131] - 12 * bvec[132] + 12 * bvec[133] - 6 * bvec[134] + 6 * bvec[135] + 24 * bvec[136] - 24 * bvec[137] + 12 * bvec[138] - 12 * bvec[139] + 12 * bvec[140] - 12 * bvec[141] + 6 * bvec[142] - 6 * bvec[143] - 18 * bvec[144] + 18 * bvec[145] - 9 * bvec[146] + 9 * bvec[147] + 18 * bvec[148] - 18 * bvec[149] + 9 * bvec[150] - 9 * bvec[151] - 18 * bvec[152] + 18 * bvec[153] - 9 * bvec[154] + 9 * bvec[155] + 18 * bvec[156] - 18 * bvec[157] + 9 * bvec[158] - 9 * bvec[159] - 18 * bvec[160] + 18 * bvec[161] + 18 * bvec[162] - 18 * bvec[163] - 9 * bvec[164] + 9 * bvec[165] + 9 * bvec[166] - 9 * bvec[167] - 18 * bvec[168] + 18 * bvec[169] + 18 * bvec[170] - 18 * bvec[171] - 9 * bvec[172] + 9 * bvec[173] + 9 * bvec[174] - 9 * bvec[175] - 16 * bvec[176] - 8 * bvec[177] - 8 * bvec[178] - 4 * bvec[179] - 8 * bvec[180] - 4 * bvec[181] - 4 * bvec[182] - 2 * bvec[183] + 16 * bvec[184] + 8 * bvec[185] + 8 * bvec[186] + 4 * bvec[187] + 8 * bvec[188] + 4 * bvec[189] + 4 * bvec[190] + 2 * bvec[191] - 12 * bvec[192] - 6 * bvec[193] - 6 * bvec[194] - 3 * bvec[195] + 12 * bvec[196] + 6 * bvec[197] + 6 * bvec[198] + 3 * bvec[199] - 12 * bvec[200] - 6 * bvec[201] - 6 * bvec[202] - 3 * bvec[203] + 12 * bvec[204] + 6 * bvec[205] + 6 * bvec[206] + 3 * bvec[207] - 12 * bvec[208] - 6 * bvec[209] + 12 * bvec[210] + 6 * bvec[211] - 6 * bvec[212] - 3 * bvec[213] + 6 * bvec[214] + 3 * bvec[215] - 12 * bvec[216] - 6 * bvec[217] + 12 * bvec[218] + 6 * bvec[219] - 6 * bvec[220] - 3 * bvec[221] + 6 * bvec[222] + 3 * bvec[223] - 12 * bvec[224] + 12 * bvec[225] - 6 * bvec[226] + 6 * bvec[227] - 6 * bvec[228] + 6 * bvec[229] - 3 * bvec[230] + 3 * bvec[231] - 12 * bvec[232] + 12 * bvec[233] - 6 * bvec[234] + 6 * bvec[235] - 6 * bvec[236] + 6 * bvec[237] - 3 * bvec[238] + 3 * bvec[239] - 8 * bvec[240] - 4 * bvec[241] - 4 * bvec[242] - 2 * bvec[243] - 4 * bvec[244] - 2 * bvec[245] - 2 * bvec[246] - 1 * bvec[247] - 8 * bvec[248] - 4 * bvec[249] - 4 * bvec[250] - 2 * bvec[251] - 4 * bvec[252] - 2 * bvec[253] - 2 * bvec[254] - 1 * bvec[255];
       alphavec[235] = +36 * bvec[0] - 36 * bvec[1] - 36 * bvec[2] + 36 * bvec[3] - 36 * bvec[4] + 36 * bvec[5] + 36 * bvec[6] - 36 * bvec[7] - 36 * bvec[8] + 36 * bvec[9] + 36 * bvec[10] - 36 * bvec[11] + 36 * bvec[12] - 36 * bvec[13] - 36 * bvec[14] + 36 * bvec[15] + 18 * bvec[16] + 18 * bvec[17] - 18 * bvec[18] - 18 * bvec[19] - 18 * bvec[20] - 18 * bvec[21] + 18 * bvec[22] + 18 * bvec[23] - 18 * bvec[24] - 18 * bvec[25] + 18 * bvec[26] + 18 * bvec[27] + 18 * bvec[28] + 18 * bvec[29] - 18 * bvec[30] - 18 * bvec[31] + 24 * bvec[32] - 24 * bvec[33] + 12 * bvec[34] - 12 * bvec[35] - 24 * bvec[36] + 24 * bvec[37] - 12 * bvec[38] + 12 * bvec[39] - 24 * bvec[40] + 24 * bvec[41] - 12 * bvec[42] + 12 * bvec[43] + 24 * bvec[44] - 24 * bvec[45] + 12 * bvec[46] - 12 * bvec[47] + 24 * bvec[48] - 24 * bvec[49] - 24 * bvec[50] + 24 * bvec[51] + 12 * bvec[52] - 12 * bvec[53] - 12 * bvec[54] + 12 * bvec[55] - 24 * bvec[56] + 24 * bvec[57] + 24 * bvec[58] - 24 * bvec[59] - 12 * bvec[60] + 12 * bvec[61] + 12 * bvec[62] - 12 * bvec[63] + 18 * bvec[64] - 18 * bvec[65] - 18 * bvec[66] + 18 * bvec[67] - 18 * bvec[68] + 18 * bvec[69] + 18 * bvec[70] - 18 * bvec[71] + 18 * bvec[72] - 18 * bvec[73] - 18 * bvec[74] + 18 * bvec[75] - 18 * bvec[76] + 18 * bvec[77] + 18 * bvec[78] - 18 * bvec[79] + 12 * bvec[80] + 12 * bvec[81] + 6 * bvec[82] + 6 * bvec[83] - 12 * bvec[84] - 12 * bvec[85] - 6 * bvec[86] - 6 * bvec[87] - 12 * bvec[88] - 12 * bvec[89] - 6 * bvec[90] - 6 * bvec[91] + 12 * bvec[92] + 12 * bvec[93] + 6 * bvec[94] + 6 * bvec[95] + 12 * bvec[96] + 12 * bvec[97] - 12 * bvec[98] - 12 * bvec[99] + 6 * bvec[100] + 6 * bvec[101] - 6 * bvec[102] - 6 * bvec[103] - 12 * bvec[104] - 12 * bvec[105] + 12 * bvec[106] + 12 * bvec[107] - 6 * bvec[108] - 6 * bvec[109] + 6 * bvec[110] + 6 * bvec[111] + 9 * bvec[112] + 9 * bvec[113] - 9 * bvec[114] - 9 * bvec[115] - 9 * bvec[116] - 9 * bvec[117] + 9 * bvec[118] + 9 * bvec[119] + 9 * bvec[120] + 9 * bvec[121] - 9 * bvec[122] - 9 * bvec[123] - 9 * bvec[124] - 9 * bvec[125] + 9 * bvec[126] + 9 * bvec[127] + 16 * bvec[128] - 16 * bvec[129] + 8 * bvec[130] - 8 * bvec[131] + 8 * bvec[132] - 8 * bvec[133] + 4 * bvec[134] - 4 * bvec[135] - 16 * bvec[136] + 16 * bvec[137] - 8 * bvec[138] + 8 * bvec[139] - 8 * bvec[140] + 8 * bvec[141] - 4 * bvec[142] + 4 * bvec[143] + 12 * bvec[144] - 12 * bvec[145] + 6 * bvec[146] - 6 * bvec[147] - 12 * bvec[148] + 12 * bvec[149] - 6 * bvec[150] + 6 * bvec[151] + 12 * bvec[152] - 12 * bvec[153] + 6 * bvec[154] - 6 * bvec[155] - 12 * bvec[156] + 12 * bvec[157] - 6 * bvec[158] + 6 * bvec[159] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 6 * bvec[164] - 6 * bvec[165] - 6 * bvec[166] + 6 * bvec[167] + 12 * bvec[168] - 12 * bvec[169] - 12 * bvec[170] + 12 * bvec[171] + 6 * bvec[172] - 6 * bvec[173] - 6 * bvec[174] + 6 * bvec[175] + 8 * bvec[176] + 8 * bvec[177] + 4 * bvec[178] + 4 * bvec[179] + 4 * bvec[180] + 4 * bvec[181] + 2 * bvec[182] + 2 * bvec[183] - 8 * bvec[184] - 8 * bvec[185] - 4 * bvec[186] - 4 * bvec[187] - 4 * bvec[188] - 4 * bvec[189] - 2 * bvec[190] - 2 * bvec[191] + 6 * bvec[192] + 6 * bvec[193] + 3 * bvec[194] + 3 * bvec[195] - 6 * bvec[196] - 6 * bvec[197] - 3 * bvec[198] - 3 * bvec[199] + 6 * bvec[200] + 6 * bvec[201] + 3 * bvec[202] + 3 * bvec[203] - 6 * bvec[204] - 6 * bvec[205] - 3 * bvec[206] - 3 * bvec[207] + 6 * bvec[208] + 6 * bvec[209] - 6 * bvec[210] - 6 * bvec[211] + 3 * bvec[212] + 3 * bvec[213] - 3 * bvec[214] - 3 * bvec[215] + 6 * bvec[216] + 6 * bvec[217] - 6 * bvec[218] - 6 * bvec[219] + 3 * bvec[220] + 3 * bvec[221] - 3 * bvec[222] - 3 * bvec[223] + 8 * bvec[224] - 8 * bvec[225] + 4 * bvec[226] - 4 * bvec[227] + 4 * bvec[228] - 4 * bvec[229] + 2 * bvec[230] - 2 * bvec[231] + 8 * bvec[232] - 8 * bvec[233] + 4 * bvec[234] - 4 * bvec[235] + 4 * bvec[236] - 4 * bvec[237] + 2 * bvec[238] - 2 * bvec[239] + 4 * bvec[240] + 4 * bvec[241] + 2 * bvec[242] + 2 * bvec[243] + 2 * bvec[244] + 2 * bvec[245] + 1 * bvec[246] + 1 * bvec[247] + 4 * bvec[248] + 4 * bvec[249] + 2 * bvec[250] + 2 * bvec[251] + 2 * bvec[252] + 2 * bvec[253] + 1 * bvec[254] + 1 * bvec[255];
       alphavec[236] = -12 * bvec[0] + 12 * bvec[2] + 12 * bvec[4] - 12 * bvec[6] + 12 * bvec[8] - 12 * bvec[10] - 12 * bvec[12] + 12 * bvec[14] - 6 * bvec[32] - 6 * bvec[34] + 6 * bvec[36] + 6 * bvec[38] + 6 * bvec[40] + 6 * bvec[42] - 6 * bvec[44] - 6 * bvec[46] - 8 * bvec[48] + 8 * bvec[50] - 4 * bvec[52] + 4 * bvec[54] + 8 * bvec[56] - 8 * bvec[58] + 4 * bvec[60] - 4 * bvec[62] - 6 * bvec[64] + 6 * bvec[66] + 6 * bvec[68] - 6 * bvec[70] - 6 * bvec[72] + 6 * bvec[74] + 6 * bvec[76] - 6 * bvec[78] - 4 * bvec[128] - 4 * bvec[130] - 2 * bvec[132] - 2 * bvec[134] + 4 * bvec[136] + 4 * bvec[138] + 2 * bvec[140] + 2 * bvec[142] - 3 * bvec[144] - 3 * bvec[146] + 3 * bvec[148] + 3 * bvec[150] - 3 * bvec[152] - 3 * bvec[154] + 3 * bvec[156] + 3 * bvec[158] - 4 * bvec[160] + 4 * bvec[162] - 2 * bvec[164] + 2 * bvec[166] - 4 * bvec[168] + 4 * bvec[170] - 2 * bvec[172] + 2 * bvec[174] - 2 * bvec[224] - 2 * bvec[226] - 1 * bvec[228] - 1 * bvec[230] - 2 * bvec[232] - 2 * bvec[234] - 1 * bvec[236] - 1 * bvec[238];
       alphavec[237] = -12 * bvec[16] + 12 * bvec[18] + 12 * bvec[20] - 12 * bvec[22] + 12 * bvec[24] - 12 * bvec[26] - 12 * bvec[28] + 12 * bvec[30] - 6 * bvec[80] - 6 * bvec[82] + 6 * bvec[84] + 6 * bvec[86] + 6 * bvec[88] + 6 * bvec[90] - 6 * bvec[92] - 6 * bvec[94] - 8 * bvec[96] + 8 * bvec[98] - 4 * bvec[100] + 4 * bvec[102] + 8 * bvec[104] - 8 * bvec[106] + 4 * bvec[108] - 4 * bvec[110] - 6 * bvec[112] + 6 * bvec[114] + 6 * bvec[116] - 6 * bvec[118] - 6 * bvec[120] + 6 * bvec[122] + 6 * bvec[124] - 6 * bvec[126] - 4 * bvec[176] - 4 * bvec[178] - 2 * bvec[180] - 2 * bvec[182] + 4 * bvec[184] + 4 * bvec[186] + 2 * bvec[188] + 2 * bvec[190] - 3 * bvec[192] - 3 * bvec[194] + 3 * bvec[196] + 3 * bvec[198] - 3 * bvec[200] - 3 * bvec[202] + 3 * bvec[204] + 3 * bvec[206] - 4 * bvec[208] + 4 * bvec[210] - 2 * bvec[212] + 2 * bvec[214] - 4 * bvec[216] + 4 * bvec[218] - 2 * bvec[220] + 2 * bvec[222] - 2 * bvec[240] - 2 * bvec[242] - 1 * bvec[244] - 1 * bvec[246] - 2 * bvec[248] - 2 * bvec[250] - 1 * bvec[252] - 1 * bvec[254];
       alphavec[238] = +36 * bvec[0] - 36 * bvec[1] - 36 * bvec[2] + 36 * bvec[3] - 36 * bvec[4] + 36 * bvec[5] + 36 * bvec[6] - 36 * bvec[7] - 36 * bvec[8] + 36 * bvec[9] + 36 * bvec[10] - 36 * bvec[11] + 36 * bvec[12] - 36 * bvec[13] - 36 * bvec[14] + 36 * bvec[15] + 24 * bvec[16] + 12 * bvec[17] - 24 * bvec[18] - 12 * bvec[19] - 24 * bvec[20] - 12 * bvec[21] + 24 * bvec[22] + 12 * bvec[23] - 24 * bvec[24] - 12 * bvec[25] + 24 * bvec[26] + 12 * bvec[27] + 24 * bvec[28] + 12 * bvec[29] - 24 * bvec[30] - 12 * bvec[31] + 18 * bvec[32] - 18 * bvec[33] + 18 * bvec[34] - 18 * bvec[35] - 18 * bvec[36] + 18 * bvec[37] - 18 * bvec[38] + 18 * bvec[39] - 18 * bvec[40] + 18 * bvec[41] - 18 * bvec[42] + 18 * bvec[43] + 18 * bvec[44] - 18 * bvec[45] + 18 * bvec[46] - 18 * bvec[47] + 24 * bvec[48] - 24 * bvec[49] - 24 * bvec[50] + 24 * bvec[51] + 12 * bvec[52] - 12 * bvec[53] - 12 * bvec[54] + 12 * bvec[55] - 24 * bvec[56] + 24 * bvec[57] + 24 * bvec[58] - 24 * bvec[59] - 12 * bvec[60] + 12 * bvec[61] + 12 * bvec[62] - 12 * bvec[63] + 18 * bvec[64] - 18 * bvec[65] - 18 * bvec[66] + 18 * bvec[67] - 18 * bvec[68] + 18 * bvec[69] + 18 * bvec[70] - 18 * bvec[71] + 18 * bvec[72] - 18 * bvec[73] - 18 * bvec[74] + 18 * bvec[75] - 18 * bvec[76] + 18 * bvec[77] + 18 * bvec[78] - 18 * bvec[79] + 12 * bvec[80] + 6 * bvec[81] + 12 * bvec[82] + 6 * bvec[83] - 12 * bvec[84] - 6 * bvec[85] - 12 * bvec[86] - 6 * bvec[87] - 12 * bvec[88] - 6 * bvec[89] - 12 * bvec[90] - 6 * bvec[91] + 12 * bvec[92] + 6 * bvec[93] + 12 * bvec[94] + 6 * bvec[95] + 16 * bvec[96] + 8 * bvec[97] - 16 * bvec[98] - 8 * bvec[99] + 8 * bvec[100] + 4 * bvec[101] - 8 * bvec[102] - 4 * bvec[103] - 16 * bvec[104] - 8 * bvec[105] + 16 * bvec[106] + 8 * bvec[107] - 8 * bvec[108] - 4 * bvec[109] + 8 * bvec[110] + 4 * bvec[111] + 12 * bvec[112] + 6 * bvec[113] - 12 * bvec[114] - 6 * bvec[115] - 12 * bvec[116] - 6 * bvec[117] + 12 * bvec[118] + 6 * bvec[119] + 12 * bvec[120] + 6 * bvec[121] - 12 * bvec[122] - 6 * bvec[123] - 12 * bvec[124] - 6 * bvec[125] + 12 * bvec[126] + 6 * bvec[127] + 12 * bvec[128] - 12 * bvec[129] + 12 * bvec[130] - 12 * bvec[131] + 6 * bvec[132] - 6 * bvec[133] + 6 * bvec[134] - 6 * bvec[135] - 12 * bvec[136] + 12 * bvec[137] - 12 * bvec[138] + 12 * bvec[139] - 6 * bvec[140] + 6 * bvec[141] - 6 * bvec[142] + 6 * bvec[143] + 9 * bvec[144] - 9 * bvec[145] + 9 * bvec[146] - 9 * bvec[147] - 9 * bvec[148] + 9 * bvec[149] - 9 * bvec[150] + 9 * bvec[151] + 9 * bvec[152] - 9 * bvec[153] + 9 * bvec[154] - 9 * bvec[155] - 9 * bvec[156] + 9 * bvec[157] - 9 * bvec[158] + 9 * bvec[159] + 12 * bvec[160] - 12 * bvec[161] - 12 * bvec[162] + 12 * bvec[163] + 6 * bvec[164] - 6 * bvec[165] - 6 * bvec[166] + 6 * bvec[167] + 12 * bvec[168] - 12 * bvec[169] - 12 * bvec[170] + 12 * bvec[171] + 6 * bvec[172] - 6 * bvec[173] - 6 * bvec[174] + 6 * bvec[175] + 8 * bvec[176] + 4 * bvec[177] + 8 * bvec[178] + 4 * bvec[179] + 4 * bvec[180] + 2 * bvec[181] + 4 * bvec[182] + 2 * bvec[183] - 8 * bvec[184] - 4 * bvec[185] - 8 * bvec[186] - 4 * bvec[187] - 4 * bvec[188] - 2 * bvec[189] - 4 * bvec[190] - 2 * bvec[191] + 6 * bvec[192] + 3 * bvec[193] + 6 * bvec[194] + 3 * bvec[195] - 6 * bvec[196] - 3 * bvec[197] - 6 * bvec[198] - 3 * bvec[199] + 6 * bvec[200] + 3 * bvec[201] + 6 * bvec[202] + 3 * bvec[203] - 6 * bvec[204] - 3 * bvec[205] - 6 * bvec[206] - 3 * bvec[207] + 8 * bvec[208] + 4 * bvec[209] - 8 * bvec[210] - 4 * bvec[211] + 4 * bvec[212] + 2 * bvec[213] - 4 * bvec[214] - 2 * bvec[215] + 8 * bvec[216] + 4 * bvec[217] - 8 * bvec[218] - 4 * bvec[219] + 4 * bvec[220] + 2 * bvec[221] - 4 * bvec[222] - 2 * bvec[223] + 6 * bvec[224] - 6 * bvec[225] + 6 * bvec[226] - 6 * bvec[227] + 3 * bvec[228] - 3 * bvec[229] + 3 * bvec[230] - 3 * bvec[231] + 6 * bvec[232] - 6 * bvec[233] + 6 * bvec[234] - 6 * bvec[235] + 3 * bvec[236] - 3 * bvec[237] + 3 * bvec[238] - 3 * bvec[239] + 4 * bvec[240] + 2 * bvec[241] + 4 * bvec[242] + 2 * bvec[243] + 2 * bvec[244] + 1 * bvec[245] + 2 * bvec[246] + 1 * bvec[247] + 4 * bvec[248] + 2 * bvec[249] + 4 * bvec[250] + 2 * bvec[251] + 2 * bvec[252] + 1 * bvec[253] + 2 * bvec[254] + 1 * bvec[255];
       alphavec[239] = -24 * bvec[0] + 24 * bvec[1] + 24 * bvec[2] - 24 * bvec[3] + 24 * bvec[4] - 24 * bvec[5] - 24 * bvec[6] + 24 * bvec[7] + 24 * bvec[8] - 24 * bvec[9] - 24 * bvec[10] + 24 * bvec[11] - 24 * bvec[12] + 24 * bvec[13] + 24 * bvec[14] - 24 * bvec[15] - 12 * bvec[16] - 12 * bvec[17] + 12 * bvec[18] + 12 * bvec[19] + 12 * bvec[20] + 12 * bvec[21] - 12 * bvec[22] - 12 * bvec[23] + 12 * bvec[24] + 12 * bvec[25] - 12 * bvec[26] - 12 * bvec[27] - 12 * bvec[28] - 12 * bvec[29] + 12 * bvec[30] + 12 * bvec[31] - 12 * bvec[32] + 12 * bvec[33] - 12 * bvec[34] + 12 * bvec[35] + 12 * bvec[36] - 12 * bvec[37] + 12 * bvec[38] - 12 * bvec[39] + 12 * bvec[40] - 12 * bvec[41] + 12 * bvec[42] - 12 * bvec[43] - 12 * bvec[44] + 12 * bvec[45] - 12 * bvec[46] + 12 * bvec[47] - 16 * bvec[48] + 16 * bvec[49] + 16 * bvec[50] - 16 * bvec[51] - 8 * bvec[52] + 8 * bvec[53] + 8 * bvec[54] - 8 * bvec[55] + 16 * bvec[56] - 16 * bvec[57] - 16 * bvec[58] + 16 * bvec[59] + 8 * bvec[60] - 8 * bvec[61] - 8 * bvec[62] + 8 * bvec[63] - 12 * bvec[64] + 12 * bvec[65] + 12 * bvec[66] - 12 * bvec[67] + 12 * bvec[68] - 12 * bvec[69] - 12 * bvec[70] + 12 * bvec[71] - 12 * bvec[72] + 12 * bvec[73] + 12 * bvec[74] - 12 * bvec[75] + 12 * bvec[76] - 12 * bvec[77] - 12 * bvec[78] + 12 * bvec[79] - 6 * bvec[80] - 6 * bvec[81] - 6 * bvec[82] - 6 * bvec[83] + 6 * bvec[84] + 6 * bvec[85] + 6 * bvec[86] + 6 * bvec[87] + 6 * bvec[88] + 6 * bvec[89] + 6 * bvec[90] + 6 * bvec[91] - 6 * bvec[92] - 6 * bvec[93] - 6 * bvec[94] - 6 * bvec[95] - 8 * bvec[96] - 8 * bvec[97] + 8 * bvec[98] + 8 * bvec[99] - 4 * bvec[100] - 4 * bvec[101] + 4 * bvec[102] + 4 * bvec[103] + 8 * bvec[104] + 8 * bvec[105] - 8 * bvec[106] - 8 * bvec[107] + 4 * bvec[108] + 4 * bvec[109] - 4 * bvec[110] - 4 * bvec[111] - 6 * bvec[112] - 6 * bvec[113] + 6 * bvec[114] + 6 * bvec[115] + 6 * bvec[116] + 6 * bvec[117] - 6 * bvec[118] - 6 * bvec[119] - 6 * bvec[120] - 6 * bvec[121] + 6 * bvec[122] + 6 * bvec[123] + 6 * bvec[124] + 6 * bvec[125] - 6 * bvec[126] - 6 * bvec[127] - 8 * bvec[128] + 8 * bvec[129] - 8 * bvec[130] + 8 * bvec[131] - 4 * bvec[132] + 4 * bvec[133] - 4 * bvec[134] + 4 * bvec[135] + 8 * bvec[136] - 8 * bvec[137] + 8 * bvec[138] - 8 * bvec[139] + 4 * bvec[140] - 4 * bvec[141] + 4 * bvec[142] - 4 * bvec[143] - 6 * bvec[144] + 6 * bvec[145] - 6 * bvec[146] + 6 * bvec[147] + 6 * bvec[148] - 6 * bvec[149] + 6 * bvec[150] - 6 * bvec[151] - 6 * bvec[152] + 6 * bvec[153] - 6 * bvec[154] + 6 * bvec[155] + 6 * bvec[156] - 6 * bvec[157] + 6 * bvec[158] - 6 * bvec[159] - 8 * bvec[160] + 8 * bvec[161] + 8 * bvec[162] - 8 * bvec[163] - 4 * bvec[164] + 4 * bvec[165] + 4 * bvec[166] - 4 * bvec[167] - 8 * bvec[168] + 8 * bvec[169] + 8 * bvec[170] - 8 * bvec[171] - 4 * bvec[172] + 4 * bvec[173] + 4 * bvec[174] - 4 * bvec[175] - 4 * bvec[176] - 4 * bvec[177] - 4 * bvec[178] - 4 * bvec[179] - 2 * bvec[180] - 2 * bvec[181] - 2 * bvec[182] - 2 * bvec[183] + 4 * bvec[184] + 4 * bvec[185] + 4 * bvec[186] + 4 * bvec[187] + 2 * bvec[188] + 2 * bvec[189] + 2 * bvec[190] + 2 * bvec[191] - 3 * bvec[192] - 3 * bvec[193] - 3 * bvec[194] - 3 * bvec[195] + 3 * bvec[196] + 3 * bvec[197] + 3 * bvec[198] + 3 * bvec[199] - 3 * bvec[200] - 3 * bvec[201] - 3 * bvec[202] - 3 * bvec[203] + 3 * bvec[204] + 3 * bvec[205] + 3 * bvec[206] + 3 * bvec[207] - 4 * bvec[208] - 4 * bvec[209] + 4 * bvec[210] + 4 * bvec[211] - 2 * bvec[212] - 2 * bvec[213] + 2 * bvec[214] + 2 * bvec[215] - 4 * bvec[216] - 4 * bvec[217] + 4 * bvec[218] + 4 * bvec[219] - 2 * bvec[220] - 2 * bvec[221] + 2 * bvec[222] + 2 * bvec[223] - 4 * bvec[224] + 4 * bvec[225] - 4 * bvec[226] + 4 * bvec[227] - 2 * bvec[228] + 2 * bvec[229] - 2 * bvec[230] + 2 * bvec[231] - 4 * bvec[232] + 4 * bvec[233] - 4 * bvec[234] + 4 * bvec[235] - 2 * bvec[236] + 2 * bvec[237] - 2 * bvec[238] + 2 * bvec[239] - 2 * bvec[240] - 2 * bvec[241] - 2 * bvec[242] - 2 * bvec[243] - 1 * bvec[244] - 1 * bvec[245] - 1 * bvec[246] - 1 * bvec[247] - 2 * bvec[248] - 2 * bvec[249] - 2 * bvec[250] - 2 * bvec[251] - 1 * bvec[252] - 1 * bvec[253] - 1 * bvec[254] - 1 * bvec[255];
       alphavec[240] = +4 * bvec[0] - 4 * bvec[4] - 4 * bvec[8] + 4 * bvec[12] + 2 * bvec[48] + 2 * bvec[52] - 2 * bvec[56] - 2 * bvec[60] + 2 * bvec[64] - 2 * bvec[68] + 2 * bvec[72] - 2 * bvec[76] + 1 * bvec[160] + 1 * bvec[164] + 1 * bvec[168] + 1 * bvec[172];
       alphavec[241] = +4 * bvec[16] - 4 * bvec[20] - 4 * bvec[24] + 4 * bvec[28] + 2 * bvec[96] + 2 * bvec[100] - 2 * bvec[104] - 2 * bvec[108] + 2 * bvec[112] - 2 * bvec[116] + 2 * bvec[120] - 2 * bvec[124] + 1 * bvec[208] + 1 * bvec[212] + 1 * bvec[216] + 1 * bvec[220];
       alphavec[242] = -12 * bvec[0] + 12 * bvec[1] + 12 * bvec[4] - 12 * bvec[5] + 12 * bvec[8] - 12 * bvec[9] - 12 * bvec[12] + 12 * bvec[13] - 8 * bvec[16] - 4 * bvec[17] + 8 * bvec[20] + 4 * bvec[21] + 8 * bvec[24] + 4 * bvec[25] - 8 * bvec[28] - 4 * bvec[29] - 6 * bvec[48] + 6 * bvec[49] - 6 * bvec[52] + 6 * bvec[53] + 6 * bvec[56] - 6 * bvec[57] + 6 * bvec[60] - 6 * bvec[61] - 6 * bvec[64] + 6 * bvec[65] + 6 * bvec[68] - 6 * bvec[69] - 6 * bvec[72] + 6 * bvec[73] + 6 * bvec[76] - 6 * bvec[77] - 4 * bvec[96] - 2 * bvec[97] - 4 * bvec[100] - 2 * bvec[101] + 4 * bvec[104] + 2 * bvec[105] + 4 * bvec[108] + 2 * bvec[109] - 4 * bvec[112] - 2 * bvec[113] + 4 * bvec[116] + 2 * bvec[117] - 4 * bvec[120] - 2 * bvec[121] + 4 * bvec[124] + 2 * bvec[125] - 3 * bvec[160] + 3 * bvec[161] - 3 * bvec[164] + 3 * bvec[165] - 3 * bvec[168] + 3 * bvec[169] - 3 * bvec[172] + 3 * bvec[173] - 2 * bvec[208] - 1 * bvec[209] - 2 * bvec[212] - 1 * bvec[213] - 2 * bvec[216] - 1 * bvec[217] - 2 * bvec[220] - 1 * bvec[221];
       alphavec[243] = +8 * bvec[0] - 8 * bvec[1] - 8 * bvec[4] + 8 * bvec[5] - 8 * bvec[8] + 8 * bvec[9] + 8 * bvec[12] - 8 * bvec[13] + 4 * bvec[16] + 4 * bvec[17] - 4 * bvec[20] - 4 * bvec[21] - 4 * bvec[24] - 4 * bvec[25] + 4 * bvec[28] + 4 * bvec[29] + 4 * bvec[48] - 4 * bvec[49] + 4 * bvec[52] - 4 * bvec[53] - 4 * bvec[56] + 4 * bvec[57] - 4 * bvec[60] + 4 * bvec[61] + 4 * bvec[64] - 4 * bvec[65] - 4 * bvec[68] + 4 * bvec[69] + 4 * bvec[72] - 4 * bvec[73] - 4 * bvec[76] + 4 * bvec[77] + 2 * bvec[96] + 2 * bvec[97] + 2 * bvec[100] + 2 * bvec[101] - 2 * bvec[104] - 2 * bvec[105] - 2 * bvec[108] - 2 * bvec[109] + 2 * bvec[112] + 2 * bvec[113] - 2 * bvec[116] - 2 * bvec[117] + 2 * bvec[120] + 2 * bvec[121] - 2 * bvec[124] - 2 * bvec[125] + 2 * bvec[160] - 2 * bvec[161] + 2 * bvec[164] - 2 * bvec[165] + 2 * bvec[168] - 2 * bvec[169] + 2 * bvec[172] - 2 * bvec[173] + 1 * bvec[208] + 1 * bvec[209] + 1 * bvec[212] + 1 * bvec[213] + 1 * bvec[216] + 1 * bvec[217] + 1 * bvec[220] + 1 * bvec[221];
       alphavec[244] = +4 * bvec[32] - 4 * bvec[36] - 4 * bvec[40] + 4 * bvec[44] + 2 * bvec[128] + 2 * bvec[132] - 2 * bvec[136] - 2 * bvec[140] + 2 * bvec[144] - 2 * bvec[148] + 2 * bvec[152] - 2 * bvec[156] + 1 * bvec[224] + 1 * bvec[228] + 1 * bvec[232] + 1 * bvec[236];
       alphavec[245] = +4 * bvec[80] - 4 * bvec[84] - 4 * bvec[88] + 4 * bvec[92] + 2 * bvec[176] + 2 * bvec[180] - 2 * bvec[184] - 2 * bvec[188] + 2 * bvec[192] - 2 * bvec[196] + 2 * bvec[200] - 2 * bvec[204] + 1 * bvec[240] + 1 * bvec[244] + 1 * bvec[248] + 1 * bvec[252];
       alphavec[246] = -12 * bvec[32] + 12 * bvec[33] + 12 * bvec[36] - 12 * bvec[37] + 12 * bvec[40] - 12 * bvec[41] - 12 * bvec[44] + 12 * bvec[45] - 8 * bvec[80] - 4 * bvec[81] + 8 * bvec[84] + 4 * bvec[85] + 8 * bvec[88] + 4 * bvec[89] - 8 * bvec[92] - 4 * bvec[93] - 6 * bvec[128] + 6 * bvec[129] - 6 * bvec[132] + 6 * bvec[133] + 6 * bvec[136] - 6 * bvec[137] + 6 * bvec[140] - 6 * bvec[141] - 6 * bvec[144] + 6 * bvec[145] + 6 * bvec[148] - 6 * bvec[149] - 6 * bvec[152] + 6 * bvec[153] + 6 * bvec[156] - 6 * bvec[157] - 4 * bvec[176] - 2 * bvec[177] - 4 * bvec[180] - 2 * bvec[181] + 4 * bvec[184] + 2 * bvec[185] + 4 * bvec[188] + 2 * bvec[189] - 4 * bvec[192] - 2 * bvec[193] + 4 * bvec[196] + 2 * bvec[197] - 4 * bvec[200] - 2 * bvec[201] + 4 * bvec[204] + 2 * bvec[205] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[228] + 3 * bvec[229] - 3 * bvec[232] + 3 * bvec[233] - 3 * bvec[236] + 3 * bvec[237] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[244] - 1 * bvec[245] - 2 * bvec[248] - 1 * bvec[249] - 2 * bvec[252] - 1 * bvec[253];
       alphavec[247] = +8 * bvec[32] - 8 * bvec[33] - 8 * bvec[36] + 8 * bvec[37] - 8 * bvec[40] + 8 * bvec[41] + 8 * bvec[44] - 8 * bvec[45] + 4 * bvec[80] + 4 * bvec[81] - 4 * bvec[84] - 4 * bvec[85] - 4 * bvec[88] - 4 * bvec[89] + 4 * bvec[92] + 4 * bvec[93] + 4 * bvec[128] - 4 * bvec[129] + 4 * bvec[132] - 4 * bvec[133] - 4 * bvec[136] + 4 * bvec[137] - 4 * bvec[140] + 4 * bvec[141] + 4 * bvec[144] - 4 * bvec[145] - 4 * bvec[148] + 4 * bvec[149] + 4 * bvec[152] - 4 * bvec[153] - 4 * bvec[156] + 4 * bvec[157] + 2 * bvec[176] + 2 * bvec[177] + 2 * bvec[180] + 2 * bvec[181] - 2 * bvec[184] - 2 * bvec[185] - 2 * bvec[188] - 2 * bvec[189] + 2 * bvec[192] + 2 * bvec[193] - 2 * bvec[196] - 2 * bvec[197] + 2 * bvec[200] + 2 * bvec[201] - 2 * bvec[204] - 2 * bvec[205] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[228] - 2 * bvec[229] + 2 * bvec[232] - 2 * bvec[233] + 2 * bvec[236] - 2 * bvec[237] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[244] + 1 * bvec[245] + 1 * bvec[248] + 1 * bvec[249] + 1 * bvec[252] + 1 * bvec[253];
       alphavec[248] = -12 * bvec[0] + 12 * bvec[2] + 12 * bvec[4] - 12 * bvec[6] + 12 * bvec[8] - 12 * bvec[10] - 12 * bvec[12] + 12 * bvec[14] - 8 * bvec[32] - 4 * bvec[34] + 8 * bvec[36] + 4 * bvec[38] + 8 * bvec[40] + 4 * bvec[42] - 8 * bvec[44] - 4 * bvec[46] - 6 * bvec[48] + 6 * bvec[50] - 6 * bvec[52] + 6 * bvec[54] + 6 * bvec[56] - 6 * bvec[58] + 6 * bvec[60] - 6 * bvec[62] - 6 * bvec[64] + 6 * bvec[66] + 6 * bvec[68] - 6 * bvec[70] - 6 * bvec[72] + 6 * bvec[74] + 6 * bvec[76] - 6 * bvec[78] - 4 * bvec[128] - 2 * bvec[130] - 4 * bvec[132] - 2 * bvec[134] + 4 * bvec[136] + 2 * bvec[138] + 4 * bvec[140] + 2 * bvec[142] - 4 * bvec[144] - 2 * bvec[146] + 4 * bvec[148] + 2 * bvec[150] - 4 * bvec[152] - 2 * bvec[154] + 4 * bvec[156] + 2 * bvec[158] - 3 * bvec[160] + 3 * bvec[162] - 3 * bvec[164] + 3 * bvec[166] - 3 * bvec[168] + 3 * bvec[170] - 3 * bvec[172] + 3 * bvec[174] - 2 * bvec[224] - 1 * bvec[226] - 2 * bvec[228] - 1 * bvec[230] - 2 * bvec[232] - 1 * bvec[234] - 2 * bvec[236] - 1 * bvec[238];
       alphavec[249] = -12 * bvec[16] + 12 * bvec[18] + 12 * bvec[20] - 12 * bvec[22] + 12 * bvec[24] - 12 * bvec[26] - 12 * bvec[28] + 12 * bvec[30] - 8 * bvec[80] - 4 * bvec[82] + 8 * bvec[84] + 4 * bvec[86] + 8 * bvec[88] + 4 * bvec[90] - 8 * bvec[92] - 4 * bvec[94] - 6 * bvec[96] + 6 * bvec[98] - 6 * bvec[100] + 6 * bvec[102] + 6 * bvec[104] - 6 * bvec[106] + 6 * bvec[108] - 6 * bvec[110] - 6 * bvec[112] + 6 * bvec[114] + 6 * bvec[116] - 6 * bvec[118] - 6 * bvec[120] + 6 * bvec[122] + 6 * bvec[124] - 6 * bvec[126] - 4 * bvec[176] - 2 * bvec[178] - 4 * bvec[180] - 2 * bvec[182] + 4 * bvec[184] + 2 * bvec[186] + 4 * bvec[188] + 2 * bvec[190] - 4 * bvec[192] - 2 * bvec[194] + 4 * bvec[196] + 2 * bvec[198] - 4 * bvec[200] - 2 * bvec[202] + 4 * bvec[204] + 2 * bvec[206] - 3 * bvec[208] + 3 * bvec[210] - 3 * bvec[212] + 3 * bvec[214] - 3 * bvec[216] + 3 * bvec[218] - 3 * bvec[220] + 3 * bvec[222] - 2 * bvec[240] - 1 * bvec[242] - 2 * bvec[244] - 1 * bvec[246] - 2 * bvec[248] - 1 * bvec[250] - 2 * bvec[252] - 1 * bvec[254];
       alphavec[250] = +36 * bvec[0] - 36 * bvec[1] - 36 * bvec[2] + 36 * bvec[3] - 36 * bvec[4] + 36 * bvec[5] + 36 * bvec[6] - 36 * bvec[7] - 36 * bvec[8] + 36 * bvec[9] + 36 * bvec[10] - 36 * bvec[11] + 36 * bvec[12] - 36 * bvec[13] - 36 * bvec[14] + 36 * bvec[15] + 24 * bvec[16] + 12 * bvec[17] - 24 * bvec[18] - 12 * bvec[19] - 24 * bvec[20] - 12 * bvec[21] + 24 * bvec[22] + 12 * bvec[23] - 24 * bvec[24] - 12 * bvec[25] + 24 * bvec[26] + 12 * bvec[27] + 24 * bvec[28] + 12 * bvec[29] - 24 * bvec[30] - 12 * bvec[31] + 24 * bvec[32] - 24 * bvec[33] + 12 * bvec[34] - 12 * bvec[35] - 24 * bvec[36] + 24 * bvec[37] - 12 * bvec[38] + 12 * bvec[39] - 24 * bvec[40] + 24 * bvec[41] - 12 * bvec[42] + 12 * bvec[43] + 24 * bvec[44] - 24 * bvec[45] + 12 * bvec[46] - 12 * bvec[47] + 18 * bvec[48] - 18 * bvec[49] - 18 * bvec[50] + 18 * bvec[51] + 18 * bvec[52] - 18 * bvec[53] - 18 * bvec[54] + 18 * bvec[55] - 18 * bvec[56] + 18 * bvec[57] + 18 * bvec[58] - 18 * bvec[59] - 18 * bvec[60] + 18 * bvec[61] + 18 * bvec[62] - 18 * bvec[63] + 18 * bvec[64] - 18 * bvec[65] - 18 * bvec[66] + 18 * bvec[67] - 18 * bvec[68] + 18 * bvec[69] + 18 * bvec[70] - 18 * bvec[71] + 18 * bvec[72] - 18 * bvec[73] - 18 * bvec[74] + 18 * bvec[75] - 18 * bvec[76] + 18 * bvec[77] + 18 * bvec[78] - 18 * bvec[79] + 16 * bvec[80] + 8 * bvec[81] + 8 * bvec[82] + 4 * bvec[83] - 16 * bvec[84] - 8 * bvec[85] - 8 * bvec[86] - 4 * bvec[87] - 16 * bvec[88] - 8 * bvec[89] - 8 * bvec[90] - 4 * bvec[91] + 16 * bvec[92] + 8 * bvec[93] + 8 * bvec[94] + 4 * bvec[95] + 12 * bvec[96] + 6 * bvec[97] - 12 * bvec[98] - 6 * bvec[99] + 12 * bvec[100] + 6 * bvec[101] - 12 * bvec[102] - 6 * bvec[103] - 12 * bvec[104] - 6 * bvec[105] + 12 * bvec[106] + 6 * bvec[107] - 12 * bvec[108] - 6 * bvec[109] + 12 * bvec[110] + 6 * bvec[111] + 12 * bvec[112] + 6 * bvec[113] - 12 * bvec[114] - 6 * bvec[115] - 12 * bvec[116] - 6 * bvec[117] + 12 * bvec[118] + 6 * bvec[119] + 12 * bvec[120] + 6 * bvec[121] - 12 * bvec[122] - 6 * bvec[123] - 12 * bvec[124] - 6 * bvec[125] + 12 * bvec[126] + 6 * bvec[127] + 12 * bvec[128] - 12 * bvec[129] + 6 * bvec[130] - 6 * bvec[131] + 12 * bvec[132] - 12 * bvec[133] + 6 * bvec[134] - 6 * bvec[135] - 12 * bvec[136] + 12 * bvec[137] - 6 * bvec[138] + 6 * bvec[139] - 12 * bvec[140] + 12 * bvec[141] - 6 * bvec[142] + 6 * bvec[143] + 12 * bvec[144] - 12 * bvec[145] + 6 * bvec[146] - 6 * bvec[147] - 12 * bvec[148] + 12 * bvec[149] - 6 * bvec[150] + 6 * bvec[151] + 12 * bvec[152] - 12 * bvec[153] + 6 * bvec[154] - 6 * bvec[155] - 12 * bvec[156] + 12 * bvec[157] - 6 * bvec[158] + 6 * bvec[159] + 9 * bvec[160] - 9 * bvec[161] - 9 * bvec[162] + 9 * bvec[163] + 9 * bvec[164] - 9 * bvec[165] - 9 * bvec[166] + 9 * bvec[167] + 9 * bvec[168] - 9 * bvec[169] - 9 * bvec[170] + 9 * bvec[171] + 9 * bvec[172] - 9 * bvec[173] - 9 * bvec[174] + 9 * bvec[175] + 8 * bvec[176] + 4 * bvec[177] + 4 * bvec[178] + 2 * bvec[179] + 8 * bvec[180] + 4 * bvec[181] + 4 * bvec[182] + 2 * bvec[183] - 8 * bvec[184] - 4 * bvec[185] - 4 * bvec[186] - 2 * bvec[187] - 8 * bvec[188] - 4 * bvec[189] - 4 * bvec[190] - 2 * bvec[191] + 8 * bvec[192] + 4 * bvec[193] + 4 * bvec[194] + 2 * bvec[195] - 8 * bvec[196] - 4 * bvec[197] - 4 * bvec[198] - 2 * bvec[199] + 8 * bvec[200] + 4 * bvec[201] + 4 * bvec[202] + 2 * bvec[203] - 8 * bvec[204] - 4 * bvec[205] - 4 * bvec[206] - 2 * bvec[207] + 6 * bvec[208] + 3 * bvec[209] - 6 * bvec[210] - 3 * bvec[211] + 6 * bvec[212] + 3 * bvec[213] - 6 * bvec[214] - 3 * bvec[215] + 6 * bvec[216] + 3 * bvec[217] - 6 * bvec[218] - 3 * bvec[219] + 6 * bvec[220] + 3 * bvec[221] - 6 * bvec[222] - 3 * bvec[223] + 6 * bvec[224] - 6 * bvec[225] + 3 * bvec[226] - 3 * bvec[227] + 6 * bvec[228] - 6 * bvec[229] + 3 * bvec[230] - 3 * bvec[231] + 6 * bvec[232] - 6 * bvec[233] + 3 * bvec[234] - 3 * bvec[235] + 6 * bvec[236] - 6 * bvec[237] + 3 * bvec[238] - 3 * bvec[239] + 4 * bvec[240] + 2 * bvec[241] + 2 * bvec[242] + 1 * bvec[243] + 4 * bvec[244] + 2 * bvec[245] + 2 * bvec[246] + 1 * bvec[247] + 4 * bvec[248] + 2 * bvec[249] + 2 * bvec[250] + 1 * bvec[251] + 4 * bvec[252] + 2 * bvec[253] + 2 * bvec[254] + 1 * bvec[255];
       alphavec[251] = -24 * bvec[0] + 24 * bvec[1] + 24 * bvec[2] - 24 * bvec[3] + 24 * bvec[4] - 24 * bvec[5] - 24 * bvec[6] + 24 * bvec[7] + 24 * bvec[8] - 24 * bvec[9] - 24 * bvec[10] + 24 * bvec[11] - 24 * bvec[12] + 24 * bvec[13] + 24 * bvec[14] - 24 * bvec[15] - 12 * bvec[16] - 12 * bvec[17] + 12 * bvec[18] + 12 * bvec[19] + 12 * bvec[20] + 12 * bvec[21] - 12 * bvec[22] - 12 * bvec[23] + 12 * bvec[24] + 12 * bvec[25] - 12 * bvec[26] - 12 * bvec[27] - 12 * bvec[28] - 12 * bvec[29] + 12 * bvec[30] + 12 * bvec[31] - 16 * bvec[32] + 16 * bvec[33] - 8 * bvec[34] + 8 * bvec[35] + 16 * bvec[36] - 16 * bvec[37] + 8 * bvec[38] - 8 * bvec[39] + 16 * bvec[40] - 16 * bvec[41] + 8 * bvec[42] - 8 * bvec[43] - 16 * bvec[44] + 16 * bvec[45] - 8 * bvec[46] + 8 * bvec[47] - 12 * bvec[48] + 12 * bvec[49] + 12 * bvec[50] - 12 * bvec[51] - 12 * bvec[52] + 12 * bvec[53] + 12 * bvec[54] - 12 * bvec[55] + 12 * bvec[56] - 12 * bvec[57] - 12 * bvec[58] + 12 * bvec[59] + 12 * bvec[60] - 12 * bvec[61] - 12 * bvec[62] + 12 * bvec[63] - 12 * bvec[64] + 12 * bvec[65] + 12 * bvec[66] - 12 * bvec[67] + 12 * bvec[68] - 12 * bvec[69] - 12 * bvec[70] + 12 * bvec[71] - 12 * bvec[72] + 12 * bvec[73] + 12 * bvec[74] - 12 * bvec[75] + 12 * bvec[76] - 12 * bvec[77] - 12 * bvec[78] + 12 * bvec[79] - 8 * bvec[80] - 8 * bvec[81] - 4 * bvec[82] - 4 * bvec[83] + 8 * bvec[84] + 8 * bvec[85] + 4 * bvec[86] + 4 * bvec[87] + 8 * bvec[88] + 8 * bvec[89] + 4 * bvec[90] + 4 * bvec[91] - 8 * bvec[92] - 8 * bvec[93] - 4 * bvec[94] - 4 * bvec[95] - 6 * bvec[96] - 6 * bvec[97] + 6 * bvec[98] + 6 * bvec[99] - 6 * bvec[100] - 6 * bvec[101] + 6 * bvec[102] + 6 * bvec[103] + 6 * bvec[104] + 6 * bvec[105] - 6 * bvec[106] - 6 * bvec[107] + 6 * bvec[108] + 6 * bvec[109] - 6 * bvec[110] - 6 * bvec[111] - 6 * bvec[112] - 6 * bvec[113] + 6 * bvec[114] + 6 * bvec[115] + 6 * bvec[116] + 6 * bvec[117] - 6 * bvec[118] - 6 * bvec[119] - 6 * bvec[120] - 6 * bvec[121] + 6 * bvec[122] + 6 * bvec[123] + 6 * bvec[124] + 6 * bvec[125] - 6 * bvec[126] - 6 * bvec[127] - 8 * bvec[128] + 8 * bvec[129] - 4 * bvec[130] + 4 * bvec[131] - 8 * bvec[132] + 8 * bvec[133] - 4 * bvec[134] + 4 * bvec[135] + 8 * bvec[136] - 8 * bvec[137] + 4 * bvec[138] - 4 * bvec[139] + 8 * bvec[140] - 8 * bvec[141] + 4 * bvec[142] - 4 * bvec[143] - 8 * bvec[144] + 8 * bvec[145] - 4 * bvec[146] + 4 * bvec[147] + 8 * bvec[148] - 8 * bvec[149] + 4 * bvec[150] - 4 * bvec[151] - 8 * bvec[152] + 8 * bvec[153] - 4 * bvec[154] + 4 * bvec[155] + 8 * bvec[156] - 8 * bvec[157] + 4 * bvec[158] - 4 * bvec[159] - 6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 6 * bvec[164] + 6 * bvec[165] + 6 * bvec[166] - 6 * bvec[167] - 6 * bvec[168] + 6 * bvec[169] + 6 * bvec[170] - 6 * bvec[171] - 6 * bvec[172] + 6 * bvec[173] + 6 * bvec[174] - 6 * bvec[175] - 4 * bvec[176] - 4 * bvec[177] - 2 * bvec[178] - 2 * bvec[179] - 4 * bvec[180] - 4 * bvec[181] - 2 * bvec[182] - 2 * bvec[183] + 4 * bvec[184] + 4 * bvec[185] + 2 * bvec[186] + 2 * bvec[187] + 4 * bvec[188] + 4 * bvec[189] + 2 * bvec[190] + 2 * bvec[191] - 4 * bvec[192] - 4 * bvec[193] - 2 * bvec[194] - 2 * bvec[195] + 4 * bvec[196] + 4 * bvec[197] + 2 * bvec[198] + 2 * bvec[199] - 4 * bvec[200] - 4 * bvec[201] - 2 * bvec[202] - 2 * bvec[203] + 4 * bvec[204] + 4 * bvec[205] + 2 * bvec[206] + 2 * bvec[207] - 3 * bvec[208] - 3 * bvec[209] + 3 * bvec[210] + 3 * bvec[211] - 3 * bvec[212] - 3 * bvec[213] + 3 * bvec[214] + 3 * bvec[215] - 3 * bvec[216] - 3 * bvec[217] + 3 * bvec[218] + 3 * bvec[219] - 3 * bvec[220] - 3 * bvec[221] + 3 * bvec[222] + 3 * bvec[223] - 4 * bvec[224] + 4 * bvec[225] - 2 * bvec[226] + 2 * bvec[227] - 4 * bvec[228] + 4 * bvec[229] - 2 * bvec[230] + 2 * bvec[231] - 4 * bvec[232] + 4 * bvec[233] - 2 * bvec[234] + 2 * bvec[235] - 4 * bvec[236] + 4 * bvec[237] - 2 * bvec[238] + 2 * bvec[239] - 2 * bvec[240] - 2 * bvec[241] - 1 * bvec[242] - 1 * bvec[243] - 2 * bvec[244] - 2 * bvec[245] - 1 * bvec[246] - 1 * bvec[247] - 2 * bvec[248] - 2 * bvec[249] - 1 * bvec[250] - 1 * bvec[251] - 2 * bvec[252] - 2 * bvec[253] - 1 * bvec[254] - 1 * bvec[255];
       alphavec[252] = +8 * bvec[0] - 8 * bvec[2] - 8 * bvec[4] + 8 * bvec[6] - 8 * bvec[8] + 8 * bvec[10] + 8 * bvec[12] - 8 * bvec[14] + 4 * bvec[32] + 4 * bvec[34] - 4 * bvec[36] - 4 * bvec[38] - 4 * bvec[40] - 4 * bvec[42] + 4 * bvec[44] + 4 * bvec[46] + 4 * bvec[48] - 4 * bvec[50] + 4 * bvec[52] - 4 * bvec[54] - 4 * bvec[56] + 4 * bvec[58] - 4 * bvec[60] + 4 * bvec[62] + 4 * bvec[64] - 4 * bvec[66] - 4 * bvec[68] + 4 * bvec[70] + 4 * bvec[72] - 4 * bvec[74] - 4 * bvec[76] + 4 * bvec[78] + 2 * bvec[128] + 2 * bvec[130] + 2 * bvec[132] + 2 * bvec[134] - 2 * bvec[136] - 2 * bvec[138] - 2 * bvec[140] - 2 * bvec[142] + 2 * bvec[144] + 2 * bvec[146] - 2 * bvec[148] - 2 * bvec[150] + 2 * bvec[152] + 2 * bvec[154] - 2 * bvec[156] - 2 * bvec[158] + 2 * bvec[160] - 2 * bvec[162] + 2 * bvec[164] - 2 * bvec[166] + 2 * bvec[168] - 2 * bvec[170] + 2 * bvec[172] - 2 * bvec[174] + 1 * bvec[224] + 1 * bvec[226] + 1 * bvec[228] + 1 * bvec[230] + 1 * bvec[232] + 1 * bvec[234] + 1 * bvec[236] + 1 * bvec[238];
       alphavec[253] = +8 * bvec[16] - 8 * bvec[18] - 8 * bvec[20] + 8 * bvec[22] - 8 * bvec[24] + 8 * bvec[26] + 8 * bvec[28] - 8 * bvec[30] + 4 * bvec[80] + 4 * bvec[82] - 4 * bvec[84] - 4 * bvec[86] - 4 * bvec[88] - 4 * bvec[90] + 4 * bvec[92] + 4 * bvec[94] + 4 * bvec[96] - 4 * bvec[98] + 4 * bvec[100] - 4 * bvec[102] - 4 * bvec[104] + 4 * bvec[106] - 4 * bvec[108] + 4 * bvec[110] + 4 * bvec[112] - 4 * bvec[114] - 4 * bvec[116] + 4 * bvec[118] + 4 * bvec[120] - 4 * bvec[122] - 4 * bvec[124] + 4 * bvec[126] + 2 * bvec[176] + 2 * bvec[178] + 2 * bvec[180] + 2 * bvec[182] - 2 * bvec[184] - 2 * bvec[186] - 2 * bvec[188] - 2 * bvec[190] + 2 * bvec[192] + 2 * bvec[194] - 2 * bvec[196] - 2 * bvec[198] + 2 * bvec[200] + 2 * bvec[202] - 2 * bvec[204] - 2 * bvec[206] + 2 * bvec[208] - 2 * bvec[210] + 2 * bvec[212] - 2 * bvec[214] + 2 * bvec[216] - 2 * bvec[218] + 2 * bvec[220] - 2 * bvec[222] + 1 * bvec[240] + 1 * bvec[242] + 1 * bvec[244] + 1 * bvec[246] + 1 * bvec[248] + 1 * bvec[250] + 1 * bvec[252] + 1 * bvec[254];
       alphavec[254] = -24 * bvec[0] + 24 * bvec[1] + 24 * bvec[2] - 24 * bvec[3] + 24 * bvec[4] - 24 * bvec[5] - 24 * bvec[6] + 24 * bvec[7] + 24 * bvec[8] - 24 * bvec[9] - 24 * bvec[10] + 24 * bvec[11] - 24 * bvec[12] + 24 * bvec[13] + 24 * bvec[14] - 24 * bvec[15] - 16 * bvec[16] - 8 * bvec[17] + 16 * bvec[18] + 8 * bvec[19] + 16 * bvec[20] + 8 * bvec[21] - 16 * bvec[22] - 8 * bvec[23] + 16 * bvec[24] + 8 * bvec[25] - 16 * bvec[26] - 8 * bvec[27] - 16 * bvec[28] - 8 * bvec[29] + 16 * bvec[30] + 8 * bvec[31] - 12 * bvec[32] + 12 * bvec[33] - 12 * bvec[34] + 12 * bvec[35] + 12 * bvec[36] - 12 * bvec[37] + 12 * bvec[38] - 12 * bvec[39] + 12 * bvec[40] - 12 * bvec[41] + 12 * bvec[42] - 12 * bvec[43] - 12 * bvec[44] + 12 * bvec[45] - 12 * bvec[46] + 12 * bvec[47] - 12 * bvec[48] + 12 * bvec[49] + 12 * bvec[50] - 12 * bvec[51] - 12 * bvec[52] + 12 * bvec[53] + 12 * bvec[54] - 12 * bvec[55] + 12 * bvec[56] - 12 * bvec[57] - 12 * bvec[58] + 12 * bvec[59] + 12 * bvec[60] - 12 * bvec[61] - 12 * bvec[62] + 12 * bvec[63] - 12 * bvec[64] + 12 * bvec[65] + 12 * bvec[66] - 12 * bvec[67] + 12 * bvec[68] - 12 * bvec[69] - 12 * bvec[70] + 12 * bvec[71] - 12 * bvec[72] + 12 * bvec[73] + 12 * bvec[74] - 12 * bvec[75] + 12 * bvec[76] - 12 * bvec[77] - 12 * bvec[78] + 12 * bvec[79] - 8 * bvec[80] - 4 * bvec[81] - 8 * bvec[82] - 4 * bvec[83] + 8 * bvec[84] + 4 * bvec[85] + 8 * bvec[86] + 4 * bvec[87] + 8 * bvec[88] + 4 * bvec[89] + 8 * bvec[90] + 4 * bvec[91] - 8 * bvec[92] - 4 * bvec[93] - 8 * bvec[94] - 4 * bvec[95] - 8 * bvec[96] - 4 * bvec[97] + 8 * bvec[98] + 4 * bvec[99] - 8 * bvec[100] - 4 * bvec[101] + 8 * bvec[102] + 4 * bvec[103] + 8 * bvec[104] + 4 * bvec[105] - 8 * bvec[106] - 4 * bvec[107] + 8 * bvec[108] + 4 * bvec[109] - 8 * bvec[110] - 4 * bvec[111] - 8 * bvec[112] - 4 * bvec[113] + 8 * bvec[114] + 4 * bvec[115] + 8 * bvec[116] + 4 * bvec[117] - 8 * bvec[118] - 4 * bvec[119] - 8 * bvec[120] - 4 * bvec[121] + 8 * bvec[122] + 4 * bvec[123] + 8 * bvec[124] + 4 * bvec[125] - 8 * bvec[126] - 4 * bvec[127] - 6 * bvec[128] + 6 * bvec[129] - 6 * bvec[130] + 6 * bvec[131] - 6 * bvec[132] + 6 * bvec[133] - 6 * bvec[134] + 6 * bvec[135] + 6 * bvec[136] - 6 * bvec[137] + 6 * bvec[138] - 6 * bvec[139] + 6 * bvec[140] - 6 * bvec[141] + 6 * bvec[142] - 6 * bvec[143] - 6 * bvec[144] + 6 * bvec[145] - 6 * bvec[146] + 6 * bvec[147] + 6 * bvec[148] - 6 * bvec[149] + 6 * bvec[150] - 6 * bvec[151] - 6 * bvec[152] + 6 * bvec[153] - 6 * bvec[154] + 6 * bvec[155] + 6 * bvec[156] - 6 * bvec[157] + 6 * bvec[158] - 6 * bvec[159] - 6 * bvec[160] + 6 * bvec[161] + 6 * bvec[162] - 6 * bvec[163] - 6 * bvec[164] + 6 * bvec[165] + 6 * bvec[166] - 6 * bvec[167] - 6 * bvec[168] + 6 * bvec[169] + 6 * bvec[170] - 6 * bvec[171] - 6 * bvec[172] + 6 * bvec[173] + 6 * bvec[174] - 6 * bvec[175] - 4 * bvec[176] - 2 * bvec[177] - 4 * bvec[178] - 2 * bvec[179] - 4 * bvec[180] - 2 * bvec[181] - 4 * bvec[182] - 2 * bvec[183] + 4 * bvec[184] + 2 * bvec[185] + 4 * bvec[186] + 2 * bvec[187] + 4 * bvec[188] + 2 * bvec[189] + 4 * bvec[190] + 2 * bvec[191] - 4 * bvec[192] - 2 * bvec[193] - 4 * bvec[194] - 2 * bvec[195] + 4 * bvec[196] + 2 * bvec[197] + 4 * bvec[198] + 2 * bvec[199] - 4 * bvec[200] - 2 * bvec[201] - 4 * bvec[202] - 2 * bvec[203] + 4 * bvec[204] + 2 * bvec[205] + 4 * bvec[206] + 2 * bvec[207] - 4 * bvec[208] - 2 * bvec[209] + 4 * bvec[210] + 2 * bvec[211] - 4 * bvec[212] - 2 * bvec[213] + 4 * bvec[214] + 2 * bvec[215] - 4 * bvec[216] - 2 * bvec[217] + 4 * bvec[218] + 2 * bvec[219] - 4 * bvec[220] - 2 * bvec[221] + 4 * bvec[222] + 2 * bvec[223] - 3 * bvec[224] + 3 * bvec[225] - 3 * bvec[226] + 3 * bvec[227] - 3 * bvec[228] + 3 * bvec[229] - 3 * bvec[230] + 3 * bvec[231] - 3 * bvec[232] + 3 * bvec[233] - 3 * bvec[234] + 3 * bvec[235] - 3 * bvec[236] + 3 * bvec[237] - 3 * bvec[238] + 3 * bvec[239] - 2 * bvec[240] - 1 * bvec[241] - 2 * bvec[242] - 1 * bvec[243] - 2 * bvec[244] - 1 * bvec[245] - 2 * bvec[246] - 1 * bvec[247] - 2 * bvec[248] - 1 * bvec[249] - 2 * bvec[250] - 1 * bvec[251] - 2 * bvec[252] - 1 * bvec[253] - 2 * bvec[254] - 1 * bvec[255];
       alphavec[255] = +16 * bvec[0] - 16 * bvec[1] - 16 * bvec[2] + 16 * bvec[3] - 16 * bvec[4] + 16 * bvec[5] + 16 * bvec[6] - 16 * bvec[7] - 16 * bvec[8] + 16 * bvec[9] + 16 * bvec[10] - 16 * bvec[11] + 16 * bvec[12] - 16 * bvec[13] - 16 * bvec[14] + 16 * bvec[15] + 8 * bvec[16] + 8 * bvec[17] - 8 * bvec[18] - 8 * bvec[19] - 8 * bvec[20] - 8 * bvec[21] + 8 * bvec[22] + 8 * bvec[23] - 8 * bvec[24] - 8 * bvec[25] + 8 * bvec[26] + 8 * bvec[27] + 8 * bvec[28] + 8 * bvec[29] - 8 * bvec[30] - 8 * bvec[31] + 8 * bvec[32] - 8 * bvec[33] + 8 * bvec[34] - 8 * bvec[35] - 8 * bvec[36] + 8 * bvec[37] - 8 * bvec[38] + 8 * bvec[39] - 8 * bvec[40] + 8 * bvec[41] - 8 * bvec[42] + 8 * bvec[43] + 8 * bvec[44] - 8 * bvec[45] + 8 * bvec[46] - 8 * bvec[47] + 8 * bvec[48] - 8 * bvec[49] - 8 * bvec[50] + 8 * bvec[51] + 8 * bvec[52] - 8 * bvec[53] - 8 * bvec[54] + 8 * bvec[55] - 8 * bvec[56] + 8 * bvec[57] + 8 * bvec[58] - 8 * bvec[59] - 8 * bvec[60] + 8 * bvec[61] + 8 * bvec[62] - 8 * bvec[63] + 8 * bvec[64] - 8 * bvec[65] - 8 * bvec[66] + 8 * bvec[67] - 8 * bvec[68] + 8 * bvec[69] + 8 * bvec[70] - 8 * bvec[71] + 8 * bvec[72] - 8 * bvec[73] - 8 * bvec[74] + 8 * bvec[75] - 8 * bvec[76] + 8 * bvec[77] + 8 * bvec[78] - 8 * bvec[79] + 4 * bvec[80] + 4 * bvec[81] + 4 * bvec[82] + 4 * bvec[83] - 4 * bvec[84] - 4 * bvec[85] - 4 * bvec[86] - 4 * bvec[87] - 4 * bvec[88] - 4 * bvec[89] - 4 * bvec[90] - 4 * bvec[91] + 4 * bvec[92] + 4 * bvec[93] + 4 * bvec[94] + 4 * bvec[95] + 4 * bvec[96] + 4 * bvec[97] - 4 * bvec[98] - 4 * bvec[99] + 4 * bvec[100] + 4 * bvec[101] - 4 * bvec[102] - 4 * bvec[103] - 4 * bvec[104] - 4 * bvec[105] + 4 * bvec[106] + 4 * bvec[107] - 4 * bvec[108] - 4 * bvec[109] + 4 * bvec[110] + 4 * bvec[111] + 4 * bvec[112] + 4 * bvec[113] - 4 * bvec[114] - 4 * bvec[115] - 4 * bvec[116] - 4 * bvec[117] + 4 * bvec[118] + 4 * bvec[119] + 4 * bvec[120] + 4 * bvec[121] - 4 * bvec[122] - 4 * bvec[123] - 4 * bvec[124] - 4 * bvec[125] + 4 * bvec[126] + 4 * bvec[127] + 4 * bvec[128] - 4 * bvec[129] + 4 * bvec[130] - 4 * bvec[131] + 4 * bvec[132] - 4 * bvec[133] + 4 * bvec[134] - 4 * bvec[135] - 4 * bvec[136] + 4 * bvec[137] - 4 * bvec[138] + 4 * bvec[139] - 4 * bvec[140] + 4 * bvec[141] - 4 * bvec[142] + 4 * bvec[143] + 4 * bvec[144] - 4 * bvec[145] + 4 * bvec[146] - 4 * bvec[147] - 4 * bvec[148] + 4 * bvec[149] - 4 * bvec[150] + 4 * bvec[151] + 4 * bvec[152] - 4 * bvec[153] + 4 * bvec[154] - 4 * bvec[155] - 4 * bvec[156] + 4 * bvec[157] - 4 * bvec[158] + 4 * bvec[159] + 4 * bvec[160] - 4 * bvec[161] - 4 * bvec[162] + 4 * bvec[163] + 4 * bvec[164] - 4 * bvec[165] - 4 * bvec[166] + 4 * bvec[167] + 4 * bvec[168] - 4 * bvec[169] - 4 * bvec[170] + 4 * bvec[171] + 4 * bvec[172] - 4 * bvec[173] - 4 * bvec[174] + 4 * bvec[175] + 2 * bvec[176] + 2 * bvec[177] + 2 * bvec[178] + 2 * bvec[179] + 2 * bvec[180] + 2 * bvec[181] + 2 * bvec[182] + 2 * bvec[183] - 2 * bvec[184] - 2 * bvec[185] - 2 * bvec[186] - 2 * bvec[187] - 2 * bvec[188] - 2 * bvec[189] - 2 * bvec[190] - 2 * bvec[191] + 2 * bvec[192] + 2 * bvec[193] + 2 * bvec[194] + 2 * bvec[195] - 2 * bvec[196] - 2 * bvec[197] - 2 * bvec[198] - 2 * bvec[199] + 2 * bvec[200] + 2 * bvec[201] + 2 * bvec[202] + 2 * bvec[203] - 2 * bvec[204] - 2 * bvec[205] - 2 * bvec[206] - 2 * bvec[207] + 2 * bvec[208] + 2 * bvec[209] - 2 * bvec[210] - 2 * bvec[211] + 2 * bvec[212] + 2 * bvec[213] - 2 * bvec[214] - 2 * bvec[215] + 2 * bvec[216] + 2 * bvec[217] - 2 * bvec[218] - 2 * bvec[219] + 2 * bvec[220] + 2 * bvec[221] - 2 * bvec[222] - 2 * bvec[223] + 2 * bvec[224] - 2 * bvec[225] + 2 * bvec[226] - 2 * bvec[227] + 2 * bvec[228] - 2 * bvec[229] + 2 * bvec[230] - 2 * bvec[231] + 2 * bvec[232] - 2 * bvec[233] + 2 * bvec[234] - 2 * bvec[235] + 2 * bvec[236] - 2 * bvec[237] + 2 * bvec[238] - 2 * bvec[239] + 1 * bvec[240] + 1 * bvec[241] + 1 * bvec[242] + 1 * bvec[243] + 1 * bvec[244] + 1 * bvec[245] + 1 * bvec[246] + 1 * bvec[247] + 1 * bvec[248] + 1 * bvec[249] + 1 * bvec[250] + 1 * bvec[251] + 1 * bvec[252] + 1 * bvec[253] + 1 * bvec[254] + 1 * bvec[255];

       return alphavec;
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
      double wstep = ws[welem + 1] - ws[welem];


      double xf = (x - xs[xelem]) / xstep;
      double yf = (y - ys[yelem]) / ystep;
      double zf = (z - zs[zelem]) / zstep;
      double wf = (w - ws[welem]) / wstep;


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

            dfxyzw.setZero();

            Eigen::DiagonalMatrix<double, 4> dmat(1.0 / xstep, 1.0 / ystep, 1.0 / zstep, 1.0 / wstep);


            for (int i = 0, start = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++, start += 4) {

                        double xterm = alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2]
                            + xf3 * alphavec[start + 3];
                        double dxterm =
                            alphavec[start + 1] + 2 * xf * alphavec[start + 2] + 3 * xf2 * alphavec[start + 3];

                        dfxyzw[0] += (yfs[k] * zfs[j] * wfs[i]) * dxterm;

                        if (k > 0) {
                            dfxyzw[1] += (k*yfs[k-1] * zfs[j] * wfs[i]) * xterm;
                        }
                        if (j > 0) {
                            dfxyzw[2] += (yfs[k] * j*zfs[j-1] * wfs[i]) * xterm;
                        }
                        if (i > 0) {
                            dfxyzw[3] += (yfs[k] * zfs[j] * i* wfs[i-1]) * xterm;
                        }

                    }
                }
            }
            dfxyzw = (dmat*dfxyzw).eval();//QED

          if (deriv > 1) {



              d2fxyzw.setZero();

            
              for (int i = 0, start = 0; i < 4; i++) {
                  for (int j = 0; j < 4; j++) {
                      for (int k = 0; k < 4; k++, start += 4) {

                          double xterm = alphavec[start] + xf * alphavec[start + 1] + xf2 * alphavec[start + 2]
                              + xf3 * alphavec[start + 3];
                          double dxterm =
                              alphavec[start + 1] + 2 * xf * alphavec[start + 2] + 3 * xf2 * alphavec[start + 3];

                          // First row of hessian, diffing this term from above
                          //dfxyzw[0] += (yfs[k] * zfs[j] * wfs[i]) * dxterm;
                          ////////////////////////////////////////////////////
                          d2fxyzw(0, 0) += (yfs[k] * zfs[j] * wfs[i]) * (2 * alphavec[start + 2] + 6 * xf * alphavec[start + 3]);

                          if (k > 0) {
                              d2fxyzw(0, 1) += (k * yfs[k - 1] * zfs[j] * wfs[i]) * dxterm;
                          }
                          if (j > 0) {
                              d2fxyzw(0, 2) += (yfs[k] * j * zfs[j - 1] * wfs[i]) * dxterm;
                          }
                          if (i > 0) {
                              d2fxyzw(0, 3) += (yfs[k] * zfs[j] * i * wfs[i - 1]) * dxterm;
                          }
                          ////////////////////////////////////////////////////

                          if (k > 0) {
                              // Second row of hessian, diffing this term from above
                              //dfxyzw[1] += (k * yfs[k - 1] * zfs[j] * wfs[i]) * xterm;
                              if (k > 1) {
                                  d2fxyzw(1, 1) += (k * (k - 1) * yfs[k - 2] * zfs[j] * wfs[i]) * xterm;
                              }
                              if (j > 0) {
                                  d2fxyzw(1, 2) += (k * yfs[k - 1] * j * zfs[j - 1] * wfs[i]) * xterm;
                              }
                              if (i > 0) {
                                  d2fxyzw(1, 3) += (k * yfs[k - 1] * zfs[j] * i * wfs[i - 1]) * xterm;
                              }
                          }
                          if (j > 0) {
                              // Third row of hessian, diffing this term from above
                              //dfxyzw[2] += (yfs[k] * j * zfs[j - 1] * wfs[i]) * xterm;
                              if (j > 1) {
                                  d2fxyzw(2, 2) += (yfs[k] * j *(j-1) * zfs[j - 2] * wfs[i]) * xterm;
                              }
                              if (i > 0) {
                                  d2fxyzw(2, 3) += (yfs[k] * j * zfs[j - 1] * i* wfs[i-1]) * xterm;
                              }
                          }
                          if (i > 0) {
                              // Fourth row of hessian, diffing this term from above
                              //dfxyzw[3] += (yfs[k] * zfs[j] * i * wfs[i - 1]) * xterm;
                              if (i > 1) {
                                  d2fxyzw(3, 3) += (yfs[k] * zfs[j] * i * (i - 1) * wfs[i - 2]) * xterm;
                              }

                          }
                      }
                  }
              }
              // First col
              d2fxyzw(1, 0) = d2fxyzw(0, 1);
              d2fxyzw(2, 0) = d2fxyzw(0, 2);
              d2fxyzw(3, 0) = d2fxyzw(0, 3);
              // Second col
              d2fxyzw(2, 1) = d2fxyzw(1, 2);
              d2fxyzw(3, 1) = d2fxyzw(1, 3);
              //Third col
              d2fxyzw(3, 2) = d2fxyzw(2, 3);

              d2fxyzw = (dmat * d2fxyzw * dmat).eval();//QED

          }
        }

      } else { /// Linear interpolation
        
        fval =0;
        dfxyzw.setZero();
        d2fxyzw.setZero();

        Eigen::DiagonalMatrix<double, 4> dmat(1.0 / xstep, 1.0 / ystep, 1.0 / zstep, 1.0 / wstep);


        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 2; l++) {

                        double fcorner = this->fs(xelem + l, yelem + k, zelem + j, welem + i);

                        double xweight = (l == 0 ? 1 - xf : xf);
                        double yweight = (k == 0 ? 1 - yf : yf);
                        double zweight = (j == 0 ? 1 - zf : zf);
                        double wweight = (i == 0 ? 1 - wf : wf);

                 
                        fval += fcorner * xweight * yweight * zweight * wweight;

                        if (deriv > 0) {
                            double dxweight = (l == 0 ? -1 : 1);
                            double dyweight = (k == 0 ? -1 : 1);
                            double dzweight = (j == 0 ? -1 : 1);
                            double dwweight = (i == 0 ? -1 : 1);

                            dfxyzw[0] += fcorner * dxweight * yweight * zweight * wweight;
                            dfxyzw[1] += fcorner * xweight * dyweight * zweight * wweight;
                            dfxyzw[2] += fcorner * xweight * yweight * dzweight * wweight;
                            dfxyzw[3] += fcorner * xweight * yweight * zweight * dwweight;

                            if (deriv > 1) {
                                d2fxyzw(0, 1) += fcorner * dxweight * dyweight * zweight * wweight;
                                d2fxyzw(0, 2) += fcorner * dxweight * yweight * dzweight * wweight;
                                d2fxyzw(0, 3) += fcorner * dxweight * yweight * zweight * dwweight;

                                d2fxyzw(1, 2) += fcorner * xweight * dyweight * dzweight * wweight;
                                d2fxyzw(1, 3) += fcorner * xweight * dyweight * zweight * dwweight;

                                d2fxyzw(2, 3) += fcorner * xweight * yweight * dzweight * dwweight;
                            }
                        }
                    }
                }
            }
        }


        if (deriv > 0) {
            dfxyzw = (dmat * dfxyzw).eval();//QED

     
          if (deriv > 1) {
              // First col
              d2fxyzw(1, 0) = d2fxyzw(0, 1);
              d2fxyzw(2, 0) = d2fxyzw(0, 2);
              d2fxyzw(3, 0) = d2fxyzw(0, 3);
              // Second col
              d2fxyzw(2, 1) = d2fxyzw(1, 2);
              d2fxyzw(3, 1) = d2fxyzw(1, 3);
              //Third col
              d2fxyzw(3, 2) = d2fxyzw(2, 3);

              d2fxyzw = (dmat * d2fxyzw * dmat).eval();//QED
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
      fx[0] = this->tab->interp(x[0], x[1], x[2],x[3]);
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


  static void InterpTable4DBuild(py::module& m) {

      auto obj = py::class_<InterpTable4D, std::shared_ptr<InterpTable4D>>(m, "InterpTable4D");

      obj.def(py::init<
          const Eigen::VectorXd&,
          const Eigen::VectorXd&,
          const Eigen::VectorXd&,
          const Eigen::VectorXd&,
          const Eigen::Tensor<double, 4>&,
          std::string,
          bool>(),
          py::arg("xs"),
          py::arg("ys"),
          py::arg("zs"),
          py::arg("ws"),
          py::arg("fs"),
          py::arg("kind") = std::string("cubic"),
          py::arg("cache") = false);


      obj.def("interp", py::overload_cast<double, double, double,double>(&InterpTable4D::interp, py::const_));
      obj.def("interp_deriv1",
          py::overload_cast<double, double, double, double>(&InterpTable4D::interp_deriv1, py::const_));
      obj.def("interp_deriv2",
          py::overload_cast<double, double, double, double>(&InterpTable4D::interp_deriv2, py::const_));

      obj.def_readwrite("WarnOutOfBounds", &InterpTable4D::WarnOutOfBounds);
      obj.def_readwrite("ThrowOutOfBounds", &InterpTable4D::ThrowOutOfBounds);

      obj.def("__call__",
          py::overload_cast<double, double, double, double>(&InterpTable4D::interp, py::const_),
          py::is_operator());

      obj.def("__call__",
          [](std::shared_ptr<InterpTable4D>& self,
              const GenericFunction<-1, 1>& x,
              const GenericFunction<-1, 1>& y,
              const GenericFunction<-1, 1>& z,
              const GenericFunction<-1, 1>& w
              ) {
                  return GenericFunction<-1, 1>(
                      InterpFunction4D(self).eval(stack(x, y, z, w)));
          });

      obj.def("__call__",
          [](std::shared_ptr<InterpTable4D>& self,
              const Segment<-1, 1, -1>& x,
              const Segment<-1, 1, -1>& y,
              const Segment<-1, 1, -1>& z,
              const Segment<-1, 1, -1>& w

              ) {
                  return GenericFunction<-1, 1>(
                      InterpFunction4D(self).eval(stack(x, y, z, w)));
          });

      obj.def("__call__", [](std::shared_ptr<InterpTable4D>& self, const Segment<-1, -1, -1>& xyzw) {
          return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
          });

      obj.def("__call__", [](std::shared_ptr<InterpTable4D>& self, const GenericFunction<-1, -1>& xyzw) {
          return GenericFunction<-1, 1>(InterpFunction4D(self).eval(xyzw));
          });

      obj.def("sf", [](std::shared_ptr<InterpTable4D>& self) {
          return GenericFunction<-1, 1>(InterpFunction4D(self));
          });
      obj.def("vf", [](std::shared_ptr<InterpTable4D>& self) {
          return GenericFunction<-1, -1>(InterpFunction4D(self));
          });


      m.def("InterpTable4DSpeedTest",
          [](const GenericFunction<-1, 1>& tabf,
              double xl,
              double xu,
              double yl,
              double yu,
              double zl,
              double zu,
              double wl,
              double wu,

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

                  Eigen::ArrayXd zsamps;
                  zsamps.setRandom(nsamps);
                  zsamps += 1;
                  zsamps /= 2;
                  zsamps *= (zu - zl);
                  zsamps += zl;


                  Eigen::ArrayXd wsamps;
                  wsamps.setRandom(nsamps);
                  wsamps += 1;
                  wsamps /= 2;
                  wsamps *= (wu - wl);
                  wsamps += wl;


                  if (lin) {
                      xsamps.setLinSpaced(xl, xu);
                      ysamps.setLinSpaced(yl, yu);
                      zsamps.setLinSpaced(zl, zu);
                      wsamps.setLinSpaced(wl, wu);
                  }


                  Eigen::VectorXd xyzw(4);
                  Vector1<double> f;
                  f.setZero();

                  Utils::Timer Runtimer;
                  Runtimer.start();

                  double tmp = 0;
                  for (int i = 0; i < nsamps; i++) {

                      xyzw[0] = xsamps[i];
                      xyzw[1] = ysamps[i];
                      xyzw[2] = zsamps[i];
                      xyzw[3] = wsamps[i];

                      tabf.compute(xyzw, f);
                      tmp += f[0] / double(i + 3);

                      f.setZero();
                  }
                  Runtimer.stop();
                  double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
                  fmt::print("Total Time: {0:} ms \n", tseconds * 1000);


                  return tmp;
          });
  }



}