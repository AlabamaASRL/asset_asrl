#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {


  /// <summary>
  /// Build Function for Everything in this header
  /// </summary>
  /// <param name="reg"></param>
  /// <param name="m"></param>
  void KeplerUtilsBuild(FunctionRegistry& reg, py::module& m);


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////              Conversions                  /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  template<class Scalar>
  Vector6<Scalar> classic_to_cartesian(const Vector6<Scalar>& oelems, Scalar mu) {


    const int MAXITERS = 15;
    const double TOL = 1.0e-12;
    const double PI = 3.14159265358979;

    Scalar a = oelems[0];
    Scalar e = oelems[1];
    Scalar i = oelems[2];
    Scalar Omega = oelems[3];
    Scalar w = oelems[4];
    Scalar M = oelems[5];

    // Calc Eccentric anomally


    Scalar x, y, vx, vy;


    if (e < 1.0) {  // Elliptic
      Scalar E = M;
      Scalar sinE;
      Scalar cosE;
      Scalar fE;
      Scalar jE;
      for (int i = 0; i < MAXITERS; i++) {
        sinE = sin(E);
        cosE = cos(E);
        fE = E - e * sinE - M;
        if (abs(fE) < TOL)
          break;
        E = E - (fE) / (1 - e * cosE);
      }
      Scalar v = 2.0 * atan2(sqrt(1. + e) * sin(E / 2.0), sqrt(1. - e) * cos(E / 2.0));
      Scalar rc = a * (1. - e * cos(E));
      Scalar vc = sqrt(mu * a) / rc;

      x = rc * cos(v);
      y = rc * sin(v);
      vx = -vc * sinE;
      vy = vc * sqrt(1. - e * e) * cosE;


    } else {  // Hyperbolic
      Scalar H = M;
      Scalar sinhH;
      Scalar coshH;
      Scalar fH;
      Scalar jH;

      for (int i = 0; i < MAXITERS; i++) {
        sinhH = sinh(H);
        coshH = cosh(H);
        fH = e * sinhH - H - M;
        if (abs(fH) < TOL)
          break;
        H = H - (fH) / (e * coshH - 1);
      }
      Scalar rc = a * (1 - e * coshH);

      Scalar v = 2.0 * atan2(sqrt(1. + e) * sinh(H / 2.0), sqrt(e - 1) * cosh(H / 2.0));
      Scalar vc = sqrt(-mu * a) / rc;

      x = rc * cos(v);
      y = rc * sin(v);
      vx = -vc * sinhH;
      vy = vc * sqrt(e * e - 1) * coshH;
    }
    /////////////////////////

    Vector6<Scalar> XV;


    Scalar ci = cos(i);
    Scalar si = sin(i);

    Scalar cw = cos(w);
    Scalar sw = sin(w);

    Scalar cO = cos(Omega);
    Scalar sO = sin(Omega);


    XV[0] = x * (cw * cO - sw * ci * sO) - y * (sw * cO + cw * ci * sO);
    XV[1] = x * (cw * sO + sw * ci * cO) + y * (cw * ci * cO - sw * sO);
    XV[2] = x * (sw * si) + y * (cw * si);


    XV[3] = vx * (cw * cO - sw * ci * sO) - vy * (sw * cO + cw * ci * sO);
    XV[4] = vx * (cw * sO + sw * ci * cO) + vy * (cw * ci * cO - sw * sO);
    XV[5] = vx * (sw * si) + vy * (cw * si);


    return XV;
  }

  template<class Scalar>
  Vector6<Scalar> cartesian_to_classic(const Vector6<Scalar>& XV, Scalar mu) {

    const double PI = 3.14159265358979;

    Vector3<Scalar> R = XV.template head<3>();
    Vector3<Scalar> V = XV.template tail<3>();

    Vector3<Scalar> hvec = R.cross(V);
    Vector3<Scalar> evec = V.cross(hvec) / mu - R.normalized();

    Vector3<Scalar> nvec;
    nvec[0] = -hvec[1];
    nvec[1] = hvec[0];

    Scalar e = evec.norm();

    Scalar drv = R.dot(V);
    Scalar v = acos(evec.normalized().dot(R.normalized()));
    if (drv < 0)
      v = 2.0 * PI - v;

    Scalar M;

    if (e < 1) {  // Elliptic
      Scalar E = 2. * atan(tan(v / 2.0) / (sqrt((1.0 + e) / (1.0 - e))));
      M = E - e * sin(E);
    } else {  // Hyperbolic
      Scalar H = 2. * atanh(tan(v / 2.0) / (sqrt((1.0 + e) / (e - 1))));
      M = e * sinh(H) - H;
    }


    Scalar Omega = acos(nvec[0] / nvec.norm());
    if (nvec[1] < 0)
      Omega = 2.0 * PI - Omega;

    Scalar w = acos(evec.normalized().dot(nvec.normalized()));
    if (evec[2] < 0)
      w = 2.0 * PI - w;

    Scalar i = acos(hvec[2] / hvec.norm());

    Scalar a = 1.0 / (2.0 / R.norm() - V.squaredNorm() / mu);

    Vector6<Scalar> oelems;

    oelems[0] = a;
    oelems[1] = e;
    oelems[2] = i;
    oelems[3] = Omega;
    oelems[4] = w;
    oelems[5] = M;
    return oelems;
  }

  template<class Scalar>
  Vector6<Scalar> modified_to_cartesian(const Vector6<Scalar>& meelems, Scalar mu) {

    Scalar p = meelems[0];
    Scalar f = meelems[1];
    Scalar g = meelems[2];
    Scalar h = meelems[3];
    Scalar k = meelems[4];
    Scalar L = meelems[5];

    Scalar cosL = cos(L);
    Scalar sinL = sin(L);


    Scalar a2 = h * h - k * k;
    Scalar s2 = 1 + h * h + k * k;
    Scalar w = 1 + f * cosL + g * sinL;
    Scalar rr = p / w;

    Scalar Xscale = rr / s2;
    Scalar Vscale = sqrt(mu / p) / s2;


    Vector6<Scalar> XV;

    XV[0] = Xscale * (cosL + a2 * cosL + 2 * h * k * sinL);
    XV[1] = Xscale * (sinL - a2 * sinL + 2 * h * k * cosL);
    XV[2] = 2 * Xscale * (h * sinL - k * cosL);

    XV[3] = -Vscale * (sinL + a2 * sinL - 2 * h * k * cosL + g - 2 * f * h * k + a2 * g);
    XV[4] = -Vscale * (-cosL + a2 * cosL + 2 * h * k * sinL - f + 2 * g * h * k + a2 * f);
    XV[5] = 2 * Vscale * (h * cosL + k * sinL + f * h + g * k);

    return XV;
  }

  template<class Scalar>
  Vector6<Scalar> modified_to_classic(const Vector6<Scalar>& meelems, Scalar mu) {

    Scalar p = meelems[0];
    Scalar f = meelems[1];
    Scalar g = meelems[2];
    Scalar h = meelems[3];
    Scalar k = meelems[4];
    Scalar L = meelems[5];


    Scalar a = p / (1 - f * f - g * g);
    Scalar e = sqrt(f * f + g * g);
    Scalar i = atan2(2 * sqrt(h * h + k * k), 1 - h * h - k * k);
    Scalar Omega = atan2(k, h);
    Scalar w = atan2(g * h - f * k, f * h + g * k);
    Scalar v = L - (Omega + w);


    Scalar M;

    if (e < 1) {  // Elliptic
      Scalar E = 2. * atan(tan(v / 2.0) / (sqrt((1.0 + e) / (1.0 - e))));
      M = E - e * sin(E);
    } else {  // Hyperbolic
      Scalar H = 2. * atanh(tan(v / 2.0) / (sqrt((1.0 + e) / (e - 1))));
      M = e * sinh(H) - H;
    }

    Vector6<Scalar> oelems;

    oelems[0] = a;
    oelems[1] = e;
    oelems[2] = i;
    oelems[3] = Omega;
    oelems[4] = w;
    oelems[5] = M;
    return oelems;
  }

  template<class Scalar>
  Vector6<Scalar> classic_to_modified(const Vector6<Scalar>& oelems, Scalar mu) {


    const int MAXITERS = 15;
    const double TOL = 1.0e-12;
    const double PI = 3.14159265358979;

    Scalar a = oelems[0];
    Scalar e = oelems[1];
    Scalar i = oelems[2];
    Scalar Omega = oelems[3];
    Scalar w = oelems[4];
    Scalar M = oelems[5];

    // Calc True anomally
    Scalar v;
    if (e < 1.0) {  // Elliptic
      Scalar E = M;
      Scalar sinE;
      Scalar cosE;
      Scalar fE;
      Scalar jE;
      for (int i = 0; i < MAXITERS; i++) {
        sinE = sin(E);
        cosE = cos(E);
        fE = E - e * sinE - M;
        if (abs(fE) < TOL)
          break;
        E = E - (fE) / (1 - e * cosE);
      }
      v = 2.0 * atan2(sqrt(1. + e) * sin(E / 2.0), sqrt(1. - e) * cos(E / 2.0));
    } else {  // Hyperbolic
      Scalar H = M;
      Scalar sinhH;
      Scalar coshH;
      Scalar fH;
      Scalar jH;

      for (int i = 0; i < MAXITERS; i++) {
        sinhH = sinh(H);
        coshH = cosh(H);
        fH = e * sinhH - H - M;
        if (abs(fH) < TOL)
          break;
        H = H - (fH) / (e * coshH - 1);
      }
      v = 2.0 * atan2(sqrt(1. + e) * sinh(H / 2.0), sqrt(e - 1) * cosh(H / 2.0));
    }
    /////////////////////////


    Vector6<Scalar> meelems;

    meelems[0] = a * (1 - e * e);          // p
    meelems[1] = e * cos(w + Omega);       // f
    meelems[2] = e * sin(w + Omega);       // g
    meelems[3] = tan(i / 2) * cos(Omega);  // h
    meelems[4] = tan(i / 2) * sin(Omega);  // k
    meelems[5] = w + Omega + v;            // L

    return meelems;
  }

  template<class Scalar>
  Vector6<Scalar> cartesian_to_modified(const Vector6<Scalar>& XV, Scalar mu) {
    Vector6<Scalar> oelems = cartesian_to_classic(XV, mu);
    return classic_to_modified(oelems, mu);
  }

  template<class Scalar>
  Vector6<Scalar> cartesian_to_classic_true(const Vector6<Scalar>& XV, Scalar mu) {

    const double PI = 3.14159265358979;

    Vector3<Scalar> R = XV.template head<3>();
    Vector3<Scalar> V = XV.template tail<3>();

    Vector3<Scalar> hvec = R.cross(V);
    Vector3<Scalar> evec = V.cross(hvec) / mu - R.normalized();

    Vector3<Scalar> nvec;
    nvec[0] = -hvec[1];
    nvec[1] = hvec[0];

    Scalar e = evec.norm();

    Scalar drv = R.dot(V);
    Scalar v = acos(evec.normalized().dot(R.normalized()));
    if (drv < 0)
      v = 2.0 * PI - v;

    Scalar Omega = acos(nvec[0] / nvec.norm());
    if (nvec[1] < 0)
      Omega = 2.0 * PI - Omega;

    Scalar w = acos(evec.normalized().dot(nvec.normalized()));
    if (evec[2] < 0)
      w = 2.0 * PI - w;

    Scalar i = acos(hvec[2] / hvec.norm());

    Scalar a = 1.0 / (2.0 / R.norm() - V.squaredNorm() / mu);

    Vector6<Scalar> oelems;

    oelems[0] = a;
    oelems[1] = e;
    oelems[2] = i;
    oelems[3] = Omega;
    oelems[4] = w;
    oelems[5] = v;
    return oelems;
  }


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////         Conversions as ASSET Functions    /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  struct ModifiedToCartesian_Impl {
    /// <summary>
    /// ASSET Function to convert from Modified Equinoctual Elements to Cartesian State
    /// </summary>
    /// <param name="mu"></param>
    /// <returns></returns>
    static auto Definition(double mu) {

      auto meelems = Arguments<6>();


      auto p = meelems.coeff<0>();
      auto f = meelems.coeff<1>();
      auto g = meelems.coeff<2>();
      auto h = meelems.coeff<3>();
      auto k = meelems.coeff<4>();
      auto L = meelems.coeff<5>();

      auto cosL = cos(L);
      auto sinL = sin(L);


      auto a2 = h * h - k * k;
      auto s2 = 1.0 + h * h + k * k;
      auto w = 1.0 + f * cosL + g * sinL;
      auto rr = p / w;

      auto Xscale = rr / s2;
      auto Vscale = sqrt(mu / p) / s2;

      auto x = (cosL + a2 * cosL + 2.0 * h * k * sinL);
      auto y = (sinL - a2 * sinL + 2.0 * h * k * cosL);
      auto z = 2.0 * (h * sinL - k * cosL);

      auto R = stack(x, y, z) * Xscale;

      auto vx = -1.0 * (sinL + a2 * sinL - 2 * h * k * cosL + g - 2 * f * h * k + a2 * g);
      auto vy = -1.0 * (-1.0 * cosL + a2 * cosL + 2 * h * k * sinL - f + 2 * g * h * k + a2 * f);
      auto vz = 2 * (h * cosL + k * sinL + f * h + g * k);

      auto V = stack(vx, vy, vz) * Vscale;

      return stack(R, V);
    }
  };

  BUILD_FROM_EXPRESSION(ModifiedToCartesian, ModifiedToCartesian_Impl, double);


  struct CartesianToClassic_Impl {
    /// <summary>
    /// ASSET Function to convert from Modified Equinoctual Elements to Cartesian State
    /// </summary>
    /// <param name="mu"></param>
    /// <returns></returns>
    static auto Definition(double mu) {
      const double PI = 3.14159265358979;

      auto RV = Arguments<6>();

      auto ZVec = Constant<6, 3>(6, Eigen::Vector3d::UnitZ());

      auto R = RV.head<3>();
      auto V = RV.tail<3>();
      auto hvec = R.cross(V);
      auto nvec = ZVec.cross(hvec);

      auto r = R.norm();
      auto v2 = V.squared_norm();
      auto eps = v2 / 2.0 - mu / r;


      auto a = (-0.5 * mu) / eps;
      auto evec = V.cross(hvec) / mu - R.normalized();
      auto e = evec.norm();

      auto drv = R.dot(V);

      auto vtmp = acos(evec.normalized().dot(R.normalized()));

      auto v = IfElseFunction {drv < 0, vtmp, 2 * PI - vtmp};

      auto M =
          [mu]() {
            auto ev = Arguments<2>();
            auto e = ev.coeff<0>();
            auto v = ev.coeff<1>();

            auto E = 2. * atan(tan(v / 2.0) / (sqrt((1.0 + e) / (1.0 - e))));
            auto ME = E - e * sin(E);

            auto H = 2. * atanh(tan(v / 2.0) / (sqrt((1.0 + e) / (e - 1.0))));
            auto MH = e * sinh(H) - H;

            auto M = IfElseFunction {e < 1.0, ME, MH};
            return M;
          }()
              .eval(stack(e, v));


      auto Omegatmp = acos(nvec.coeff<0>() / nvec.norm());
      auto Omega = IfElseFunction {nvec.coeff<1>() < 0, 2.0 * PI - Omegatmp, Omegatmp};
      auto wtmp = acos(evec.normalized().dot(nvec.normalized()));
      auto w = IfElseFunction {evec.coeff<2>() < 0, 2.0 * PI - wtmp, wtmp};
      auto i = acos(hvec.coeff<2>() / hvec.norm());


      return stack(a, e, i, Omega, w, M);
    }
  };

  BUILD_FROM_EXPRESSION(CartesianToClassic, CartesianToClassic_Impl, double)


  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////              Propagators                  /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  template<class Scalar>
  Vector6<Scalar> propagate_cartesian(const Vector6<Scalar>& RV, Scalar dt, Scalar mu) {

    const double Pi = 3.14159265358979;
    const double ptol = 1.0e-9;
    const double tol = 1.0e-12;
    const int iters = 19;
    const double conictol = 1.0e-11;


    Scalar r = RV.template head<3>().norm();
    Scalar v = RV.template tail<3>().norm();
    Scalar SQM = sqrt(mu);

    Scalar alpha = -(v * v) / (mu) + (2.0 / r);
    Scalar a = 1.0 / alpha;
    Scalar X0;
    if (alpha > conictol)
      X0 = SQM * dt * alpha;
    else if (alpha < conictol && alpha > -conictol) {
      Vector3<Scalar> h = RV.template head<3>().cross(RV.template tail<3>());
      Scalar hmag = h.norm();
      Scalar p = hmag * hmag / mu;
      Scalar s = (Pi / 2.0 - atan(3.0 * sqrt(mu / (p * p * p)) * dt)) / 2.0;
      Scalar w = atan(cbrt(tan(s)));
      X0 = sqrt(p) * 2 / tan(2 * w);
    } else {
      X0 = (abs(dt) / dt) * sqrt(-a)
           * log(abs((-2.0 * mu * alpha * dt)
                     / (RV.template head<3>().dot(RV.template tail<3>()) + abs(dt) / dt) * sqrt(-mu * a)
                     * (1.0 - r * alpha)));
    }
    Scalar DRV = RV.template head<3>().dot(RV.template tail<3>()) / SQM;
    Scalar SQMDT = SQM * dt;
    Scalar c2, c3, Xn, psi, rs, X02, X03, X0tOmPsiC3, X02tC2, err;
    for (int i = 0; i < iters; i++) {
      X02 = X0 * X0;
      X03 = X02 * X0;
      psi = X02 * alpha;

      if (psi > ptol) {  // ellpitic
        Scalar sqsi = sqrt(psi);
        c2 = (1.0 - cos(sqsi)) / psi;
        c3 = (sqsi - sin(sqsi)) / (sqsi * psi);
      } else if (psi > -ptol && psi < ptol) {  // parabolic
        c2 = 0.5;
        c3 = 1.0 / 6.0;
      } else {  // hyperbolic
        c2 = (1.0 - cosh(sqrt(-psi))) / psi;
        c3 = (sinh(sqrt(-psi)) - sqrt(-psi)) / sqrt(-psi * psi * psi);
      }
      X0tOmPsiC3 = X0 * (1.0 - psi * c3);
      X02tC2 = X02 * c2;
      rs = X02tC2 + DRV * X0tOmPsiC3 + r * (1.0 - psi * c2);
      Xn = X0 + (SQMDT - X03 * c3 - DRV * X02tC2 - r * X0tOmPsiC3) / rs;
      err = Xn - X0;
      X0 = Xn;
      if (abs(err) < tol)
        break;
    }
    Scalar Xn2 = Xn * Xn;
    Scalar f = 1.0 - Xn2 * c2 / r;
    Scalar g = dt - (Xn2 * Xn) * c3 / SQM;
    Scalar fdot = Xn * (psi * c3 - 1.0) * SQM / (rs * r);
    Scalar gdot = 1.0 - c2 * (Xn2) / rs;

    Vector6<Scalar> fx;

    fx.template head<3>() = f * RV.template head<3>() + g * RV.template tail<3>();
    fx.template tail<3>() = fdot * RV.template head<3>() + gdot * RV.template tail<3>();
    return fx;
  }

  template<class Scalar>
  Vector6<Scalar> propagate_classic(const Vector6<Scalar>& oelems, Scalar dt, Scalar mu) {
    Scalar a = oelems[0];
    Scalar n = sqrt(mu / abs(a * a * a));
    Vector6<Scalar> noelems = oelems;
    noelems[5] += n * dt;
    return noelems;
  }

  template<class Scalar>
  Vector6<Scalar> propagate_modified(const Vector6<Scalar>& meelems, Scalar dt, Scalar mu) {
    Vector6<Scalar> oelems = modified_to_classic(meelems, mu);
    Vector6<Scalar> noelems = propagate_classic(oelems, dt, mu);
    return classic_to_modified(noelems, mu);
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////              Propagators                  /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////


}  // namespace ASSET