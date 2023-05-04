#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {


  struct KeperPropagator_Impl {

    static auto C2(double tol) {

      auto psi = Arguments<1>().coeff<0>();
      auto sqsi = sqrt(psi);


      auto ell = (1.0 - cos(sqsi)) / psi;
      auto hyp = (1.0 - cosh(sqrt(-1.0 * psi))) / psi;
      Vector1<double> v;
      v[0] = .5;

      auto par = Constant<1, 1>(1, v);

      auto f = IfElseFunction {psi > tol, ell, IfElseFunction {psi < -tol, hyp, par}};

      return GenericFunction<1, 1>(f);
    }

    static auto C3(double tol) {

      auto psi = Arguments<1>().coeff<0>();
      auto sqsi = sqrt(psi);


      auto ell = (sqsi - sin(sqsi)) / (sqsi * psi);
      auto hyp = (sinh(sqrt(-1.0 * psi)) - sqrt(-1.0 * psi)) / sqrt(-1.0 * psi * psi * psi);

      Vector1<double> v;

      v[0] = 1 / 6.0;

      auto par = Constant<1, 1>(1, v);

      auto f = IfElseFunction {psi > tol, ell, IfElseFunction {psi < -tol, hyp, par}};


      return GenericFunction<1, 1>(f);
    }


    static auto C2C3(double tol) {

      auto psi = Arguments<1>().coeff<0>();
      auto sqsi = sqrt(psi);


      auto ell = stack((1.0 - cos(sqsi)) / psi, (sqsi - sin(sqsi)) / (sqsi * psi));
      auto hyp = stack((1.0 - cosh(sqrt(-1.0 * psi))) / psi,
                       (sinh(sqrt(-1.0 * psi)) - sqrt(-1.0 * psi)) / sqrt(-1.0 * psi * psi * psi));

      Vector2<double> v;
      v[0] = .5;
      v[1] = 1 / 6.0;

      auto par = Constant<1, 2>(1, v);

      auto f = IfElseFunction {psi > tol, ell, IfElseFunction {psi < -tol, hyp, par}};


      return GenericFunction<1, 2>(f);
    }

    static auto Funiv(double mu, double conictol, double roottol, int iters) {

      auto args = Arguments<5>();

      auto X0 = args.coeff<0>();
      auto dt = args.coeff<1>();
      auto r = args.coeff<2>();
      auto drv = args.coeff<3>();
      auto alpha = args.coeff<4>();


      auto X02 = X0 * X0;


      auto psi = X02 * alpha;

      auto c2 = C2(conictol).eval(psi);
      auto c3 = C3(conictol).eval(psi);

      auto c2c3 = C2C3(conictol).eval(psi);

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
        auto dFU = -1.0 * (X02tC2 + drv * X0tOmPsiC3 + r * (1.0 - psi * c2));

        return std::tuple<decltype(FU), decltype(dFU)> {FU, dFU};
      }();

      auto FU = F.eval(Nargs);
      auto dFU = dF.eval(Nargs);


      auto X0F = ScalarRootFinder<decltype(FU), decltype(dFU)> {FU, dFU, iters, roottol};

      return stack(GenericFunction<5, 1>(X0F), args.tail<4>());
    }


    static auto FGs(double mu, double conictol) {

      auto args = Arguments<5>();

      auto X0 = args.coeff<0>();
      auto dt = args.coeff<1>();
      auto r = args.coeff<2>();
      auto drv = args.coeff<3>();
      auto alpha = args.coeff<4>();


      auto X02 = X0 * X0;


      auto psi = X02 * alpha;

      auto c2 = C2(conictol).eval(psi);
      auto c3 = C3(conictol).eval(psi);

      auto c2c3 = C2C3(conictol).eval(psi);


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
        auto rs = (X02tC2 + drv * X0tOmPsiC3 + r * (1.0 - psi * c2));

        auto f = 1.0 - X02 * c2 / r;
        auto g = dt - (X02 * X0) * c3 / SQM;
        auto fdot = X0 * (psi * c3 - 1.0) * SQM / (rs * r);
        auto gdot = 1.0 - c2 * (X02) / rs;


        return stack(f, g, fdot, gdot);
      }();


      return FG.eval(Nargs);
    }


    static auto ApplyRVFG() {

      auto args = Arguments<10>();

      auto R = args.head<3>();
      auto V = args.segment<3, 3>();

      auto f = args.coeff<6>();
      auto g = args.coeff<7>();
      auto fdot = args.coeff<8>();
      auto gdot = args.coeff<9>();

      auto Rf = f * R + g * V;
      auto Vf = fdot * R + gdot * V;

      return stack(Rf, Vf);
    }


    static auto Definition(double mu, double conictol, double roottol, int iters) {

      double SQM = sqrt(mu);

      auto RVdt = Arguments<7>();

      auto R = RVdt.head<3>();
      auto V = RVdt.segment<3, 3>();
      auto dt = RVdt.coeff<6>();


      auto r = R.norm();
      auto v = V.norm();
      auto drv = R.dot(V) / SQM;
      auto alpha = -1.0 * (v * v) / (mu) + (2.0 / r);

      auto X0ell = SQM * dt * alpha;


      auto signdt = SignFunction {dt};
      auto X0hyp =
          signdt * sqrt(-1.0 / alpha)
          * log(abs((-2.0 * mu * alpha * dt) / (R.dot(V) + signdt) * sqrt(-mu / alpha) * (1.0 - r * alpha)));

      auto X0IG = GenericFunction<7, 1>(IfElseFunction {alpha >= 0.0, X0ell, X0hyp});

      auto XF = Funiv(mu, conictol, roottol, iters).eval(stack(X0IG, dt, r, drv, alpha));

      auto FG = FGs(mu, conictol).eval(XF);

      constexpr bool d = FG.IsVectorizable;

      return ApplyRVFG().eval(stack(RVdt.head<6>(), FG));
    }
  };


  struct KeplerPropagator : VectorFunction<KeplerPropagator,
                                           7,
                                           6,
                                           ASSET::DenseDerivativeModes::Analytic,
                                           ASSET::DenseDerivativeModes::Analytic> {
    using Base = VectorFunction<KeplerPropagator,
                                7,
                                6,
                                ASSET::DenseDerivativeModes::Analytic,
                                ASSET::DenseDerivativeModes::Analytic>;

    DENSE_FUNCTION_BASE_TYPES(Base);


    GenericFunction<-1, -1> propfunc;
    double mu = 1.0;
    double tol = 1.0e-12;
    int iters = 19;
    double conictol = 1.0e-11;
    const double Pi = 3.14159265358979;
    double t0 = 0.0;

    static const bool IsVectorizable = true;

    KeplerPropagator(double m) {
      this->mu = m;
      this->propfunc = KeperPropagator_Impl::Definition(this->mu, this->conictol, this->tol, this->iters);
    }
    KeplerPropagator() {
      this->propfunc = KeperPropagator_Impl::Definition(this->mu, this->conictol, this->tol, this->iters);
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<KeplerPropagator>(m, name);
      obj.def(py::init<double>());
      Base::DenseBaseBuild(obj);
    }

    template<class Scalar>
    static Scalar CBRT(Scalar x) {
      if constexpr (std::is_same<Scalar, double>::value) {
        return cbrt(x);
      } else {
        return Scalar(pow(x, 1.0 / 3.0));
      }
    }

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      this->propfunc.compute(x, fx_);
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      this->propfunc.compute_jacobian(x, fx_, jx_);
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


      this->propfunc.compute_jacobian_adjointgradient_adjointhessian(
          x, fx_, jx_, adjgrad_, adjhess_, adjvars);
    }

    /*
    template <class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
        Eigen::MatrixBase<OutType> const& fx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
        Eigen::Matrix<Scalar, 6, 1> RV = x.template head<6>();
        Scalar dt = x[6];

        Scalar r = RV.template head<3>().norm();
        Scalar v = RV.template tail<3>().norm();
        double SQM = sqrt(mu);

        Scalar alpha = -(v * v) / (mu)+(2.0 / r);
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
        }
        else {
            X0 = (abs(dt) / dt) * sqrt(-a) *
                log(abs((-2.0 * mu * alpha * dt) /
                    (RV.template head<3>().dot(RV.template tail<3>()) +
                        abs(dt) / dt) *
                    sqrt(-mu * a) * (1.0 - r * alpha)));
        }
        Scalar DRV = RV.template head<3>().dot(RV.template tail<3>()) / SQM;
        Scalar SQMDT = SQM * dt;
        double ptol = 1.0e-9;
        Scalar c2, c3, Xn, psi, rs, X02, X03, X0tOmPsiC3, X02tC2, err;
        for (int i = 0; i < iters; i++) {
            X02 = X0 * X0;
            X03 = X02 * X0;
            psi = X02 * alpha;

            if (psi > ptol) {  // ellpitic
                Scalar sqsi = sqrt(psi);
                c2 = (1.0 - cos(sqsi)) / psi;
                c3 = (sqsi - sin(sqsi)) / (sqsi * psi);
            }
            else if (psi > -ptol && psi < ptol) {  // parabolic
                c2 = 0.5;
                c3 = 1.0 / 6.0;
            }
            else {  // hyperbolic
                c2 = (1.0 - cosh(sqrt(-psi))) / psi;
                c3 = (sinh(sqrt(-psi)) - sqrt(-psi)) / sqrt(-psi * psi * psi);
            }
            X0tOmPsiC3 = X0 * (1.0 - psi * c3);
            X02tC2 = X02 * c2;
            rs = X02tC2 + DRV * X0tOmPsiC3 + r * (1.0 - psi * c2);
            Xn = X0 + (SQMDT - X03 * c3 - DRV * X02tC2 - r * X0tOmPsiC3) / rs;
            err = Xn - X0;
            X0 = Xn;
            if (abs(err) < tol) break;
        }
        Scalar Xn2 = Xn * Xn;
        Scalar f = 1.0 - Xn2 * c2 / r;
        Scalar g = dt - (Xn2 * Xn) * c3 / SQM;
        Scalar fdot = Xn * (psi * c3 - 1.0) * SQM / (rs * r);
        Scalar gdot = 1.0 - c2 * (Xn2) / rs;



        fx.template head<3>() =
            f * RV.template head<3>() + g * RV.template tail<3>();
        fx.template tail<3>() =
            fdot * RV.template head<3>() + gdot * RV.template tail<3>();
    }

    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
        Eigen::MatrixBase<OutType> const& fx_,
        Eigen::MatrixBase<JacType> const& jx_) const {
        typedef typename InType::Scalar Scalar;
        Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
        Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

        Eigen::Matrix<Scalar, 6, 1> RV = x.template head<6>();
        Scalar dt = x[6];

        Scalar r = RV.template head<3>().norm();
        Scalar v = RV.template tail<3>().norm();
        double SQM = sqrt(mu);

        Scalar alpha = -(v * v) / (mu)+(2.0 / r);
        Scalar a = 1.0 / alpha;
        Scalar X0;
        if (alpha > conictol)
            X0 = SQM * dt * alpha;
        else if (alpha < conictol && alpha > -conictol) {
            Vector3<Scalar> h = RV.template head<3>().cross(RV.template tail<3>());
            Scalar hmag = h.norm();
            Scalar p = hmag * hmag / mu;
            Scalar s = (Pi / 2.0 - atan(3.0 * sqrt(mu / (p * p * p)) * dt)) / 2.0;
            Scalar w = atan(CBRT(Scalar(tan(s))));
            X0 = sqrt(p) * 2 / tan(2 * w);
        }
        else {
            X0 = (abs(dt) / dt) * sqrt(-a) *
                log(abs((-2.0 * mu * alpha * dt) /
                    (RV.template head<3>().dot(RV.template tail<3>()) +
                        abs(dt) / dt) *
                    sqrt(-mu * a) * (1.0 - r * alpha)));
            // std::cout << "here";
        }
        Scalar DRV = RV.template head<3>().dot(RV.template tail<3>()) / SQM;
        Scalar SQMDT = SQM * dt;
        double ptol = 1.0e-9;
        Scalar c2, c3, Xn, psi, rs, X02, X03, X0tOmPsiC3, X02tC2, err;
        for (int i = 0; i < iters; i++) {

            //std::cout << i << " "<< X0 << std::endl;
            X02 = X0 * X0;
            X03 = X02 * X0;
            psi = X02 * alpha;
            // StumpfC2C3(psi, c2, c3);
            if (psi > ptol) {  // ellpitic
                Scalar sqsi = sqrt(psi);
                c2 = (1.0 - cos(sqsi)) / psi;
                c3 = (sqsi - sin(sqsi)) / (sqsi * psi);
            }
            else if (psi > -ptol && psi < ptol) {  // parabolic
                c2 = 0.5;
                c3 = 1.0 / 6.0;
                //std::cout << "ellint";
            }
            else {  // hyperbolic
                c2 = (1.0 - cosh(sqrt(-psi))) / psi;
                c3 = (sinh(sqrt(-psi)) - sqrt(-psi)) / sqrt(-psi * psi * psi);
            }
            X0tOmPsiC3 = X0 * (1.0 - psi * c3);
            X02tC2 = X02 * c2;
            rs = X02tC2 + DRV * X0tOmPsiC3 + r * (1.0 - psi * c2);
            Xn = X0 + (SQMDT - X03 * c3 - DRV * X02tC2 - r * X0tOmPsiC3) / rs;
            err = Xn - X0;
            X0 = Xn;

            if (abs(err) < tol) break;
        }
        Scalar Xn2 = Xn * Xn;
        Scalar f = 1.0 - Xn2 * c2 / r;
        Scalar g = dt - (Xn2 * Xn) * c3 / SQM;
        Scalar fdot = Xn * (psi * c3 - 1.0) * SQM / (rs * r);
        Scalar gdot = 1.0 - c2 * (Xn2) / rs;

        Vector3<Scalar> R0 = RV.template head<3>();
        Vector3<Scalar> V0 = RV.template tail<3>();
        Vector3<Scalar> R1 = f * RV.template head<3>() + g * RV.template tail<3>();
        Vector3<Scalar> V1 =
            fdot * RV.template head<3>() + gdot * RV.template tail<3>();

        Scalar rn = R1.norm();
        Scalar rn2 = rn * rn;
        Scalar rn3 = rn2 * rn;

        Scalar r2 = r * r;
        Scalar r3 = r2 * r;

        Scalar XI = alpha * SQM * dt + V1.dot(R1) / SQM - DRV;

        Scalar U2 = r * (1.0 - f);
        Scalar U3 = SQM * (dt - g);
        Scalar U4 = (XI * XI / 2.0 - U2) * a;
        Scalar U5 = (XI * XI * XI / 6.0 - U3) * a;

        Scalar C = (1.0 / SQM) * (3.0 * U5 - XI * U4 - SQM * dt * U2);

        Eigen::Matrix<Scalar, 3, 3> I;
        I.setIdentity();

        Vector3<Scalar> DR = R1 - R0;
        Vector3<Scalar> DV = V1 - V0;

        Eigen::Matrix<Scalar, 3, 3> DVT = DV * DV.transpose();
        Eigen::Matrix<Scalar, 3, 3> R1R0T = R1 * R0.transpose();
        Eigen::Matrix<Scalar, 3, 3> V1R0T = V1 * R0.transpose();

        Eigen::Matrix<Scalar, 3, 3> DRV0T = DR * V0.transpose();
        Eigen::Matrix<Scalar, 3, 3> DVR0T = DV * R0.transpose();
        Eigen::Matrix<Scalar, 3, 3> VVT = V1 * V0.transpose();

        Eigen::Matrix<Scalar, 3, 3> R1DVT = R1 * DV.transpose();
        Eigen::Matrix<Scalar, 3, 3> R1R1T = ((1.0 / (rn2)) * R1) * R1.transpose();

        Eigen::Matrix<Scalar, 3, 3> R1V1T_V1R1T =
            (1.0 / (mu * rn)) * (R1 * V1.transpose() - V1 * R1.transpose());
        // Matrix<Scalar, 3, 3> V1R1T = V1 * R1.transpose();

        Eigen::Matrix<Scalar, 3, 3> R1V0T = R1 * V0.transpose();

        fx.template head<3>() = R1;
        fx.template tail<3>() = V1;

        Scalar rmrf = (r - r * f);

        jx.template block<3, 3>(0, 0) =
            (rn / mu) * (DVT)+(1.0 / (r3)) * ((rmrf)*R1R0T + C * V1R0T) + f * I;
        jx.template block<3, 3>(0, 3) =
            ((r / mu) * (1.0 - f)) * (DRV0T - DVR0T) + (C / mu) * VVT + g * I;

        jx.template block<3, 3>(3, 0) = (-1.0 / (r2)) * DVR0T +
            (-1.0 / (rn2)) * R1DVT +
            fdot * (I - R1R1T + (R1V1T_V1R1T)*R1DVT) -
            (mu * C / (rn3 * r3)) * R1R0T;
        jx.template block<3, 3>(3, 3) =
            (r / mu) * DVT + (1.0 / (rn3)) * ((rmrf)*R1R0T - C * R1V0T) + gdot * I;

        jx.col(6).template head<3>() = V1;
        jx.col(6).template tail<3>() = -mu * R1 / (rn3);





    }*/
  };


}  // namespace ASSET