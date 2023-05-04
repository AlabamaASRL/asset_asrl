#pragma once

#include "OptimalControlFlags.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "pch.h"


namespace ASSET {

  template<class DODE, class Integrator>
  struct ShootingDefect_Impl {
    static auto Definition(const DODE& ode, const Integrator& integ) {
      constexpr int IRC = SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value;
      int input_rows = ode.XtUVars() * 2 + ode.PVars();

      auto args = Arguments<IRC>(input_rows);
      // Input[x1,t1,u1,x2,t2,u2,pv]

      auto x1 = args.template head<DODE::XtUV>(ode.XtUVars());
      auto t1 = x1.template coeff<DODE::XV>(ode.XVars());
      auto x2 = args.template segment<DODE::XtUV, DODE::XtUV>(ode.XtUVars(), ode.XtUVars());
      auto t2 = x2.template coeff<DODE::XV>(ode.XVars());

      auto tm = 0.5 * (t1 + t2);

      auto pvars = args.template tail<DODE::PV>(ode.PVars());

      auto make_state = [&](auto xx) {
        if constexpr (DODE::PV == 0) {
          return StackedOutputs {xx, tm};
        } else {
          return StackedOutputs {xx, pvars, tm};
        }
      };

      auto Arc1Input = make_state(x1);
      auto Arc2Input = make_state(x2);

      auto defect = integ.eval(Arc1Input).template head<DODE::XV>(ode.XVars())
                    - integ.eval(Arc2Input).template head<DODE::XV>(ode.XVars());

      return defect;
    }
  };

  template<class DODE, class Integrator>
  struct ShootingDefect : VectorExpression<ShootingDefect<DODE, Integrator>,
                                           ShootingDefect_Impl<DODE, Integrator>,
                                           const DODE&,
                                           const Integrator&> {
    using Base = VectorExpression<ShootingDefect<DODE, Integrator>,
                                  ShootingDefect_Impl<DODE, Integrator>,
                                  const DODE&,
                                  const Integrator&>;
    // using Base::Base;
    ShootingDefect() {
    }
    ShootingDefect(const DODE& ode, const Integrator& integ) : Base(ode, integ) {
    }
    bool EnableHessianSparsity = false;
  };


  template<class DODE, class Integrator>
  struct CentralShootingDefect : VectorFunction<CentralShootingDefect<DODE, Integrator>,
                                                SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value,
                                                DODE::XV> {

    using Base = VectorFunction<CentralShootingDefect<DODE, Integrator>,
                                SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value,
                                DODE::XV>;

    DENSE_FUNCTION_BASE_TYPES(Base);


    template<class Scalar>
    using ODEState = typename DODE::template Input<Scalar>;
    template<class Scalar>
    using ODEDeriv = typename DODE::template Output<Scalar>;
    template<class Scalar>
    using IntegJac = typename Integrator::template Jacobian<Scalar>;

    static const bool IsVectorizable = true;
    bool EnableHessianSparsity = false;

    DODE ode;
    Integrator integ;

    CentralShootingDefect(const DODE& ode, const Integrator& integ) : ode(ode), integ(integ) {
      this->setIORows(2 * this->ode.XtUVars() + this->ode.PVars(), this->ode.XVars());
    }


    CentralShootingDefect() {
    }

    template<class InType>
    void extract_scalar_inputs(ConstVectorBaseRef<InType> X1X2, std::vector<Input<double>>& X1X2s) const {

      typedef typename InType::Scalar Scalar;


      X1X2s.resize(Scalar::SizeAtCompileTime);
      for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
        X1X2s[v].resize(this->IRows());
        for (int i = 0; i < this->IRows(); i++) {
          X1X2s[v][i] = X1X2[i][v];
        }
      }
    }

    template<class InType>
    void extract_scalar_lmults(ConstVectorBaseRef<InType> Lf, std::vector<Output<double>>& Lfs) const {

      typedef typename InType::Scalar Scalar;

      Lfs.resize(Scalar::SizeAtCompileTime);
      for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
        Lfs[v].resize(this->ORows());
        for (int i = 0; i < this->ORows(); i++) {
          Lfs[v][i] = Lf[i][v];
        }
      }
    }


    void get_input_states_tfs(const std::vector<Input<double>>& X1X2s,
                              std::vector<ODEState<double>>& Xs,
                              Eigen::VectorXd& tfs) const {


      Xs.resize(2 * X1X2s.size());
      tfs.resize(2 * X1X2s.size());

      for (int i = 0; i < X1X2s.size(); i++) {

        Xs[2 * i].resize(this->ode.IRows());
        Xs[2 * i + 1].resize(this->ode.IRows());

        Xs[2 * i].head(this->ode.XtUVars()) = X1X2s[i].head(this->ode.XtUVars());
        Xs[2 * i + 1].head(this->ode.XtUVars()) = X1X2s[i].segment(this->ode.XtUVars(), this->ode.XtUVars());

        double tm = (Xs[2 * i][this->ode.TVar()] + Xs[2 * i + 1][this->ode.TVar()]) / 2.0;

        tfs[2 * i] = tm;
        tfs[2 * i + 1] = tm;

        if constexpr (DODE::PV != 0) {

          Xs[2 * i].tail(this->ode.PVars()) = X1X2s[i].tail(this->ode.PVars());
          Xs[2 * i + 1].tail(this->ode.PVars()) = X1X2s[i].tail(this->ode.PVars());
        }
      }
    }
    void get_lmults(const std::vector<Output<double>>& Ls, std::vector<ODEState<double>>& Lfs) const {

      Lfs.resize(2 * Ls.size());

      for (int i = 0; i < Ls.size(); i++) {

        Lfs[2 * i].resize(this->ode.IRows());
        Lfs[2 * i + 1].resize(this->ode.IRows());
        Lfs[2 * i].setZero();
        Lfs[2 * i + 1].setZero();

        Lfs[2 * i].head(this->ode.XVars()) = Ls[i];
        Lfs[2 * i + 1].head(this->ode.XVars()) = Ls[i];
      }
    }


    std::vector<Output<double>> compute_impl_v(const std::vector<Input<double>>& X1X2s) const {

      std::vector<ODEState<double>> Xs;
      Eigen::VectorXd tfs;
      std::vector<ODEState<double>> Xfs;

      this->get_input_states_tfs(X1X2s, Xs, tfs);

      Xfs = this->integ.integrate(Xs, tfs);

      std::vector<Output<double>> fxs(X1X2s.size());

      for (int i = 0; i < X1X2s.size(); i++) {
        fxs[i] = Xfs[2 * i].head(this->ode.XVars()) - Xfs[2 * i + 1].head(this->ode.XVars());
      }
      return fxs;
    }


    std::tuple<std::vector<Output<double>>, std::vector<Jacobian<double>>> compute_jacobian_impl_v(
        const std::vector<Input<double>>& X1X2s) const {


      std::vector<ODEState<double>> Xs;
      Eigen::VectorXd tfs;

      this->get_input_states_tfs(X1X2s, Xs, tfs);
      auto Xfs_Jfs = this->integ.integrate_stm(Xs, tfs);

      std::vector<Output<double>> fxs(X1X2s.size());
      std::vector<Jacobian<double>> jxs(X1X2s.size());


      Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(ode.ORows(),
                                                                               integ.IRows() * 2);
      Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(integ.IRows() * 2,
                                                                                this->IRows());

      XJac.setZero();

      XJac.topLeftCorner(ode.XtUVars(), ode.XtUVars()).setIdentity();
      XJac.block(ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars()).setIdentity();
      XJac(ode.IRows(), ode.TVar()) = .5;
      XJac(ode.IRows(), ode.XtUVars() + ode.TVar()) = .5;


      XJac.block(integ.IRows(), ode.XtUVars(), ode.XtUVars(), ode.XtUVars()).setIdentity();
      XJac.block(integ.IRows() + ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars()).setIdentity();

      XJac(integ.IRows() + ode.IRows(), ode.TVar()) = .5;
      XJac(integ.IRows() + ode.IRows(), ode.XtUVars() + ode.TVar()) = .5;


      for (int i = 0; i < X1X2s.size(); i++) {

        auto& [Xf1, Jf1] = Xfs_Jfs[2 * i];
        auto& [Xf2, Jf2] = Xfs_Jfs[2 * i + 1];

        Jf2 *= -1.0;

        fxs[i] = Xf1.head(ode.XVars()) - Xf2.head(ode.XVars());

        IJac.leftCols(integ.IRows()) = Jf1.topRows(ode.XVars());
        IJac.rightCols(integ.IRows()) = Jf2.topRows(ode.XVars());


        jxs[i].noalias() = IJac * XJac;
      }
      return std::tuple {fxs, jxs};
    }


    std::tuple<std::vector<Output<double>>, std::vector<Jacobian<double>>, std::vector<Hessian<double>>>
        compute_all_impl_v(const std::vector<Input<double>>& X1X2s,
                           const std::vector<Output<double>>& Ls) const {


      std::vector<ODEState<double>> Xs;
      Eigen::VectorXd tfs;
      std::vector<ODEState<double>> Lfs;

      this->get_input_states_tfs(X1X2s, Xs, tfs);
      this->get_lmults(Ls, Lfs);


      auto Xfs_Jfs_Hfs = this->integ.integrate_stm2(Xs, tfs, Lfs);

      std::vector<Output<double>> fxs(X1X2s.size());
      std::vector<Jacobian<double>> jxs(X1X2s.size());
      std::vector<Hessian<double>> hxs(X1X2s.size());


      Eigen::Matrix<double, DODE::XV, SZ_PROD<Integrator::IRC, 2>::value> IJac(ode.ORows(),
                                                                               integ.IRows() * 2);
      IJac.setZero();

      Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, SZ_PROD<Integrator::IRC, 2>::value> IHess(
          integ.IRows() * 2, integ.IRows() * 2);

      IHess.setZero();

      Eigen::Matrix<double, SZ_PROD<Integrator::IRC, 2>::value, Base::IRC> XJac(integ.IRows() * 2,
                                                                                this->IRows());
      XJac.setZero();

      XJac.topLeftCorner(ode.XtUVars(), ode.XtUVars()).setIdentity();
      XJac.block(ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars()).setIdentity();
      XJac(ode.IRows(), ode.TVar()) = .5;
      XJac(ode.IRows(), ode.XtUVars() + ode.TVar()) = .5;


      XJac.block(integ.IRows(), ode.XtUVars(), ode.XtUVars(), ode.XtUVars()).setIdentity();
      XJac.block(integ.IRows() + ode.XtUVars(), 2 * ode.XtUVars(), ode.PVars(), ode.PVars()).setIdentity();

      XJac(integ.IRows() + ode.IRows(), ode.TVar()) = .5;
      XJac(integ.IRows() + ode.IRows(), ode.XtUVars() + ode.TVar()) = .5;


      for (int i = 0; i < X1X2s.size(); i++) {

        auto& [Xf1, Jf1, Hf1] = Xfs_Jfs_Hfs[2 * i];
        auto& [Xf2, Jf2, Hf2] = Xfs_Jfs_Hfs[2 * i + 1];

        Jf2 *= -1.0;
        Hf2 *= -1.0;

        fxs[i] = Xf1.head(this->ode.XVars()) - Xf2.head(this->ode.XVars());

        IJac.leftCols(integ.IRows()) = Jf1.topRows(ode.XVars());
        IJac.rightCols(integ.IRows()) = Jf2.topRows(ode.XVars());

        IHess.topLeftCorner(integ.IRows(), integ.IRows()) = Hf1;
        IHess.bottomRightCorner(integ.IRows(), integ.IRows()) = Hf2;


        jxs[i].noalias() = IJac * XJac;
        hxs[i].noalias() = XJac.transpose() * IHess * XJac;
      }
      return std::tuple {fxs, jxs, hxs};
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();


      std::vector<Input<double>> X1X2s;

      if constexpr (!Is_SuperScalar<Scalar>::value) {
        X1X2s.push_back(x);
      } else {
        this->extract_scalar_inputs(x, X1X2s);
      }

      auto fxs = this->compute_impl_v(X1X2s);

      if constexpr (!Is_SuperScalar<Scalar>::value) {
        fx = fxs.front();
      } else {
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
          for (int i = 0; i < this->ORows(); i++) {
            fx[i][v] = fxs[v][i];
          }
        }
      }
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      std::vector<Input<double>> X1X2s;

      if constexpr (!Is_SuperScalar<Scalar>::value) {
        X1X2s.push_back(x);
      } else {
        this->extract_scalar_inputs(x, X1X2s);
      }

      auto [fxs, jxs] = this->compute_jacobian_impl_v(X1X2s);

      if constexpr (!Is_SuperScalar<Scalar>::value) {
        fx = fxs.front();
        jx = jxs.front();

      } else {
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {
          for (int i = 0; i < this->ORows(); i++) {
            fx[i][v] = fxs[v][i];
          }

          for (int j = 0; j < this->IRows(); j++) {
            for (int i = 0; i < this->ORows(); i++) {
              jx(i, j)[v] = jxs[v](i, j);
            }
          }
        }
      }
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

      std::vector<Input<double>> X1X2s;
      std::vector<Output<double>> Lfs;

      if constexpr (!Is_SuperScalar<Scalar>::value) {
        X1X2s.push_back(x);
        Lfs.push_back(adjvars);
      } else {
        this->extract_scalar_inputs(x, X1X2s);
        this->extract_scalar_lmults(adjvars, Lfs);
      }

      auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);

      if constexpr (!Is_SuperScalar<Scalar>::value) {
        fx = fxs.front();
        jx = jxs.front();
        adjhess = hxs.front();
      } else {
        for (int v = 0; v < Scalar::SizeAtCompileTime; v++) {

          for (int i = 0; i < this->ORows(); i++) {
            fx[i][v] = fxs[v][i];
          }

          for (int j = 0; j < this->IRows(); j++) {
            for (int i = 0; i < this->ORows(); i++) {
              jx(i, j)[v] = jxs[v](i, j);
            }
          }

          for (int j = 0; j < this->IRows(); j++) {
            for (int i = 0; i < this->IRows(); i++) {
              adjhess(i, j)[v] = hxs[v](i, j);
            }
          }
        }
      }

      adjgrad = jx.transpose() * adjvars;
    }


    void constraints_jacobian_adjointgradient_adjointhessian_test(
        ConstEigenRef<Eigen::VectorXd> X,
        ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX,
        EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations,
        EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {

      Input<double> x(this->IRows());
      Output<double> l(this->ORows());

      Eigen::Map<Output<double>> fx(NULL, this->ORows());
      Eigen::Map<Input<double>> agx(NULL, this->IRows());


      std::vector<Input<double>> X1X2s;
      std::vector<Output<double>> Lfs;


      for (int V = 0; V < data.NumAppl(); V++) {
        this->gatherInput(X, x, V, data);
        this->gatherMult(L, l, V, data);

        X1X2s.push_back(x);
        Lfs.push_back(l);
      }

      auto [fxs, jxs, hxs] = this->compute_all_impl_v(X1X2s, Lfs);


      for (int V = 0; V < data.NumAppl(); V++) {

        new (&fx) Eigen::Map<Output<double>>(FX.data() + data.InnerConstraintStarts[V], this->ORows());
        new (&agx) Eigen::Map<Input<double>>(AGX.data() + data.InnerGradientStarts[V], this->IRows());

        fx = fxs[V];
        agx = jxs[V].transpose() * Lfs[V];
        this->derived().KKTFillAll(V, jxs[V], hxs[V], KKTmat, KKTLocations, KKTClashes, KKTLocks, data);
      }
    }
  };


}  // namespace ASSET
