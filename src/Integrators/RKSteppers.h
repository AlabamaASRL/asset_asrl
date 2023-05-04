#pragma once


#include "RKCoeffs.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {


  template<class DODE, RKOptions RKOp>
  struct RKStepper : VectorFunction<RKStepper<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC> {
    using Base = VectorFunction<RKStepper<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class Scalar>
    using ODEDeriv = typename DODE::template Output<Scalar>;
    template<class Scalar>
    using ODEState = typename DODE::template Input<Scalar>;
    template<class Scalar>
    using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    template<class Scalar>
    using ODEHessian = typename DODE::template Hessian<Scalar>;

    static const bool IsVectorizable = true;

    using RKData = RKCoeffs<RKOp>;
    static const int Stages = RKData::Stages;
    static const int Stgsm1 = RKData::Stages - 1;
    static const bool isDiag = RKData::isDiag;

    DODE ode;


    RKStepper() {
    }
    RKStepper(DODE ode) : ode(ode) {
      this->setIORows(this->ode.IRows() + 1, this->ode.IRows());
    }

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      auto Impl = [&](auto& Kvals, auto& xtup) {
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        Scalar t0 = xtup[this->ode.TVar()];
        Scalar tf = x[this->ode.IRows()];
        Scalar h = tf - t0;


        this->ode.compute(xtup, Kvals[0]);
        Kvals[0] *= h;


        for (int i = 0; i < Stgsm1; i++) {
          Scalar ti = t0 + RKData::Times[i] * h;
          xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
          xtup[this->ode.TVar()] = ti;
          const int ip1 = i + 1;
          const int js = isDiag ? i : 0;
          for (int j = js; j < ip1; j++) {
            xtup.template segment<DODE::XV>(0, this->ode.XVars()) += Scalar(RKData::ACoeffs[i][j]) * Kvals[j];
          }

          this->ode.compute(xtup, Kvals[ip1]);

          Kvals[ip1] *= h;
        }
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        xtup[this->ode.TVar()] = tf;
        for (int i = 0; i < Stages; i++) {
          xtup.template segment<DODE::XV>(0, this->ode.XVars()) += Scalar(RKData::BCoeffs[i]) * Kvals[i];
        }
        fx = xtup;  // Next State
      };

      MemoryManager::allocate_run(this->ode.IRows(),
                                  Impl,
                                  ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode.ORows(), 1),
                                  TempSpec<ODEState<Scalar>>(this->ode.IRows(), 1));
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      auto Impl = [&](auto& Kvals, auto& xtup, auto& Kjac, auto& Xijac, auto& KXjacs) {
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        Scalar t0 = xtup[this->ode.TVar()];
        Scalar tf = x[this->ode.IRows()];
        Scalar h = tf - t0;

        Xijac.setIdentity();
        Xijac(this->ode.TVar(), this->IRows() - 1) = 0;

        this->ode.compute_jacobian(xtup, Kvals[0], Kjac);

        Kjac *= h;
        KXjacs[0].noalias() = Kjac * Xijac;
        KXjacs[0].col(this->ode.TVar()).template segment<DODE::XV>(0, this->ode.XVars()) -= Kvals[0];
        KXjacs[0].col(this->IRows() - 1).template segment<DODE::XV>(0, this->ode.XVars()) += Kvals[0];

        Kvals[0] *= h;

        for (int i = 0; i < Stgsm1; i++) {
          Scalar ti = t0 + RKData::Times[i] * h;
          xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
          Xijac.setIdentity();


          xtup[this->ode.TVar()] = ti;

          Xijac(this->ode.TVar(), this->ode.TVar()) = Scalar(1.0) - Scalar(RKData::Times[i]);
          Xijac(this->ode.TVar(), this->IRows() - 1) = Scalar(RKData::Times[i]);

          const int ip1 = i + 1;
          const int js = isDiag ? i : 0;
          for (int j = js; j < ip1; j++) {
            xtup.template segment<DODE::XV>(0, this->ode.XVars()) += Scalar(RKData::ACoeffs[i][j]) * Kvals[j];
            Xijac.template topRows<DODE::XV>(this->ode.XVars()) += Scalar(RKData::ACoeffs[i][j]) * KXjacs[j];
          }
          Kjac.setZero();

          this->ode.compute_jacobian(xtup, Kvals[ip1], Kjac);

          KXjacs[ip1].noalias() = h * Kjac * Xijac;
          KXjacs[ip1].col(this->ode.TVar()).template head<DODE::XV>(this->ode.XVars()) -= Kvals[ip1];
          KXjacs[ip1].col(this->IRows() - 1).template head<DODE::XV>(this->ode.XVars()) += Kvals[ip1];

          Kvals[ip1] *= h;
        }
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        xtup[this->ode.TVar()] = tf;

        Xijac.setIdentity();

        Xijac(this->ode.TVar(), this->ode.TVar()) = Scalar(0);
        Xijac(this->ode.TVar(), (this->IRows() - 1)) = Scalar(1);


        for (int i = 0; i < Stages; i++) {
          xtup.template segment<DODE::XV>(0, this->ode.XVars()) += Scalar(RKData::BCoeffs[i]) * Kvals[i];
          Xijac.template topRows<DODE::XV>(this->ode.XVars()) += Scalar(RKData::BCoeffs[i]) * KXjacs[i];
        }
        fx = xtup;  // Next State
        jx = Xijac;
      };

      using KXjacType = Eigen::Matrix<Scalar, DODE::XV, Base::IRC>;

      MemoryManager::allocate_run(this->ode.IRows(),
                                  Impl,
                                  ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode.ORows(), 1),
                                  TempSpec<ODEState<Scalar>>(this->ode.IRows(), 1),
                                  TempSpec<ODEJacobian<Scalar>>(this->ode.ORows(), this->ode.IRows()),
                                  TempSpec<Jacobian<Scalar>>(this->ORows(), this->IRows()),
                                  ArrayOfTempSpecs<KXjacType, Stages>(this->ode.ORows(), this->IRows()));
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


      auto Impl = [&](auto& Kvals,
                      auto& xtup,
                      auto& Kjacs,
                      auto& Xijac,
                      auto& KXjacs,
                      auto& Xs,
                      auto& Kgrads,
                      auto& Khesses,
                      auto& KXmults,
                      auto& HTpar) {
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());


        Scalar t0 = xtup[this->ode.TVar()];
        Scalar tf = x[this->ode.IRows()];
        Scalar h = tf - t0;

        Xs[0] = xtup;
        this->ode.compute(xtup, Kvals[0]);
        Kvals[0] *= h;


        for (int i = 0; i < Stgsm1; i++) {
          Scalar ti = t0 + RKData::Times[i] * h;
          xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
          xtup[this->ode.TVar()] = ti;
          const int ip1 = i + 1;
          const int js = isDiag ? i : 0;
          for (int j = js; j < ip1; j++) {
            xtup.template segment<DODE::XV>(0, this->ode.XVars()) += Scalar(RKData::ACoeffs[i][j]) * Kvals[j];
          }
          Xs[ip1] = xtup;
          this->ode.compute(xtup, Kvals[ip1]);
          Kvals[ip1] *= h;
        }


        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        xtup[this->ode.TVar()] = tf;
        for (int i = 0; i < Stages; i++) {
          xtup.template segment<DODE::XV>(0, this->ode.XVars()) += Scalar(RKData::BCoeffs[i]) * Kvals[i];
          KXmults[i] = adjvars * Scalar(RKData::BCoeffs[i]) * h;
        }

        fx = xtup;  // Next State

        for (int i = Stgsm1 - 1; i >= 0; i--) {

          const int ip1 = i + 1;
          const int js = isDiag ? i : 0;
          Kvals[ip1].setZero();
          this->ode.compute_jacobian_adjointgradient_adjointhessian(Xs[ip1],
                                                                    Kvals[ip1],
                                                                    Kjacs[ip1],
                                                                    Kgrads[ip1],
                                                                    Khesses[ip1],
                                                                    KXmults[ip1].head(this->ode.ORows()));

          for (int j = js; j < ip1; j++) {
            KXmults[j] += Kgrads[ip1] * Scalar(RKData::ACoeffs[i][j]) * h;
          }
        }

        Kvals[0].setZero();
        this->ode.compute_jacobian_adjointgradient_adjointhessian(
            Xs[0], Kvals[0], Kjacs[0], Kgrads[0], Khesses[0], KXmults[0].head(this->ode.ORows()));


        adjhess.topLeftCorner(this->ode.IRows(), this->ode.IRows()) += Khesses[0];

        Xijac.setIdentity();
        Xijac(this->ode.TVar(), this->IRows() - 1) = Scalar(0.0);


        Kjacs[0] *= h;
        KXjacs[0].noalias() = Kjacs[0] * Xijac;
        KXjacs[0].col(this->ode.TVar()).template segment<DODE::XV>(0, this->ode.XVars()) -= Kvals[0];
        KXjacs[0].col(this->IRows() - 1).template segment<DODE::XV>(0, this->ode.XVars()) += Kvals[0];

        Kvals[0] *= h;

        HTpar = (Xijac.transpose() * Kgrads[0]) * (1.0 / h);


        //

        for (int i = 0; i < Stgsm1; i++) {
          Scalar ti = t0 + RKData::Times[i] * h;
          Xijac.setIdentity();


          Xijac(this->ode.TVar(), this->ode.TVar()) = Scalar(1.0) - Scalar(RKData::Times[i]);
          Xijac(this->ode.TVar(), this->IRows() - 1) = Scalar(RKData::Times[i]);

          const int ip1 = i + 1;
          const int js = isDiag ? i : 0;
          for (int j = js; j < ip1; j++) {

            Xijac.template topRows<DODE::XV>(this->ode.XVars()) += Scalar(RKData::ACoeffs[i][j]) * KXjacs[j];
          }

          KXjacs[ip1].noalias() = h * Kjacs[ip1] * Xijac;
          KXjacs[ip1].col(this->ode.TVar()).template head<DODE::XV>(this->ode.XVars()) -= Kvals[ip1];
          KXjacs[ip1].col(this->IRows() - 1).template head<DODE::XV>(this->ode.XVars()) += Kvals[ip1];

          Kvals[ip1] *= h;


          adjhess += Xijac.transpose() * Khesses[ip1] * Xijac;

          HTpar += (Xijac.transpose() * Kgrads[ip1]) * (1.0 / h);
        }
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        xtup[this->ode.TVar()] = tf;

        Xijac.setIdentity();

        Xijac(this->ode.TVar(), this->ode.TVar()) = Scalar(0.0);
        Xijac(this->ode.TVar(), (this->IRows() - 1)) = Scalar(1.0);


        for (int i = 0; i < Stages; i++) {

          Xijac.template topRows<DODE::XV>(this->ode.XVars()) += Scalar(RKData::BCoeffs[i]) * KXjacs[i];
        }

        adjhess.col(this->ode.TVar()) -= HTpar;
        adjhess.col(this->IRows() - 1) += HTpar;

        adjhess.row(this->ode.TVar()) -= HTpar.transpose();
        adjhess.row(this->IRows() - 1) += HTpar.transpose();


        jx = Xijac;
        adjgrad = jx.transpose() * adjvars;
      };

      using KXjacType = Eigen::Matrix<Scalar, DODE::XV, Base::IRC>;

      MemoryManager::allocate_run(
          this->ode.IRows(),
          Impl,
          ArrayOfTempSpecs<ODEDeriv<Scalar>, Stages>(this->ode.ORows(), 1),
          TempSpec<ODEState<Scalar>>(this->ode.IRows(), 1),
          ArrayOfTempSpecs<ODEJacobian<Scalar>, Stages>(this->ode.ORows(), this->ode.IRows()),
          TempSpec<Jacobian<Scalar>>(this->ORows(), this->IRows()),
          ArrayOfTempSpecs<KXjacType, Stages>(this->ode.ORows(), this->IRows()),
          ArrayOfTempSpecs<ODEState<Scalar>, Stages>(this->ode.IRows(), 1),
          ArrayOfTempSpecs<ODEState<Scalar>, Stages>(this->ode.IRows(), 1),
          ArrayOfTempSpecs<ODEHessian<Scalar>, Stages>(this->ode.IRows(), this->ode.IRows()),
          ArrayOfTempSpecs<ODEState<Scalar>, Stages>(this->ode.IRows(), 1),
          TempSpec<Input<Scalar>>(this->IRows(), 1));
    }


    /// These Methods are being nulled because it is not
    /// possible for them to be called

    void constraints(ConstEigenRef<Eigen::VectorXd> X,
                     EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData& data) const {};
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L,
                                     EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData& data) const {};
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const {
    }
    void constraints_jacobian_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                              ConstEigenRef<Eigen::VectorXd> L,
                                              Eigen::Ref<Eigen::VectorXd> FX,
                                              Eigen::Ref<Eigen::VectorXd> AGX,
                                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                              EigenRef<Eigen::VectorXi> KKTLocations,
                                              EigenRef<Eigen::VectorXi> KKTClashes,
                                              std::vector<std::mutex>& KKTLocks,
                                              const SolverIndexingData& data) const {
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X,
        ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX,
        EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations,
        EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
    }
  };


  template<class DODE, RKOptions RKOp>
  struct RKStepper_Impl {
    static auto Definition(const DODE& ode) {
      auto ks0 = std::tuple {};
      return ComputeXf<-1, decltype(ks0)>(ode, ks0);
    }

    template<int Stg, int Elem>
    struct ACoeff : StaticScaleBase<ACoeff<Stg, Elem>> {
      static constexpr double value = RKCoeffs<RKOp>::ACoeffs[Stg][Elem];
    };
    template<int Elem>
    struct BCoeff : StaticScaleBase<BCoeff<Elem>> {
      static constexpr double value = RKCoeffs<RKOp>::BCoeffs[Elem];
    };

    template<int Stg, class Ks>
    static auto ComputeXf(const DODE& ode, const Ks& ks) {
      constexpr int XV = DODE::XV;
      constexpr int UV = DODE::UV;
      constexpr int PV = DODE::PV;
      constexpr int IRC = SZ_SUM<DODE::IRC, 1>::value;

      auto args = Arguments<DODE::IRC + 1 + (Stg + 1) * DODE::XV>();
      auto empty = std::tuple {};

      // auto xi = KthStageSum<Stg, 0, decltype(args)>(ode, args);
      auto xi = KthStageSum2<Stg, 0, decltype(args), decltype(empty)>(ode, args, empty);

      auto t0 = args.template coeff<XV>();
      auto tf = args.template coeff<IRC - 1>();
      auto u = args.template segment<UV, XV + 1>();
      auto p = args.template segment<PV, XV + 1 + UV>();
      auto h = tf - t0;

      auto tlam = [&]() {
        if constexpr (Stg == -1)
          return t0;
        else
          return t0 + h * RKCoeffs<RKOp>::Times[Stg];
      };
      auto ti = tlam();
      auto xti = make_state(ode, xi, ti, u, p);

      auto ki = ode.eval(xti) * h;

      if constexpr (Stg == RKCoeffs<RKOp>::Stages - 2) {
        auto xxf = FinalStateSum<0, decltype(args), decltype(ki)>(ode, args, ki);
        auto xfun = make_state(ode, xxf, tf, u, p);
        return NestedCallAndAppendChain {xfun, ks};
      } else {
        auto knew = std::tuple_cat(ks, std::make_tuple(ki));

        return ComputeXf<Stg + 1, decltype(knew)>(ode, knew);
      }
    }

    template<class Xtype, class Titype, class Utype, class Ptype>
    static auto make_state(
        const DODE& ode, const Xtype& x0, const Titype& ti, const Utype& u, const Ptype& p) {
      if constexpr (DODE::UV > 0) {
        if constexpr (DODE::PV > 0) {
        } else if constexpr (DODE::PV == 0) {
          return StackedOutputs {x0, ti, u};
        }
      } else if constexpr (DODE::UV == 0) {
        if constexpr (DODE::PV > 0) {
        } else if constexpr (DODE::PV == 0) {
          return StackedOutputs {x0, ti};
        }
      } else if constexpr (DODE::UV == -1) {
        if constexpr (DODE::PV > 0) {
        } else if constexpr (DODE::PV == 0) {
        }
      }
    }

    template<int Stg, int Elem, class Args>
    static auto KthStageSum(const DODE& ode, const Args& args) {
      if constexpr (Elem == Stg + 1) {
        return args.template head<DODE::XV>();
      } else {
        if constexpr (RKCoeffs<RKOp>::ACoeffs[Stg][Elem] == 0.0) {
          return KthStageSum<Stg, Elem + 1, Args>(ode, args);
        } else {
          return args.template tail<DODE::XV*(Stg + 1)>().template segment<DODE::XV, DODE::XV * Elem>()
                     * ACoeff<Stg, Elem>().value
                 + KthStageSum<Stg, Elem + 1, Args>(ode, args);
        }
      };
    }

    template<int Stg, int Elem, class Args, class Ks>
    static auto KthStageSum2(const DODE& ode, const Args& args, const Ks& ks) {
      if constexpr (Elem == Stg + 1) {
        auto next = args.template head<DODE::XV>();
        auto knew = std::tuple_cat(ks, std::make_tuple(next));

        return make_sum_tuple(knew);
      } else {
        if constexpr (RKCoeffs<RKOp>::ACoeffs[Stg][Elem] == 0.0) {
          return KthStageSum2<Stg, Elem + 1, Args, Ks>(ode, args, ks);
        } else {
          auto next = args.template tail<DODE::XV*(Stg + 1)>().template segment<DODE::XV, DODE::XV * Elem>()
                      * ACoeff<Stg, Elem>().value;

          auto knew = std::tuple_cat(ks, std::make_tuple(next));

          return KthStageSum2<Stg, Elem + 1, Args, decltype(knew)>(ode, args, knew);
        }
      };
    }

    template<int Elem, class Args, class KF>
    static auto FinalStateSum(const DODE& ode, const Args& args, const KF& kf) {
      if constexpr (RKOp == RK4Classic) {
        return make_sum(kf * BCoeff<3>().value,
                        args.template head<DODE::XV>(),
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 0>()
                            * BCoeff<0>().value,
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 1>()
                            * BCoeff<1>().value,
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 2>()
                            * BCoeff<2>().value);
      } else if constexpr (RKOp == DOPRI5) {
        return make_sum(kf * BCoeff<5>(),
                        args.template head<DODE::XV>(),
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 0>()
                            * BCoeff<0>(),
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 2>()
                            * BCoeff<2>(),
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 3>()
                            * BCoeff<3>(),
                        args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                                .template segment<DODE::XV, DODE::XV * 4>()
                            * BCoeff<4>());
      } else if constexpr (Elem == RKCoeffs<RKOp>::Stages - 1) {
        if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
          return args.template head<DODE::XV>();
        else
          return kf * BCoeff<Elem>() + args.template head<DODE::XV>();
      } else {
        if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
          return FinalStateSum<Elem + 1, Args, KF>(ode, args, kf);
        else
          return args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                         .template segment<DODE::XV, DODE::XV * Elem>()
                     * BCoeff<Elem>()
                 + FinalStateSum<Elem + 1, Args, KF>(ode, args, kf);
      };
    }
  };


  template<class DODE, RKOptions RKOp>
  struct RKStepper_Impl_NEW {

    template<int Stg, int Elem>
    struct ACoeff : StaticScaleBase<ACoeff<Stg, Elem>> {
      static constexpr double value = RKCoeffs<RKOp>::ACoeffs[Stg][Elem];
    };
    template<int Elem>
    struct BCoeff : StaticScaleBase<BCoeff<Elem>> {
      static constexpr double value = RKCoeffs<RKOp>::BCoeffs[Elem];
    };


    static auto Definition(const DODE& ode) {
      auto ks0 = std::tuple {};
      return ComputeXf<-1, decltype(ks0)>(ode, ks0);
    }


    template<int Stg, class Ks>
    static auto ComputeXf(const DODE& ode, const Ks& ks) {
      constexpr int XV = DODE::XV;
      constexpr int UV = DODE::UV;
      constexpr int PV = DODE::PV;
      constexpr int IRC = SZ_SUM<DODE::IRC, 1>::value;
      constexpr int ARGSIZE = SZ_SUM<IRC, SZ_PROD<(Stg + 1), XV>::value>::value;

      int xv = ode.XVars();
      int uv = ode.UVars();
      int pv = ode.PVars();
      int irows = ode.IRows() + 1;
      int argsize = irows + (Stg + 1) * xv;

      auto args = Arguments<ARGSIZE>(argsize);
      auto empty = std::tuple {};

      auto xi = KthStageSum<Stg, 0, decltype(args), decltype(empty)>(ode, args, empty);

      auto t0 = args.template coeff<XV>(xv);
      auto up = args.template segment<SZ_SUM<UV, PV>::value, SZ_SUM<XV, 1>::value>(xv + 1, uv + pv);
      auto tf = args.template coeff<SZ_DIFF<IRC, 1>::value>(irows - 1);
      auto h = tf - t0;

      auto tlam = [&]() {
        if constexpr (Stg == -1)
          return t0;
        else
          return t0 + h * RKCoeffs<RKOp>::Times[Stg];
      };
      auto make_state = [&](const auto& xii, const auto& tii, const auto& upii) {
        if constexpr (SZ_SUM<DODE::UV, DODE::PV>::value == 0)
          return StackedOutputs {xii, tii};
        else
          return StackedOutputs {xii, tii, upii};
      };

      auto ti = tlam();
      auto xti = make_state(xi, ti, up);
      auto ki = ode.eval(xti) * h;

      if constexpr (Stg == RKCoeffs<RKOp>::Stages - 2) {
        auto xxf = FinalStateSum<0, decltype(args), decltype(ki)>(ode, args, ki);
        auto xfun = make_state(xxf, tf, up);
        return NestedCallAndAppendChain2 {xfun, ks};
      } else {
        auto knew = std::tuple_cat(ks, std::make_tuple(ki));
        return ComputeXf<Stg + 1, decltype(knew)>(ode, knew);
      }
    }


    template<int Stg, int Elem, class Args, class Ks>
    static auto KthStageSum(const DODE& ode, const Args& args, const Ks& ks) {
      if constexpr (Elem == Stg + 1) {
        auto next = args.template head<DODE::XV>(ode.XVars());
        auto knew = std::tuple_cat(ks, std::make_tuple(next));
        return make_sum_tuple(knew);
      } else {
        if constexpr (RKCoeffs<RKOp>::ACoeffs[Stg][Elem] == 0.0) {
          return KthStageSum<Stg, Elem + 1, Args, Ks>(ode, args, ks);
        } else {
          auto next =
              args.template tail<SZ_PROD<(Stg + 1), DODE::XV>::value>((Stg + 1) * ode.XVars())
                  .template segment<DODE::XV, SZ_PROD<Elem, DODE::XV>::value>(ode.XVars() * Elem, ode.XVars())
              * ACoeff<Stg, Elem>().value;
          auto knew = std::tuple_cat(ks, std::make_tuple(next));
          return KthStageSum<Stg, Elem + 1, Args, decltype(knew)>(ode, args, knew);
        }
      };
    }

    template<int Elem, class Args, class KF>
    static auto FinalStateSum(const DODE& ode, const Args& args, const KF& kf) {
      //// Finish this
      if constexpr (RKOp == RK4Classic) {

        // constexpr int XV       = DODE::XV;
        constexpr int TAILSIZE = SZ_PROD<DODE::XV, (RKCoeffs<RKOp>::Stages - 1)>::value;
        int xv = ode.XVars();
        int tailsize = xv * (RKCoeffs<RKOp>::Stages - 1);
        return make_sum(
            kf * BCoeff<3>().value,
            args.template head<DODE::XV>(xv),

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 0>::value>(0,
                                                                                                           xv)
                * BCoeff<0>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 1>::value>(xv,
                                                                                                           xv)
                * BCoeff<1>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 2>::value>(
                2 * xv, xv)
                * BCoeff<2>().value);
      } else if constexpr (RKOp == DOPRI5) {

        // constexpr int XV       = DODE::XV;
        constexpr int TAILSIZE = SZ_PROD<DODE::XV, (RKCoeffs<RKOp>::Stages - 1)>::value;
        int xv = ode.XVars();
        int tailsize = xv * (RKCoeffs<RKOp>::Stages - 1);
        return make_sum(
            kf * BCoeff<5>().value,
            args.template head<DODE::XV>(xv),

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 0>::value>(0,
                                                                                                           xv)
                * BCoeff<0>().value,

            // args.template tail<TAILSIZE>(tailsize)
            // .template segment<DODE::XV, SZ_PROD<DODE::XV, 1>::value>(xv, xv) *
            // BCoeff<1>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 2>::value>(
                2 * xv, xv)
                * BCoeff<2>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 3>::value>(
                3 * xv, xv)
                * BCoeff<3>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 4>::value>(
                4 * xv, xv)
                * BCoeff<4>().value);
      } else if constexpr (RKOp == DOPRI87) {

        // constexpr int XV       = DODE::XV;
        constexpr int TAILSIZE = SZ_PROD<DODE::XV, (RKCoeffs<RKOp>::Stages - 1)>::value;
        int xv = ode.XVars();
        int tailsize = xv * (RKCoeffs<RKOp>::Stages - 1);
        return make_sum(
            kf * BCoeff<12>().value,
            args.template head<DODE::XV>(xv),

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 0>::value>(0,
                                                                                                           xv)
                * BCoeff<0>().value,

            // args.template tail<TAILSIZE>(tailsize)
            // .template segment<DODE::XV, SZ_PROD<DODE::XV, 1>::value>(xv, xv) *
            // BCoeff<1>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 5>::value>(
                5 * xv, xv)
                * BCoeff<5>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 6>::value>(
                6 * xv, xv)
                * BCoeff<6>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 7>::value>(
                7 * xv, xv)
                * BCoeff<7>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 8>::value>(
                8 * xv, xv)
                * BCoeff<8>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 9>::value>(
                9 * xv, xv)
                * BCoeff<9>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 10>::value>(
                10 * xv, xv)
                * BCoeff<10>().value,

            args.template tail<TAILSIZE>(tailsize).template segment<DODE::XV, SZ_PROD<DODE::XV, 11>::value>(
                11 * xv, xv)
                * BCoeff<11>().value);
      }

      else if constexpr (Elem == RKCoeffs<RKOp>::Stages - 1) {
        if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
          return args.template head<DODE::XV>();
        else
          return kf * BCoeff<Elem>() + args.template head<DODE::XV>();
      } else {
        if constexpr (RKCoeffs<RKOp>::BCoeffs[Elem] == 0.0)
          return FinalStateSum<Elem + 1, Args, KF>(ode, args, kf);
        else
          return args.template tail<DODE::XV*(RKCoeffs<RKOp>::Stages - 1)>()
                         .template segment<DODE::XV, DODE::XV * Elem>()
                     * BCoeff<Elem>()
                 + FinalStateSum<Elem + 1, Args, KF>(ode, args, kf);
      };
    }
  };

  template<class DODE, RKOptions RKOp>
  struct RKStepper_NEW
      : VectorExpression<RKStepper_NEW<DODE, RKOp>, RKStepper_Impl_NEW<DODE, RKOp>, const DODE&> {
    using Base = VectorExpression<RKStepper_NEW<DODE, RKOp>, RKStepper_Impl_NEW<DODE, RKOp>, const DODE&>;
    // using Base::Base;
    // static const bool IsVectorizable = false;

    RKStepper_NEW(const DODE& ode) : Base(ode) {
    }


    /// These Methods are being nulled because it is not
    /// possible for them to be called

    void constraints(ConstEigenRef<Eigen::VectorXd> X,
                     EigenRef<Eigen::VectorXd> FX,
                     const SolverIndexingData& data) const {};
    void constraints_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                     ConstEigenRef<Eigen::VectorXd> L,
                                     EigenRef<Eigen::VectorXd> FX,
                                     EigenRef<Eigen::VectorXd> AGX,
                                     const SolverIndexingData& data) const {};
    void constraints_jacobian(ConstEigenRef<Eigen::VectorXd> X,
                              Eigen::Ref<Eigen::VectorXd> FX,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                              Eigen::Ref<Eigen::VectorXi> KKTLocations,
                              Eigen::Ref<Eigen::VectorXi> KKTClashes,
                              std::vector<std::mutex>& KKTLocks,
                              const SolverIndexingData& data) const {
    }
    void constraints_jacobian_adjointgradient(ConstEigenRef<Eigen::VectorXd> X,
                                              ConstEigenRef<Eigen::VectorXd> L,
                                              Eigen::Ref<Eigen::VectorXd> FX,
                                              Eigen::Ref<Eigen::VectorXd> AGX,
                                              Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                                              EigenRef<Eigen::VectorXi> KKTLocations,
                                              EigenRef<Eigen::VectorXi> KKTClashes,
                                              std::vector<std::mutex>& KKTLocks,
                                              const SolverIndexingData& data) const {
    }
    void constraints_jacobian_adjointgradient_adjointhessian(
        ConstEigenRef<Eigen::VectorXd> X,
        ConstEigenRef<Eigen::VectorXd> L,
        EigenRef<Eigen::VectorXd> FX,
        EigenRef<Eigen::VectorXd> AGX,
        Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
        EigenRef<Eigen::VectorXi> KKTLocations,
        EigenRef<Eigen::VectorXi> KKTClashes,
        std::vector<std::mutex>& KKTLocks,
        const SolverIndexingData& data) const {
    }
  };


  template<class DODE, RKOptions RKOp>
  struct RKStepper2
      : VectorFunction<RKStepper2<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC, FDiffFwd, FDiffFwd> {
    using Base =
        VectorFunction<RKStepper2<DODE, RKOp>, SZ_SUM<DODE::IRC, 1>::value, DODE::IRC, FDiffFwd, FDiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    template<class Scalar>
    using ODEDeriv = typename DODE::template Output<Scalar>;
    template<class Scalar>
    using ODEState = typename DODE::template Input<Scalar>;
    template<class Scalar>
    using ODEJacobian = typename DODE::template Jacobian<Scalar>;

    using RKData = RKCoeffs<RKOp>;
    static const int Stages = RKData::Stages;
    static const int Stgsm1 = RKData::Stages - 1;
    static const bool isDiag = RKData::isDiag;

    DODE ode;

    template<class T, int SZ>
    using STDarray = std::array<T, SZ>;

    RKStepper2() {
    }
    RKStepper2(DODE ode) : ode(ode) {
      this->setIORows(this->ode.IRows() + 1, this->ode.IRows());
      this->setJacFDSteps(1.0e-6);
      this->setHessFDSteps(1.0e-6);
    }

    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      STDarray<ODEDeriv<Scalar>, Stages> Kvals;
      for (auto& K: Kvals)
        K = ODEDeriv<Scalar>::Zero(this->ode.ORows());
      ODEState<Scalar> xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
      Scalar t0 = xtup[this->ode.TVar()];
      Scalar tf = x[this->IRows() - 1];
      Scalar h = tf - t0;

      this->ode.compute(xtup, Kvals[0]);
      Kvals[0] *= h;

      for (int i = 0; i < Stgsm1; i++) {
        Scalar ti = t0 + RKData::Times[i] * h;
        xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
        xtup[this->ode.TVar()] = ti;
        const int ip1 = i + 1;
        const int js = isDiag ? i : 0;
        for (int j = js; j < ip1; j++) {
          xtup.template segment<DODE::XV>(0, this->ode.XVars()) += RKData::ACoeffs[i][j] * Kvals[j];
        }

        this->ode.compute(xtup, Kvals[ip1]);

        Kvals[ip1] *= h;
      }
      xtup = x.template segment<DODE::IRC>(0, this->ode.IRows());
      xtup[this->ode.TVar()] = tf;
      for (int i = 0; i < Stages; i++) {
        xtup.template segment<DODE::XV>(0, this->ode.XVars()) += RKData::BCoeffs[i] * Kvals[i];
      }
      fx = xtup;
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl1(const Eigen::MatrixBase<InType>& x,
                                       Eigen::MatrixBase<OutType> const& fx_,
                                       Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      STDarray<ODEDeriv<Scalar>, Stages> Kvals;
      ODEJacobian<Scalar> Kjacs;
      Jacobian<Scalar> Xijac = Jacobian<Scalar>::Zero(this->ORows(), this->IRows());
      ;

      STDarray<Eigen::Matrix<Scalar, DODE::XV, Base::IRC>, Stages> KXjacs;

      for (auto& K: Kvals)
        K = ODEDeriv<Scalar>::Zero(this->ode.ORows());

      ODEState<Scalar> xtup = x.template head<DODE::IRC>(this->ode.IRows());

      Scalar t0 = xtup[this->ode.TVar()];
      Scalar tf = x[this->IRows() - 1];
      Scalar h = tf - t0;
      Xijac.setIdentity();
      Xijac(this->ode.TVar(), this->IRows() - 1) = 0;
      Kjacs.setZero();
      this->ode.compute_jacobian(xtup, Kvals[0], Kjacs);

      Kjacs *= h;
      KXjacs[0] = Kjacs * Xijac;
      KXjacs[0].col(this->ode.TVar()).template head<DODE::XV>(this->ode.XVars()) -= Kvals[0];
      KXjacs[0].col(this->IRows() - 1).template head<DODE::XV>(this->ode.XVars()) += Kvals[0];
      Kvals[0] *= h;

      for (int i = 0; i < Stgsm1; i++) {
        const int ip1 = i + 1;
        Scalar ti = t0 + RKData::Times[i] * h;
        xtup = x.template head<DODE::IRC>(this->ode.IRows());
        Xijac.setIdentity();

        xtup[this->ode.TVar()] = ti;
        Xijac(this->ode.TVar(), this->ode.TVar()) = 1.0 - RKData::Times[i];
        Xijac(this->ode.TVar(), this->IRows() - 1) = RKData::Times[i];

        const int js = isDiag ? i : 0;
        for (int j = js; j < ip1; j++) {
          xtup.template head<DODE::XV>(this->ode.XVars()) += RKData::ACoeffs[i][j] * Kvals[j];
          Xijac.template topRows<DODE::XV>(this->ode.XVars()) += RKData::ACoeffs[i][j] * KXjacs[j];
        }

        Kjacs.setZero();
        this->ode.compute_jacobian(xtup, Kvals[ip1], Kjacs);
        KXjacs[ip1].noalias() = h * Kjacs * Xijac;
        KXjacs[ip1].col(this->ode.TVar()).template head<DODE::XV>(this->ode.XVars()) -= Kvals[ip1];
        KXjacs[ip1].col(this->IRows() - 1).template head<DODE::XV>(this->ode.XVars()) += Kvals[ip1];
        Kvals[ip1] *= h;
      }
      xtup = x.template head<DODE::IRC>(this->ode.IRows());
      xtup[this->ode.TVar()] = tf;
      Xijac.setIdentity();

      Xijac(this->ode.TVar(), this->ode.TVar()) = 0;
      Xijac(this->ode.TVar(), (this->IRows() - 1)) = 1;

      for (int i = 0; i < Stages; i++) {
        xtup.template head<DODE::XV>(this->ode.XVars()) += RKData::BCoeffs[i] * Kvals[i];
        Xijac.template topRows<DODE::XV>(this->ode.XVars()) += RKData::BCoeffs[i] * KXjacs[i];
      }

      fx = xtup;
      jx = Xijac;
    }
  };

}  // namespace ASSET
