#pragma once

#include "TranscriptionSizing.h"
#include "VectorFunctions/VectorFunction.h"

namespace ASSET {


  template<class DODE>
  struct TrapezoidalDefects : VectorFunction<TrapezoidalDefects<DODE>,
                                             DefectConstSizes<2, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                             DefectConstSizes<2, DODE::XV, DODE::UV, DODE::PV>::DefORC> {
    static const int CS = 2;
    using Base = VectorFunction<TrapezoidalDefects<DODE>,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefIRC,
                                DefectConstSizes<CS, DODE::XV, DODE::UV, DODE::PV>::DefORC>;

    /////////////////////////////////////////////////////////////////////////////////
    template<class Scalar>
    using Output = typename Base::template Output<Scalar>;
    template<class Scalar>
    using Input = typename Base::template Input<Scalar>;
    template<class Scalar>
    using Jacobian = typename Base::template Jacobian<Scalar>;
    template<class Scalar>
    using Hessian = typename Base::template Hessian<Scalar>;
    ///////////////////////////////////////////////////////////////////////////////////
    template<class Scalar>
    using ODEOutput = typename DODE::template Output<Scalar>;
    template<class Scalar>
    using ODEInput = typename DODE::template Input<Scalar>;
    template<class Scalar>
    using ODEGrad = typename DODE::template Gradient<Scalar>;
    template<class Scalar>
    using ODEJacobian = typename DODE::template Jacobian<Scalar>;
    template<class Scalar>
    using ODEHessian = typename DODE::template Hessian<Scalar>;
    DODE ode;
    bool EnableHessianSparsity = false;
    Eigen::MatrixXi nzlocs;
    static const bool IsVectorizable = DODE::IsVectorizable;

    void exactHessianSparsity(Eigen::VectorXd xtup1, Eigen::VectorXd xtup2) {

      Input<double> xin(this->IRows());
      xin.head(this->ode.XtUVars()) = xtup1.head(this->ode.XtUVars());
      xin.segment(this->ode.XtUVars(), this->ode.XtUVars()) = xtup2.head(this->ode.XtUVars());
      xin.tail(this->ode.PVars()) = xtup2.tail(this->ode.PVars());

      Eigen::VectorXd ran(this->IRows());
      ran.setRandom();
      ran *= 1.0e-10;

      xin += ran;

      Output<double> lm(this->ORows());
      lm.setRandom();

      Hessian<double> hess = this->adjointhessian(xin, lm);

      for (int i = 0; i < this->IRows(); i++) {
        for (int j = 0; j < this->IRows(); j++) {
          if (nzlocs(i, j) == 1) {
            if (abs(hess(i, j)) == 0.0) {
              nzlocs(i, j) = 0;
            }
          }
        }
      }
    }


    TrapezoidalDefects(const DODE& od) {
      this->setODE(od);
    }
    void setODE(const DODE& od) {
      this->ode = od;
      this->setOutputRows(this->ode.ORows() * (CS - 1));
      this->setInputRows(CS * this->ode.XtUVars() + this->ode.PVars());

      nzlocs.resize(this->IRows(), this->IRows());
      nzlocs.setZero();

      int xtu = this->ode.XtUVars();
      nzlocs.topLeftCorner(xtu, xtu).setOnes();
      nzlocs.block(xtu, xtu, xtu, xtu).setOnes();
      nzlocs.bottomRightCorner(this->ode.PVars(), this->ode.PVars()).setOnes();

      int j = 0;
      int Cardinals = 2;
      nzlocs
          .block(j * this->ode.XtUVars(),
                 Cardinals * this->ode.XtUVars(),
                 this->ode.XtUVars(),
                 this->ode.PVars())
          .setOnes();

      nzlocs
          .block(Cardinals * this->ode.XtUVars(),
                 j * this->ode.XtUVars(),
                 this->ode.PVars(),
                 this->ode.XtUVars())
          .setOnes();
      j = 1;
      nzlocs
          .block(j * this->ode.XtUVars(),
                 Cardinals * this->ode.XtUVars(),
                 this->ode.XtUVars(),
                 this->ode.PVars())
          .setOnes();

      nzlocs
          .block(Cardinals * this->ode.XtUVars(),
                 j * this->ode.XtUVars(),
                 this->ode.PVars(),
                 this->ode.XtUVars())
          .setOnes();

      nzlocs.col(this->ode.TVar()).setOnes();
      nzlocs.col(this->ode.TVar() + this->ode.XtUVars() * (Cardinals - 1)).setOnes();
      nzlocs.row(this->ode.TVar()).setOnes();
      nzlocs.row(this->ode.TVar() + this->ode.XtUVars() * (Cardinals - 1)).setOnes();
    }


    inline bool HessianElemIsNonZero(int row, int col) const {
      if (this->EnableHessianSparsity) {
        return bool(this->nzlocs(row, col));
      } else {
        return true;
      }
    }
    inline void AddHessianElem(double v, int row, int col, double* mpt, const int* lpt, int& freeloc) const {
      if (this->EnableHessianSparsity) {
        if (bool(this->nzlocs(row, col))) {
          mpt[lpt[freeloc]] += v;
          freeloc++;
        }
      } else {
        mpt[lpt[freeloc]] += v;
        freeloc++;
      }
    }


    template<class InType, class OutType>
    inline void compute_impl(const Eigen::MatrixBase<InType>& x,
                             Eigen::MatrixBase<OutType> const& fx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();


      auto Impl = [&](auto& X0, auto& X1, auto& FX0, auto& FX1) {
        X0.template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
            x.template segment<DODE::XtUV>(0, this->ode.XtUVars());
        X0.tail(this->ode.PVars()) = x.tail(this->ode.PVars());

        X1.template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
            x.template segment<DODE::XtUV>(this->ode.XtUVars(), this->ode.XtUVars());
        X1.tail(this->ode.PVars()) = x.tail(this->ode.PVars());
        Scalar h = X1[this->ode.TVar()] - X0[this->ode.TVar()];

        this->ode.compute(X0, FX0);
        this->ode.compute(X1, FX1);

        fx = (X1.template segment<DODE::XV>(0, this->ode.XVars())
              - X0.template segment<DODE::XV>(0, this->ode.XVars()))
             - (h / 2.0) * (FX0 + FX1);
      };
      const int irows = this->ode.IRows();
      const int orows = this->ode.ORows();

      using IType = ODEInput<Scalar>;
      using OType = ODEOutput<Scalar>;


      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<IType>(irows, 1),
                                  TempSpec<IType>(irows, 1),
                                  TempSpec<OType>(orows, 1),
                                  TempSpec<OType>(orows, 1));
      fx *= Scalar(-1.0);
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(const Eigen::MatrixBase<InType>& x,
                                      Eigen::MatrixBase<OutType> const& fx_,
                                      Eigen::MatrixBase<JacType> const& jx_) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();

      Scalar One_Half(0.5);
      Scalar One(1.0);
      auto Impl = [&](auto& X0, auto& X1, auto& FX0, auto& FX1, auto& JX0, auto& JX1) {
        X0.template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
            x.template segment<DODE::XtUV>(0, this->ode.XtUVars());
        X0.tail(this->ode.PVars()) = x.tail(this->ode.PVars());

        X1.template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
            x.template segment<DODE::XtUV>(this->ode.XtUVars(), this->ode.XtUVars());
        X1.tail(this->ode.PVars()) = x.tail(this->ode.PVars());
        Scalar h = X1[this->ode.TVar()] - X0[this->ode.TVar()];

        this->ode.compute_jacobian(X0, FX0, JX0);
        this->ode.compute_jacobian(X1, FX1, JX1);

        fx = (X1.template segment<DODE::XV>(0, this->ode.XVars())
              - X0.template segment<DODE::XV>(0, this->ode.XVars()))
             - (h / 2.0) * (FX0 + FX1);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        jx.template block<DODE::XV, DODE::XV>(0, 0, this->ode.XVars(), this->ode.XVars())
            .diagonal()
            .setConstant(-One);
        jx.template block<DODE::XV, DODE::XV>(0, this->ode.XtUVars(), this->ode.XVars(), this->ode.XVars())
            .diagonal()
            .setConstant(One);

        ODEOutput<Scalar> Tds = -One_Half * (FX0 + FX1);
        jx.col(this->ode.TVar()) -= Tds;
        jx.col(this->ode.XtUVars() + this->ode.TVar()) += Tds;

        jx.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode.XVars(), this->ode.XtUVars()) +=
            (-h / 2.0)
            * JX0.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode.XVars(), this->ode.XtUVars());

        jx.template block<DODE::XV, DODE::XtUV>(
            0, this->ode.XtUVars(), this->ode.XVars(), this->ode.XtUVars()) +=
            (-h / 2.0)
            * JX1.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode.XVars(), this->ode.XtUVars());

        jx.template rightCols<DODE::PV>(this->ode.PVars()) =
            (-h / 2.0)
            * (JX0.template rightCols<DODE::PV>(this->ode.PVars())
               + JX1.template rightCols<DODE::PV>(this->ode.PVars()));

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      };

      const int irows = this->ode.IRows();
      const int orows = this->ode.ORows();

      using IType = ODEInput<Scalar>;
      using OType = ODEOutput<Scalar>;
      using JType = ODEJacobian<Scalar>;

      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<IType>(irows, 1),
                                  TempSpec<IType>(irows, 1),
                                  TempSpec<OType>(orows, 1),
                                  TempSpec<OType>(orows, 1),
                                  TempSpec<JType>(orows, irows),
                                  TempSpec<JType>(orows, irows));

      fx *= Scalar(-1.0);
      jx *= Scalar(-1.0);
    }


    template<class InType,
             class OutType,
             class JacType,
             class AdjGradType,
             class AdjHessType,
             class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        const Eigen::MatrixBase<InType>& x,
        Eigen::MatrixBase<OutType> const& fx_,
        Eigen::MatrixBase<JacType> const& jx_,
        Eigen::MatrixBase<AdjGradType> const& adjgrad_,
        Eigen::MatrixBase<AdjHessType> const& adjhess_,
        const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      typedef typename InType::Scalar Scalar;
      Eigen::MatrixBase<JacType>& jx = jx_.const_cast_derived();
      Eigen::MatrixBase<OutType>& fx = fx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      Eigen::MatrixBase<AdjHessType>& adjhess = adjhess_.const_cast_derived();


      Scalar One_Half(0.5);
      Scalar One(1.0);
      auto Impl = [&](auto& X0,
                      auto& X1,
                      auto& FX0,
                      auto& FX1,
                      auto& JX0,
                      auto& JX1,
                      auto& AGX0,
                      auto& AGX1,
                      auto& HX0,
                      auto& HX1) {
        X0.template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
            x.template segment<DODE::XtUV>(0, this->ode.XtUVars());
        X0.tail(this->ode.PVars()) = x.tail(this->ode.PVars());

        X1.template segment<DODE::XtUV>(0, this->ode.XtUVars()) =
            x.template segment<DODE::XtUV>(this->ode.XtUVars(), this->ode.XtUVars());
        X1.tail(this->ode.PVars()) = x.tail(this->ode.PVars());
        Scalar h = X1[this->ode.TVar()] - X0[this->ode.TVar()];

        this->ode.compute_jacobian_adjointgradient_adjointhessian(X0, FX0, JX0, AGX0, HX0, adjvars);
        this->ode.compute_jacobian_adjointgradient_adjointhessian(X1, FX1, JX1, AGX1, HX1, adjvars);
        fx = (X1.template segment<DODE::XV>(0, this->ode.XVars())
              - X0.template segment<DODE::XV>(0, this->ode.XVars()))
             - (h / 2.0) * (FX0 + FX1);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        jx.template block<DODE::XV, DODE::XV>(0, 0, this->ode.XVars(), this->ode.XVars())
            .diagonal()
            .setConstant(-One);
        jx.template block<DODE::XV, DODE::XV>(0, this->ode.XtUVars(), this->ode.XVars(), this->ode.XVars())
            .diagonal()
            .setConstant(One);

        ODEOutput<Scalar> Tds = -One_Half * (FX0 + FX1);
        jx.col(this->ode.TVar()) -= Tds;
        jx.col(this->ode.XtUVars() + this->ode.TVar()) += Tds;

        jx.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode.XVars(), this->ode.XtUVars()) +=
            (-h / 2.0)
            * JX0.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode.XVars(), this->ode.XtUVars());

        jx.template block<DODE::XV, DODE::XtUV>(
            0, this->ode.XtUVars(), this->ode.XVars(), this->ode.XtUVars()) +=
            (-h / 2.0)
            * JX1.template block<DODE::XV, DODE::XtUV>(0, 0, this->ode.XVars(), this->ode.XtUVars());

        jx.template rightCols<DODE::PV>(this->ode.PVars()) =
            (-h / 2.0)
            * (JX0.template rightCols<DODE::PV>(this->ode.PVars())
               + JX1.template rightCols<DODE::PV>(this->ode.PVars()));

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        adjgrad = (adjvars.transpose() * jx).transpose();

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        adjhess.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode.XtUVars(), this->ode.XtUVars()) =
            (-h / 2.0)
            * HX0.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode.XtUVars(), this->ode.XtUVars());
        adjhess.template block<DODE::XtUV, DODE::XtUV>(
            this->ode.XtUVars(), this->ode.XtUVars(), this->ode.XtUVars(), this->ode.XtUVars()) =
            (-h / 2.0)
            * HX1.template block<DODE::XtUV, DODE::XtUV>(0, 0, this->ode.XtUVars(), this->ode.XtUVars());

        adjhess.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode.PVars(), this->ode.PVars()) =
            (-h / 2.0)
            * (HX0.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode.PVars(), this->ode.PVars())
               + HX1.template bottomRightCorner<DODE::PV, DODE::PV>(this->ode.PVars(), this->ode.PVars()));

        int j = 0;
        constexpr int Cardinals = 2;
        Input<Scalar> HTpar(this->IRows());
        HTpar.setZero();

        adjhess.template block<DODE::XtUV, DODE::PV>(j * this->ode.XtUVars(),
                                                     Cardinals * this->ode.XtUVars(),
                                                     this->ode.XtUVars(),
                                                     this->ode.PVars()) +=
            (-h / 2.0)
            * HX0.template block<DODE::XtUV, DODE::PV>(
                0, this->ode.XtUVars(), this->ode.XtUVars(), this->ode.PVars());
        adjhess.template block<DODE::PV, DODE::XtUV>(Cardinals * this->ode.XtUVars(),
                                                     j * this->ode.XtUVars(),
                                                     this->ode.PVars(),
                                                     this->ode.XtUVars()) +=
            (-h / 2.0)
            * HX0.template block<DODE::PV, DODE::XtUV>(
                this->ode.XtUVars(), 0, this->ode.PVars(), this->ode.XtUVars());
        HTpar.template segment<DODE::XtUV>(j * this->ode.XtUVars(), this->ode.XtUVars()) +=
            -AGX0.template segment<DODE::XtUV>(0, this->ode.XtUVars()) * One_Half;
        HTpar.tail(this->ode.PVars()) += -AGX0.tail(this->ode.PVars()) * One_Half;

        j = 1;
        adjhess.template block<DODE::XtUV, DODE::PV>(j * this->ode.XtUVars(),
                                                     Cardinals * this->ode.XtUVars(),
                                                     this->ode.XtUVars(),
                                                     this->ode.PVars()) +=
            (-h / 2.0)
            * HX1.template block<DODE::XtUV, DODE::PV>(
                0, this->ode.XtUVars(), this->ode.XtUVars(), this->ode.PVars());
        adjhess.template block<DODE::PV, DODE::XtUV>(Cardinals * this->ode.XtUVars(),
                                                     j * this->ode.XtUVars(),
                                                     this->ode.PVars(),
                                                     this->ode.XtUVars()) +=
            (-h / 2.0)
            * HX1.template block<DODE::PV, DODE::XtUV>(
                this->ode.XtUVars(), 0, this->ode.PVars(), this->ode.XtUVars());
        HTpar.template segment<DODE::XtUV>(j * this->ode.XtUVars(), this->ode.XtUVars()) +=
            -AGX1.template segment<DODE::XtUV>(0, this->ode.XtUVars()) * One_Half;
        HTpar.tail(this->ode.PVars()) += -AGX1.tail(this->ode.PVars()) * One_Half;

        adjhess.col(this->ode.TVar()) -= HTpar;
        adjhess.col(this->ode.TVar() + this->ode.XtUVars() * (Cardinals - 1)) += HTpar;
        adjhess.row(this->ode.TVar()) -= HTpar;
        adjhess.row(this->ode.TVar() + this->ode.XtUVars() * (Cardinals - 1)) += HTpar;
      };


      const int irows = this->ode.IRows();
      const int orows = this->ode.ORows();

      using IType = ODEInput<Scalar>;
      using OType = ODEOutput<Scalar>;
      using JType = ODEJacobian<Scalar>;
      using GType = ODEGrad<Scalar>;
      using HType = ODEHessian<Scalar>;

      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<IType>(irows, 1),
                                  TempSpec<IType>(irows, 1),

                                  TempSpec<OType>(orows, 1),
                                  TempSpec<OType>(orows, 1),

                                  TempSpec<JType>(orows, irows),
                                  TempSpec<JType>(orows, irows),

                                  TempSpec<GType>(irows, 1),
                                  TempSpec<GType>(irows, 1),

                                  TempSpec<HType>(irows, irows),
                                  TempSpec<HType>(irows, irows));


      fx *= Scalar(-1.0);
      jx *= Scalar(-1.0);
      adjgrad *= Scalar(-1.0);
      adjhess *= Scalar(-1.0);
    }
  };


}  // namespace ASSET
