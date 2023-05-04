/*
File Name: Scaled.h

File Description: Defines ASSET all VectorFunction for computing the root of a Scalar function inside of
an expression.

////////////////////////////////////////////////////////////////////////////////

Original File Developer : James B. Pezent - jbpezent - jbpezent@crimson.ua.edu

Current File Maintainers:
    1. James B. Pezent - jbpezent         - jbpezent@crimson.ua.edu
    2. Full Name       - GitHub User Name - Current Email
    3. ....


Usage of this source code is governed by the license found
in the LICENSE file in ASSET's top level directory.

*/

#pragma once
#include "VectorFunction.h"


namespace ASSET {

  template<class Derived, class FX, class DFX>
  struct ScalarRootFinder_Impl;


  template<class FX, class DFX>
  struct ScalarRootFinder : ScalarRootFinder_Impl<ScalarRootFinder<FX, DFX>, FX, DFX> {
    using Base = ScalarRootFinder_Impl<ScalarRootFinder<FX, DFX>, FX, DFX>;
    using Base::Base;
  };

  /// <summary>
  /// Defines a Vector Function for computing the root of Scalar Function FX, with derivative DFX.
  /// By definition, the first input to FX is the Scalar parameter that we can adjust to find the root,
  /// and should be set to your desired initial guess. The rest of the inputs to FX are parameters that
  /// the output will be differentiated wrt to. You can also provide a second function scalar DFX that is the
  /// derivative of the FX wrt to only the iteration variable. Otherwise it will be computed from the jacobian
  /// of FX
  /// </summary>
  /// <typeparam name="Derived"></typeparam>
  /// <typeparam name="FX"></typeparam>
  /// <typeparam name="DFX"></typeparam>
  template<class Derived, class FX, class DFX>
  struct ScalarRootFinder_Impl : VectorFunction<Derived, FX::IRC, 1> {

    using Base = VectorFunction<Derived, FX::IRC, 1>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    double tol = 1.0e-9;
    int MaxIters = 10;

    FX fxfunc;
    DFX dfxfunc;

    ScalarRootFinder_Impl() {
    }

    ScalarRootFinder_Impl(FX f, DFX df, int iter, double tol)
        : fxfunc(f), dfxfunc(df), MaxIters(iter), tol(tol) {

      if constexpr (!std::is_same<DFX, std::false_type>::value) {
        if (f.IRows() != df.IRows()) {
          throw std::invalid_argument(
              "Root and First Derivative functions must have same number of Input Rows");
        }
      }
      this->setIORows(f.IRows(), 1);
    }

    template<class VecType, class JacType>
    void find_root(VecType& x, JacType& jx) const {
      typedef typename VecType::Scalar Scalar;
      Vector1<Scalar> fx;
      Vector1<Scalar> dfx;

      for (int i = 0; i < this->MaxIters; i++) {

        if constexpr (!std::is_same<DFX, std::false_type>::value) {
          this->fxfunc.compute(x, fx);
          if (abs(fx[0]) < tol)
            break;
          this->dfxfunc.compute(x, dfx);
          x[0] -= fx[0] / dfx[0];
          fx[0] = 0;
          dfx[0] = 0;
        } else {
          this->fxfunc.compute_jacobian(x, fx, jx);
          if (abs(fx[0]) < tol)
            break;
          x[0] -= fx[0] / jx(0, 0);
          fx[0] = 0;
          jx.setZero();
        }
      }
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      const int irows = this->IRows();


      auto Impl = [&](auto& xtmp, auto& jtmp) {
        xtmp = x;
        this->find_root(xtmp, jtmp);
        fx[0] = xtmp[0];
      };


      MemoryManager::allocate_run(
          irows, Impl, TempSpec<Input<Scalar>>(irows, 1), TempSpec<Jacobian<Scalar>>(1, irows));
    }


    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      const int irows = this->IRows();


      auto Impl = [&](auto& xtmp, auto& jtmp) {
        xtmp = x;
        this->find_root(xtmp, jtmp);

        fx[0] = xtmp[0];
        Vector1<Scalar> fxtmp;
        this->fxfunc.compute_jacobian(xtmp, fxtmp, jx);
        jx /= -jx(0, 0);
        jx(0, 0) = 0;
      };


      MemoryManager::allocate_run(
          irows, Impl, TempSpec<Input<Scalar>>(irows, 1), TempSpec<Jacobian<Scalar>>(1, irows));
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


      const int irows = this->IRows();


      auto Impl = [&](auto& xtmp, auto& jtmp, auto& htmp) {
        xtmp = x;
        this->find_root(xtmp, jtmp);

        fx[0] = xtmp[0];
        Vector1<Scalar> fxtmp;
        this->fxfunc.compute_jacobian_adjointgradient_adjointhessian(xtmp, fxtmp, jx, adjgrad, htmp, adjvars);
        htmp /= -jx(0, 0);
        adjhess.noalias() =
            htmp
            + (jx.transpose() * (jx * htmp(0, 0) / jx(0, 0) - htmp.row(0)) - htmp.col(0) * jx) / jx(0, 0);


        jx /= -jx(0, 0);
        jx(0, 0) = 0;
        adjgrad = jx.transpose() * adjvars[0];
        adjhess.row(0).setZero();
        adjhess.col(0).setZero();
      };


      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<Input<Scalar>>(irows, 1),
                                  TempSpec<Jacobian<Scalar>>(1, irows),
                                  TempSpec<Hessian<Scalar>>(irows, irows));
    }
  };


}  // namespace ASSET