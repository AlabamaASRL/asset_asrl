#pragma once

#include "DenseDerivatives.h"
/*
namespace ASSET {

template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeModes::AutodiffFwd>
    : DenseFunction<Derived, IR, OR> {
  using Base = DenseFunction<Derived, IR, OR>;
  DENSE_FUNCTION_BASE_TYPES(Base);

  template <class Scalar>
  using dual = autodiff::Dual<Scalar, Scalar>;

  template <class InType, class OutType, class JacType>
  inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                               ConstVectorBaseRef<OutType> fx_,
                               ConstMatrixBaseRef<JacType> jx_) const {
    typedef typename InType::Scalar Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

    Input<dual<Scalar>> xdual;
    if constexpr (autodiff::isDual<Scalar>) {
      for (int i = 0; i < this->IRows(); i++) {
        xdual[i].val = x[i];
        xdual[i].grad = Scalar(0.0);
      }
    } else {
      xdual = x.template cast<dual<Scalar>>();
    }
    Output<dual<Scalar>> fdual(this->ORows());
    fdual.setZero();

    for (int i = 0; i < this->IRows(); i++) {
      xdual[i].grad = Scalar(1.0);
      this->derived().compute(xdual, fdual);

      for (int j = 0; j < this->ORows(); j++) {
        jx(j, i) = fdual[j].grad;
      }

      if (i == 0) {
        for (int j = 0; j < this->ORows(); j++) {
          fx[j] = fdual[j].val;
        }
      }

      fdual.setZero();
      xdual[i].grad = Scalar(0.0);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

template <class Derived, int IR, int OR, int JMode>
struct DenseSecondDerivatives<Derived, IR, OR, JMode,
                              DenseDerivativeModes::AutodiffFwd>
    : DenseFirstDerivatives<Derived, IR, OR, JMode> {
  using Base = DenseFirstDerivatives<Derived, IR, OR, JMode>;
  DENSE_FUNCTION_BASE_TYPES(Base);
  using Base::adjointhessian;
  template <class Scalar>
  using dual = autodiff::Dual<Scalar, Scalar>;

  template <class InType, class AdjHessType, class AdjVarType>
  inline void adjointhessian(ConstVectorBaseRef<InType> x,
                             ConstMatrixBaseRef<AdjHessType> adjhess_,
                             ConstVectorBaseRef<AdjVarType> adjvars) const {
    typedef typename InType::Scalar Scalar;
    MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

    Input<dual<Scalar>> xdual;
    if constexpr (autodiff::isDual<Scalar>) {
      for (int i = 0; i < this->IRows(); i++) {
        xdual[i].val = x[i];
        xdual[i].grad = Scalar(0.0);
      }
    } else {
      xdual = x.template cast<dual<Scalar>>();
    }
    Gradient<dual<Scalar>> adjg;
    adjg.setZero();
    for (int i = 0; i < this->IRows(); i++) {
      xdual[i].grad = Scalar(1.0);
      this->derived().adjointgradient(xdual, adjg, adjvars);

      for (int j = 0; j < this->IRows(); j++) {
        adjhess(j, i) = adjg[i].grad;
      }

      adjg.setZero();
      xdual[i].grad = Scalar(0.0);
    }
  }

  template <class InType, class OutType, class JacType, class AdjGradType,
            class AdjHessType, class AdjVarType>
  inline void compute_jacobian_adjointgradient_adjointhessian_impl(
      ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
      ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
      ConstMatrixBaseRef<AdjHessType> adjhess_,
      ConstVectorBaseRef<AdjVarType> adjvars) const {
    this->derived().compute_jacobian_adjointgradient(x, fx_, jx_, adjgrad_,
                                                     adjvars);
    adjointhessian(x, adjhess_, adjvars);
  }
};

}  // namespace ASSET
*/