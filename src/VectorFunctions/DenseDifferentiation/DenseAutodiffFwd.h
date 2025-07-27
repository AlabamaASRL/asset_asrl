#pragma once

#include "DenseDerivatives.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace ASSET {

template <class Derived, int IR, int OR>
struct DenseFirstDerivatives<Derived, IR, OR, DenseDerivativeModes::AutodiffFwd>
    : DenseFunction<Derived, IR, OR> {
  using Base = DenseFunction<Derived, IR, OR>;
  DENSE_FUNCTION_BASE_TYPES(Base);

  template <class Scalar>
  using dual = autodiff::detail::HigherOrderDual<1U,Scalar>;

  template <class InType, class OutType, class JacType>
  inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                               ConstVectorBaseRef<OutType> fx_,
                               ConstMatrixBaseRef<JacType> jx_) const {
    typedef typename InType::Scalar Scalar;
    VectorBaseRef<OutType> fx = fx_.const_cast_derived();
    MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

    Input<dual<Scalar>> xdual = x.template cast<dual<Scalar>>();
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
  //using Base::adjointhessian;


  template <class Scalar>
  using dual = autodiff::detail::HigherOrderDual<2U, Scalar>;


  

  template <class InType, class OutType, class JacType, class AdjGradType,
            class AdjHessType, class AdjVarType>
  inline void compute_jacobian_adjointgradient_adjointhessian_impl(
      ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
      ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
      ConstMatrixBaseRef<AdjHessType> adjhess_,
      ConstVectorBaseRef<AdjVarType> adjvars) const {

      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> gx = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> hx = adjhess_.const_cast_derived();

      Input<dual<Scalar>> xdual = x.template cast<dual<Scalar>>();
      Output<dual<Scalar>> fdual(this->ORows());

      
      for (int i = 0; i < this->IRows(); i++)
      {

          xdual[i].grad.val = 1.0;
          dual<Scalar> vv ;

          for (int j = i; j < this->IRows(); j++)
          {

              fdual.setZero();

              xdual[j].val.grad = 1.0;

              this->derived().compute(xdual, fdual);

              xdual[j].val.grad = 0.0;

              vv = 0.0;
              for (int k = 0; k < this->ORows(); k++) {
                 
                  vv += fdual[k] * adjvars[k];
              }

              hx(i, j) = vv.grad.grad;
              hx(j, i) = hx(i, j);
          }
          gx[i] = vv.grad.val;
          for (int k = 0; k < this->ORows(); k++) {
              jx(k, i) = fdual[k].grad.val;
          }
          if (i == 0) {
			  for (int j = 0; j < this->ORows(); j++) {
				  fx[j] = fdual[j].val.val;
			  }
		  }

          xdual[i].grad.val = 0.0;

      }
  }
};

}  // namespace ASSET
