#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Func>
  struct SignFunction : VectorFunction<SignFunction<Func>, Func::IRC, Func::ORC> {
    using Base = VectorFunction<SignFunction<Func>, Func::IRC, Func::ORC>;

    Func func;
    DENSE_FUNCTION_BASE_TYPES(Base);

    // using INPUT_DOMAIN = SingleDomain<Func::IRC, 0, 0>;


    SignFunction() {
    }
    SignFunction(Func func) : func(std::move(func)) {
      this->setIORows(this->func.IRows(), this->func.ORows());
      DomainMatrix dmn(2, 1);
      dmn(0, 0) = 0;
      dmn(1, 0) = 0;
      this->set_input_domain(this->IRows(), {dmn});
    }

    template<class OutType>
    void sign_impl(OutType& fx) const {
      typedef typename OutType::Scalar Scalar;
      if constexpr (Is_SuperScalar<Scalar>::value) {
        for (int i = 0; i < this->ORows(); i++) {
          fx[i] = fx[i].sign();
        }
      } else {
        fx = fx.array().sign();
      }
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      this->func.compute(x, fx);
      this->sign_impl(fx);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      this->func.compute(x, fx);
      this->sign_impl(fx);
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

      this->func.compute(x, fx);
      this->sign_impl(fx);
    }
  };


}  // namespace ASSET