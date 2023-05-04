#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class Func, int IR, int OR>
  struct FunctionHolder : VectorFunction<Derived, IR, OR, Analytic> {
    using Base = VectorFunction<Derived, IR, OR, Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    Func func;
    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static const bool IsLinearFunction = Func::IsLinearFunction;
    static const bool IsVectorizable = Func::IsVectorizable;

    FunctionHolder() {
    }
    FunctionHolder(Func f) : func(std::move(f)) {
      this->setIORows(this->func.IRows(), this->func.ORows());
      this->set_input_domain(this->IRows(), {func.input_domain()});
    }
    void set_function(Func f) {
      this->func = f;
      this->setIORows(this->func.IRows(), this->func.ORows());
      this->set_input_domain(this->IRows(), {func.input_domain()});
    }
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      this->func.compute(x, fx_);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      this->func.compute_jacobian(x, fx_, jx_);
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
      this->func.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_, adjvars);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      this->func.right_jacobian_product(target_, left, right, assign, aliased);
    }
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void symetric_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                          ConstEigenBaseRef<Left> left,
                                          ConstEigenBaseRef<Right> right,
                                          Assignment assign,
                                          std::bool_constant<Aliased> aliased) const {
      this->func.symetric_jacobian_product(target_, left, right, assign, aliased);
    }
    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      this->func.accumulate_jacobian(target_, right, assign);
    }
    template<class Target, class JacType, class Assignment>
    inline void accumulate_gradient(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      this->func.accumulate_gradient(target_, right, assign);
    }
    template<class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right,
                                   Assignment assign) const {
      this->func.accumulate_hessian(target_, right, assign);
    }
    template<class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      this->func.scale_jacobian(target_, s);
    }
    template<class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      this->func.scale_gradient(target_, s);
    }
    template<class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      this->func.scale_hessian(target_, s);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET
