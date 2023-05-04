#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<int St>
  struct UpperPadHolder {
    static const int UPad = St;
    UpperPadHolder(int st) {};
    UpperPadHolder() {};
  };

  template<>
  struct UpperPadHolder<-1> {
    int UPad = 0;
    UpperPadHolder(int st) {
      UPad = st;
    };
    UpperPadHolder() {};
  };

  template<class Func, int UP, int LP>
  struct PaddedOutput
      : VectorFunction<PaddedOutput<Func, UP, LP>, Func::IRC, SZ_SUM<LP, UP, Func::ORC>::value>,
        UpperPadHolder<UP> {
    using Base = VectorFunction<PaddedOutput<Func, UP, LP>, Func::IRC, SZ_SUM<LP, UP, Func::ORC>::value>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    Func func;
    static const bool IsVectorizable = Func::IsVectorizable;

    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static const bool IsLinearFunction = Func::IsLinearFunction;

    PaddedOutput(Func f, int upad, int lpad) : UpperPadHolder<UP>(upad), func(std::move(f)) {
      this->setIORows(this->func.IRows(), this->func.ORows() + upad + lpad);

      this->set_input_domain(this->IRows(), {func.input_domain()});
    }

    bool is_linear() const {
      return this->func.is_linear();
    }

    PaddedOutput() {
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      this->func.compute(x, fx.template segment<Func::ORC>(this->UPad, this->func.ORows()));
    }

    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      this->func.compute_jacobian(x,
                                  fx.template segment<Func::ORC>(this->UPad, this->func.ORows()),
                                  jx.template middleRows<Func::ORC>(this->UPad, this->func.ORows()));
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
        Eigen::MatrixBase<AdjGradType> const& adjgrad_,
        Eigen::MatrixBase<AdjHessType> const& adjhess_,
        const Eigen::MatrixBase<AdjVarType>& adjvars) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      Eigen::MatrixBase<AdjGradType>& adjgrad = adjgrad_.const_cast_derived();
      Eigen::MatrixBase<AdjHessType>& adjhess = adjhess_.const_cast_derived();

      this->func.compute_jacobian_adjointgradient_adjointhessian(
          x,
          fx.template segment<Func::ORC>(this->UPad, this->func.ORows()),
          jx.template middleRows<Func::ORC>(this->UPad, this->func.ORows()),
          adjgrad,
          adjhess,
          adjvars.template segment<Func::ORC>(this->UPad, this->func.ORows()));
    }

    //////////////////////////////////////////////////////////////////////////////
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
        ConstMatrixBaseRef<Right> right_ref(right.derived());
        ConstDiagonalBaseRef<Left> left_ref(left.derived());
        Base::right_jacobian_product(target_, left_ref, right_ref, assign, aliased);
      } else {
        MatrixBaseRef<Right> right_ref = right.const_cast_derived();
        MatrixBaseRef<Left> left_ref = left.const_cast_derived();
        this->func.right_jacobian_product(
            target_,
            left_ref.template middleCols<Func::ORC>(this->UPad, this->func.ORows()),
            right_ref.template middleRows<Func::ORC>(this->UPad, this->func.ORows()),
            assign,
            aliased);
      }
    }

    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right_,
                                    Assignment assign) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      MatrixBaseRef<JacType> right = right_.const_cast_derived();

      this->func.accumulate_jacobian(target.template middleRows<Func::ORC>(this->UPad, this->func.ORows()),
                                     right.template middleRows<Func::ORC>(this->UPad, this->func.ORows()),
                                     assign);
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
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      this->func.scale_jacobian(target.template middleRows<Func::ORC>(this->UPad, this->func.ORows()), s);
    }
    template<class Target, class Scalar>
    inline void scale_gradient(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      this->func.scale_gradient(target_, s);
    }
    template<class Target, class Scalar>
    inline void scale_hessian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      this->func.scale_hessian(target_, s);
    }
  };

}  // namespace ASSET
