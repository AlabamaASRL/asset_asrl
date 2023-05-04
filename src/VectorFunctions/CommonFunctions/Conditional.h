#pragma once

#include "VectorFunction.h"

namespace ASSET {

  enum ConditionalFlags {
    LessThanFlag,
    GreaterThanFlag,
    LessThanEqualToFlag,
    GreaterThanEqualToFlag,
    EqualToFlag,
    ANDFlag,
    ORFlag,
  };


  template<class LHS, class RHS>
  struct ConditionalStatement {

    static const int IRC = SZ_MAX<LHS::IRC, RHS::IRC>::value;
    static const bool IsConditional = true;
    static const bool MetaConditional = LHS::IsConditional && RHS::IsConditional;

    template<class Scalar>
    using Input = Eigen::Matrix<Scalar, IRC, 1>;
    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;

    ConditionalStatement() {
    }
    ConditionalStatement(LHS lhss, ConditionalFlags flagss, RHS rhss)
        : lhs(std::move(lhss)), flag(flagss), rhs(std::move(rhss)) {
      this->InputRows = lhs.IRows();
      if (lhs.IRows() != rhs.IRows()) {
        throw std::invalid_argument("LHS and RHS of conditional statement must have same input rows");
      }
      if constexpr (!MetaConditional) {
        if (lhs.ORows() > 1 || rhs.ORows() > 1) {
          throw std::invalid_argument("LHS and RHS of conditional statement must be scalar functions");
        }
        if (flag == ConditionalFlags::ANDFlag || flag == ConditionalFlags::ORFlag) {
          throw std::invalid_argument("AND OR not defined for scalar conditionals");
        }
      } else {
        if (flag != ConditionalFlags::ANDFlag && flag != ConditionalFlags::ORFlag) {
          throw std::invalid_argument("Comparisons not defined for meta conditionals");
        }
      }
    }

    template<class InType>
    inline bool compute(ConstVectorBaseRef<InType> x) const {
      typedef typename InType::Scalar Scalar;
      if constexpr (MetaConditional) {
        bool left = this->lhs.compute(x);
        bool right = this->rhs.compute(x);
        bool result = false;
        if (this->flag == ConditionalFlags::ANDFlag) {
          result = left && right;
        } else if (this->flag == ConditionalFlags::ORFlag) {
          result = left || right;
        } else {
        }
        return result;
      } else {
        Vector1<Scalar> left;
        Vector1<Scalar> right;
        this->lhs.compute(x, left);
        this->rhs.compute(x, right);
        bool result = false;
        switch (this->flag) {
          case ConditionalFlags::LessThanFlag: {
            result = left[0] < right[0];
            break;
          }
          case ConditionalFlags::GreaterThanFlag: {
            result = left[0] > right[0];
            break;
          }
          case ConditionalFlags::LessThanEqualToFlag: {
            result = left[0] <= right[0];
            break;
          }
          case ConditionalFlags::GreaterThanEqualToFlag: {
            result = left[0] >= right[0];
            break;
          }
          case ConditionalFlags::EqualToFlag: {
            result = left[0] == right[0];
            break;
          }
          default: {
          }
        }

        return result;
      }
    }


    int IRows() const {
      return this->InputRows;
    }
    std::string name() const {
      return this->name_;
    }

   protected:
    ConditionalFlags flag;
    LHS lhs;
    RHS rhs;
    std::string name_;
    int InputRows = 0;
  };


  struct ConstantConditional {
    template<class Scalar>
    using Input = Eigen::Matrix<Scalar, -1, 1>;
    template<class Scalar>
    using ConstVectorBaseRef = const Eigen::MatrixBase<Scalar>&;

    ConstantConditional() {
    }
    ConstantConditional(int irows, bool value) : InputRows(irows), value(value) {
    }
    ConstantConditional(bool value) : value(value) {
    }
    template<class InType>
    inline bool compute(ConstVectorBaseRef<InType> x) const {
      return this->value;
    }

    int IRows() const {
      return this->InputRows;
    }
    std::string name() const {
      return this->name_;
    }

   protected:
    bool value;

    int InputRows = 0;
    std::string name_;
  };


  template<class TestFunc, class TrueFunc, class FalseFunc>
  struct IfElseFunction : VectorFunction<IfElseFunction<TestFunc, TrueFunc, FalseFunc>,
                                         SZ_MAX<TrueFunc::IRC, FalseFunc::IRC>::value,
                                         SZ_MAX<TrueFunc::ORC, FalseFunc::ORC>::value> {
    using Base = VectorFunction<IfElseFunction<TestFunc, TrueFunc, FalseFunc>,
                                SZ_MAX<TrueFunc::IRC, FalseFunc::IRC>::value,
                                SZ_MAX<TrueFunc::ORC, FalseFunc::ORC>::value>;
    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename TrueFunc::INPUT_DOMAIN, typename FalseFunc::INPUT_DOMAIN>;

    DENSE_FUNCTION_BASE_TYPES(Base);
    static const bool IsVectorizable = false;


    TestFunc test_func;
    TrueFunc true_func;
    FalseFunc false_func;
    IfElseFunction() {
    }

    IfElseFunction(TestFunc test, TrueFunc _true, FalseFunc _false)
        : test_func(std::move(test)), true_func(std::move(_true)), false_func(std::move(_false)) {
      this->setIORows(this->true_func.IRows(), this->true_func.ORows());

      this->set_input_domain(this->IRows(),
                             {this->true_func.input_domain(), this->false_func.input_domain()});
      if (this->true_func.ORows() != this->false_func.ORows()) {
        throw std::invalid_argument(
            "True and false functions in conditional statement must have same number of outputrows.");
      }
      if (this->true_func.IRows() != this->false_func.IRows()) {
        throw std::invalid_argument(
            "True and false functions in conditional statement must have same number of inputrows.");
      }
      if (this->test_func.IRows() != this->false_func.IRows()) {

        throw std::invalid_argument(
            "Test,True,and False functions in conditional statement must have same number of inputrows.");
      }
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {

      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      if (this->test_func.compute(x)) {
        this->true_func.compute(x, fx);
      } else {
        this->false_func.compute(x, fx);
      }
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      if (this->test_func.compute(x)) {
        this->true_func.compute_jacobian(x, fx_, jx_);
      } else {
        this->false_func.compute_jacobian(x, fx_, jx_);
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
      if (this->test_func.compute(x)) {
        this->true_func.compute_jacobian_adjointgradient_adjointhessian(
            x, fx_, jx_, adjgrad_, adjhess_, adjvars);
      } else {
        this->false_func.compute_jacobian_adjointgradient_adjointhessian(
            x, fx_, jx_, adjgrad_, adjhess_, adjvars);
      }
    }
  };

}  // namespace ASSET
