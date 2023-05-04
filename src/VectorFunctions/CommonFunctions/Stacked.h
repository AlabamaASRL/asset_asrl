#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class Func1, class Func2>
  struct StackTwoOutputs_Impl;

  template<class Func1, class Func2, class... Funcs>
  struct StackedOutputs : StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Funcs...>,
                                               StackedOutputs<Func1, Func2>,
                                               StackedOutputs<Funcs...>> {
    using Base = StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Funcs...>,
                                      StackedOutputs<Func1, Func2>,
                                      StackedOutputs<Funcs...>>;
    using Base::Base;
    StackedOutputs() {};
    StackedOutputs(Func1 f1, Func2 f2, Funcs... fs)
        : Base(StackedOutputs<Func1, Func2>(f1, f2), StackedOutputs<Funcs...>(fs...)) {};
  };

  template<class Func1, class Func2, class Func3>
  struct StackedOutputs<Func1, Func2, Func3>
      : StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Func3>, StackedOutputs<Func1, Func2>, Func3> {
    using Base =
        StackTwoOutputs_Impl<StackedOutputs<Func1, Func2, Func3>, StackedOutputs<Func1, Func2>, Func3>;
    using Base::Base;
    StackedOutputs() {};
    StackedOutputs(Func1 f1, Func2 f2, Func3 f3) : Base(StackedOutputs<Func1, Func2>(f1, f2), f3) {};
  };

  template<class Func1, class Func2>
  struct StackedOutputs<Func1, Func2> : StackTwoOutputs_Impl<StackedOutputs<Func1, Func2>, Func1, Func2> {
    using Base = StackTwoOutputs_Impl<StackedOutputs<Func1, Func2>, Func1, Func2>;
    using Base::Base;
  };


  template<class Func1, class Func2, class... Funcs>
  StackedOutputs<Func1, Func2, Funcs...> stack(Func1 f1, Func2 f2, Funcs... fs) {
    return StackedOutputs<Func1, Func2, Funcs...>(f1, f2, fs...);
  }


  template<class RetType, class FuncType>
  RetType make_dynamic_stack(const std::vector<FuncType>& funcs) {
    int size = funcs.size();
    RetType stacked;
    if (size == 0) {
    } else if (size == 1) {
      stacked = funcs[0];
    } else if (size == 2) {
      stacked = StackedOutputs {funcs[0], funcs[1]};
    } else if (size == 3) {
      stacked = StackedOutputs {funcs[0], funcs[1], funcs[2]};
    } else if (size == 4) {
      stacked = StackedOutputs {funcs[0], funcs[1], funcs[2], funcs[3]};
    } else if (size == 5) {
      stacked = StackedOutputs {funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]};
    } else {
      RetType stackedT = StackedOutputs {funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]};
      std::vector<FuncType> nfuncs;
      for (int i = 5; i < funcs.size(); i++) {
        nfuncs.push_back(funcs[i]);
      }
      RetType rest = ASSET::make_dynamic_stack<RetType, FuncType>(nfuncs);
      stacked = StackedOutputs {stackedT, rest};
    }
    return stacked;
  }

  template<class Derived, class Func1, class Func2>
  struct StackTwoOutputs_Impl : VectorFunction<Derived,
                                               SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                               SZ_SUM<Func1::ORC, Func2::ORC>::value> {
    using Base =
        VectorFunction<Derived, SZ_MAX<Func1::IRC, Func2::IRC>::value, SZ_SUM<Func1::ORC, Func2::ORC>::value>;
    using Base::compute;

    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    static const bool IsLinearFunction = Func1::IsLinearFunction && Func2::IsLinearFunction;
    static const bool IsVectorizable = Func1::IsVectorizable && Func2::IsVectorizable;

    Func1 func1;
    Func2 func2;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

    StackTwoOutputs_Impl() {
    }
    StackTwoOutputs_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
      this->setIORows(irtemp, this->func1.ORows() + this->func2.ORows());

      if (this->func1.IRows() != this->func2.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in StackOutputs/vf.stack method !!!\n"
                   "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 (IRows = {1:}).\n",
                   this->func1.IRows(),
                   this->func2.IRows());
        throw std::invalid_argument("");
      }

      this->set_input_domain(this->IRows(), {func1.input_domain(), func2.input_domain()});
    }

    bool is_linear() const {
      return func1.is_linear() && func2.is_linear();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      this->func1.compute(x, fx.template segment<Func1::ORC>(0, this->func1.ORows()));
      this->func2.compute(x, fx.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      this->func1.compute_jacobian(x,
                                   fx.template segment<Func1::ORC>(0, this->func1.ORows()),
                                   jx.template topRows<Func1::ORC>(this->func1.ORows()));

      this->func2.compute_jacobian(x,
                                   fx.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()),
                                   jx.template bottomRows<Func2::ORC>(this->func2.ORows()));
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

      auto Impl = [&](auto& func2_adjgrad, auto& func2_adjhess) {
        if constexpr (Func1::IsGenericFunction && Func2::IsGenericFunction) {
          if (this->func1.is_linear()) {
            this->func1.compute_jacobian_adjointgradient_adjointhessian(
                x,
                fx.template segment<Func1::ORC>(0, this->func1.ORows()),
                jx.template topRows<Func1::ORC>(this->func1.ORows()),
                func2_adjgrad,
                func2_adjhess,
                adjvars.template segment<Func1::ORC>(0, this->func1.ORows()));

            this->func2.compute_jacobian_adjointgradient_adjointhessian(
                x,
                fx.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()),
                jx.template bottomRows<Func2::ORC>(this->func2.ORows()),
                adjgrad,
                adjhess,
                adjvars.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()));

            this->func1.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());

          } else {
            this->func1.compute_jacobian_adjointgradient_adjointhessian(
                x,
                fx.template segment<Func1::ORC>(0, this->func1.ORows()),
                jx.template topRows<Func1::ORC>(this->func1.ORows()),
                adjgrad,
                adjhess,
                adjvars.template segment<Func1::ORC>(0, this->func1.ORows()));

            this->func2.compute_jacobian_adjointgradient_adjointhessian(
                x,
                fx.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()),
                jx.template bottomRows<Func2::ORC>(this->func2.ORows()),
                func2_adjgrad,
                func2_adjhess,
                adjvars.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()));

            // if (!this->func2.is_linear())
            this->func2.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
            this->func2.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());
          }

        } else {
          this->func1.compute_jacobian_adjointgradient_adjointhessian(
              x,
              fx.template segment<Func1::ORC>(0, this->func1.ORows()),
              jx.template topRows<Func1::ORC>(this->func1.ORows()),
              adjgrad,
              adjhess,
              adjvars.template segment<Func1::ORC>(0, this->func1.ORows()));

          this->func2.compute_jacobian_adjointgradient_adjointhessian(
              x,
              fx.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()),
              jx.template bottomRows<Func2::ORC>(this->func2.ORows()),
              func2_adjgrad,
              func2_adjhess,
              adjvars.template segment<Func2::ORC>(this->func1.ORows(), this->func2.ORows()));

          this->func2.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
          this->func2.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());
        }
      };


      const int irows = this->func2.IRows();
      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<Func2_gradient<Scalar>>(irows, 1),
                                  TempSpec<Func2_hessian<Scalar>>(irows, irows));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      if constexpr (Is_EigenDiagonalMatrix<Left>::value) {
        Base::right_jacobian_product(target_, left, right, assign, aliased);

      } else {
        MatrixBaseRef<Right> right_ref = right.const_cast_derived();
        MatrixBaseRef<Left> left_ref = left.const_cast_derived();

        this->func1.right_jacobian_product(target_,
                                           left_ref.template leftCols<Func1::ORC>(this->func1.ORows()),
                                           right_ref.template topRows<Func1::ORC>(this->func1.ORows()),
                                           assign,
                                           aliased);
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
          this->func2.right_jacobian_product(target_,
                                             left_ref.template rightCols<Func2::ORC>(this->func2.ORows()),
                                             right_ref.template bottomRows<Func2::ORC>(this->func2.ORows()),
                                             PlusEqualsAssignment(),
                                             aliased);
        } else {
          this->func2.right_jacobian_product(target_,
                                             left_ref.template rightCols<Func2::ORC>(this->func2.ORows()),
                                             right_ref.template bottomRows<Func2::ORC>(this->func2.ORows()),
                                             assign,
                                             aliased);
        }
      }
    }

    template<class Target, class Scalar>
    inline void scale_jacobian(ConstMatrixBaseRef<Target> target_, Scalar s) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      this->func1.scale_jacobian(target.template topRows<Func1::ORC>(this->func1.ORows()), s);
      this->func2.scale_jacobian(target.template bottomRows<Func2::ORC>(this->func2.ORows()), s);
    }

    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      MatrixBaseRef<JacType> right_ref = right.const_cast_derived();

      // typedef typename Target::Scalar Scalar;

      this->func1.accumulate_jacobian(target.template topRows<Func1::ORC>(this->func1.ORows()),
                                      right_ref.template topRows<Func1::ORC>(this->func1.ORows()),
                                      assign);
      this->func2.accumulate_jacobian(target.template bottomRows<Func2::ORC>(this->func2.ORows()),
                                      right_ref.template bottomRows<Func2::ORC>(this->func2.ORows()),
                                      assign);
    }

    template<class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right,
                                   Assignment assign) const {
      if constexpr (Func1::IsLinearFunction) {
        this->func2.accumulate_hessian(target_, right, assign);
      } else if constexpr (Func2::IsLinearFunction) {
        this->func1.accumulate_hessian(target_, right, assign);
      } else {
        Base::accumulate_hessian(target_, right, assign);
      }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };


  template<class Func>
  struct DynamicStackedOutputs : VectorFunction<DynamicStackedOutputs<Func>, Func::IRC, -1> {
    using Base = VectorFunction<DynamicStackedOutputs<Func>, Func::IRC, -1>;
    using Base::compute;

    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func);

    static const bool IsLinearFunction = Func::IsLinearFunction;
    static const bool IsVectorizable = Func::IsVectorizable;

    std::vector<Func> funcs;

    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;

    DynamicStackedOutputs() {
    }
    DynamicStackedOutputs(const std::vector<Func>& funcs) : funcs(funcs) {


      if (this->funcs.size() == 0) {
        throw std::invalid_argument("Empty List passed to Dynamic Stack");
      }
      int ortemp = 0;
      int irtemp = this->funcs[0].IRows();

      std::vector<DomainMatrix> dmn;

      for (auto& func: this->funcs) {
        ortemp += func.ORows();
        dmn.push_back(func.input_domain());

        if (!func.is_linear())
          this->_linear = false;

        if (func.IRows() != irtemp) {
          fmt::print(fmt::fg(fmt::color::red),
                     "Math Error in StackOutputs/vf.stack method !!!\n"
                     "Input Size of all Functions must match.\n");
          throw std::invalid_argument("");
        }
      }

      this->setIORows(irtemp, ortemp);
      this->set_input_domain(this->IRows(), dmn);
    }


    bool is_linear() const {
      return this->_linear;
    }
    bool _linear = false;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      int start = 0;
      for (auto& func: this->funcs) {
        int orows = func.ORows();
        func.compute(x, fx.template segment<Func::ORC>(start, orows));
        start += orows;
      }
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


      int start = 0;
      for (auto& func: this->funcs) {
        int orows = func.ORows();
        func.compute_jacobian(
            x, fx.template segment<Func::ORC>(start, orows), jx.template middleRows<Func::ORC>(start, orows));
        start += orows;
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

      auto Impl = [&](auto& func_adjgrad, auto& func_adjhess) {
        int start = 0;
        for (auto& func: this->funcs) {
          int orows = func.ORows();

          if (start == 0) {
            func.compute_jacobian_adjointgradient_adjointhessian(
                x,
                fx.template segment<Func::ORC>(start, orows),
                jx.template middleRows<Func::ORC>(start, orows),
                adjgrad,
                adjhess,
                adjvars.template segment<Func::ORC>(start, orows));
          } else {
            func.compute_jacobian_adjointgradient_adjointhessian(
                x,
                fx.template segment<Func::ORC>(start, orows),
                jx.template middleRows<Func::ORC>(start, orows),
                func_adjgrad,
                func_adjhess,
                adjvars.template segment<Func::ORC>(start, orows));

            func.accumulate_hessian(adjhess, func_adjhess, PlusEqualsAssignment());
            func.accumulate_gradient(adjgrad, func_adjgrad, PlusEqualsAssignment());

            func.zero_matrix_domain(func_adjhess);
            func_adjgrad.setZero();
          }
          start += orows;
        }
      };


      const int irows = this->IRows();
      MemoryManager::allocate_run(irows,
                                  Impl,
                                  TempSpec<Func_gradient<Scalar>>(irows, 1),
                                  TempSpec<Func_hessian<Scalar>>(irows, irows));
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };


}  // namespace ASSET
