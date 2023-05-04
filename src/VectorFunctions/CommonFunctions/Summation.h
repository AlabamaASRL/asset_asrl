#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class Func1, class Func2, bool DoDifference>
  struct TwoFunctionSum_Impl;

  template<class Derived, class Func1, class Func2, class... Funcs>
  struct MultiFunctionSum_Impl;

  template<class Func1, class Func2>
  struct TwoFunctionSum : TwoFunctionSum_Impl<TwoFunctionSum<Func1, Func2>, Func1, Func2, false> {
    using Base = TwoFunctionSum_Impl<TwoFunctionSum<Func1, Func2>, Func1, Func2, false>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };

  template<class Func1, class Func2>
  struct FunctionDifference : TwoFunctionSum_Impl<FunctionDifference<Func1, Func2>, Func1, Func2, true> {
    using Base = TwoFunctionSum_Impl<FunctionDifference<Func1, Func2>, Func1, Func2, true>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };

  template<class Func1, class Func2, class... Funcs>
  struct MultiFunctionSum
      : MultiFunctionSum_Impl<MultiFunctionSum<Func1, Func2, Funcs...>, Func1, Func2, Funcs...> {
    using Base = MultiFunctionSum_Impl<MultiFunctionSum<Func1, Func2, Funcs...>, Func1, Func2, Funcs...>;
    using Base::Base;
    DENSE_FUNCTION_BASE_TYPES(Base);
  };

  template<class F>
  struct Is_SumorDiff : std::false_type {};

  template<class F1, class F2>
  struct Is_SumorDiff<TwoFunctionSum<F1, F2>> : std::true_type {};

  template<class F1, class F2>
  struct Is_SumorDiff<FunctionDifference<F1, F2>> : std::true_type {};

  template<class Func1, class Func2, class... Funcs>
  auto make_sum(Func1 f1, Func2 f2, Funcs... fs) {
    return MultiFunctionSum<Func1, Func2, Funcs...>(f1, f2, fs...);
  }

  template<class Func1, class Func2>
  auto make_sum(Func1 f1, Func2 f2) {
    return TwoFunctionSum<Func1, Func2>(f1, f2);
  }

  template<class... Funcs>
  auto make_sum_tuple(std::tuple<Funcs...> fs) {
    constexpr int sz = sizeof...(Funcs);
    if constexpr (sz == 1) {
      return std::get<0>(fs);
    } else {
      return MultiFunctionSum<Funcs...>(fs);
    }
  }

  template<class Func1, class Func2>
  auto sum(Func1 f1, Func2 f2) {
    return TwoFunctionSum<Func1, Func2>(f1, f2);
  }

  template<class Func1, class Func2, class... Funcs>
  auto sum(Func1 f1, Func2 f2, Funcs... fs) {
    return MultiFunctionSum<Func1, Func2, Funcs...>(f1, f2, std::tuple {fs...});
  }


  template<class RetType, class FuncType>
  RetType make_dynamic_sum(const std::vector<FuncType>& funcs) {
    int size = funcs.size();
    RetType summed;
    if (size == 0) {
    } else if (size == 1) {
      summed = funcs[0];
    } else if (size == 2) {
      summed = make_sum(funcs[0], funcs[1]);
    } else if (size == 3) {
      summed = make_sum(funcs[0], funcs[1], funcs[2]);
    } else if (size == 4) {
      summed = make_sum(funcs[0], funcs[1], funcs[2], funcs[3]);
    } else if (size == 5) {
      summed = make_sum(funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]);
    } else {
      RetType summedT = make_sum(funcs[0], funcs[1], funcs[2], funcs[3], funcs[4]);
      std::vector<FuncType> nfuncs;
      for (int i = 5; i < funcs.size(); i++) {
        nfuncs.push_back(funcs[i]);
      }
      RetType rest = ASSET::make_dynamic_sum<RetType, FuncType>(nfuncs);
      summed = make_sum(summedT, rest);
    }
    return summed;
  }

  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////

  template<class Derived, class Func1, class Func2, bool DoDifference>
  struct TwoFunctionSum_Impl : VectorFunction<Derived,
                                              SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                              SZ_MAX<Func1::ORC, Func2::ORC>::value,
                                              Analytic> {
    using Base = VectorFunction<Derived,
                                SZ_MAX<Func1::IRC, Func2::IRC>::value,
                                SZ_MAX<Func1::ORC, Func2::ORC>::value,
                                Analytic>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    Func1 func1;
    Func2 func2;

    static const bool func1_is_segment = Is_Segment<Func1>::value || Is_ScaledSegment<Func1>::value;
    static const bool func2_is_segment = Is_Segment<Func2>::value || Is_ScaledSegment<Func2>::value;
    static const bool is_sum_of_segments = func1_is_segment && func2_is_segment;

    static const bool func1_is_sumordiff = Is_SumorDiff<Func1>::value;
    static const bool func2_is_sumordiff = Is_SumorDiff<Func2>::value;

    static const bool is_sum_of_sums = func1_is_sumordiff || func2_is_sumordiff;
    static const bool IsSegmentOp = Is_Segment<Func1>::value && Is_Segment<Func2>::value;

    static const bool IsLinearFunction = Func1::IsLinearFunction && Func2::IsLinearFunction;
    static const bool IsVectorizable = Func1::IsVectorizable && Func2::IsVectorizable;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename Func1::INPUT_DOMAIN, typename Func2::INPUT_DOMAIN>;

    TwoFunctionSum_Impl() {
    }
    TwoFunctionSum_Impl(Func1 f1, Func2 f2) : func1(std::move(f1)), func2(std::move(f2)) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());

      if (this->func1.ORows() != this->func2.ORows()) {
        fmt::print(
            fmt::fg(fmt::color::red),
            "Math Error in TwoFunctionSum/+- method !!!\n"
            "Output Size of Func1 (ORows = {0:}) does not match Output Size of Func2 (ORows = {1:}).\n",
            this->func1.ORows(),
            this->func2.ORows());
        throw std::invalid_argument("");
      }
      if (this->func1.IRows() != this->func2.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in TwoFunctionSum/+- method !!!\n"
                   "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 (IRows = {1:}).\n",
                   this->func1.IRows(),
                   this->func2.IRows());
        throw std::invalid_argument("");
      }

      this->setIORows(irtemp, this->func1.ORows());
      this->set_input_domain(this->IRows(), {func1.input_domain(), func2.input_domain()});
    }

    bool is_linear() const {
      return func1.is_linear() && func2.is_linear();
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Derived>(m, name);
      obj.def(py::init<Func1, Func2>());
      Base::DenseBaseBuild(obj);
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto& func2_fx) {
        this->func1.compute(x, fx);
        this->func2.compute(x, func2_fx);
        if constexpr (DoDifference) {
          fx -= func2_fx;
        } else {
          fx += func2_fx;
        }
      };

      const int orows = this->func2.ORows();
      const int crit_size = orows;
      using FType = Func2_Output<Scalar>;
      MemoryManager::allocate_run(crit_size, Impl, TempSpec<FType>(orows, 1));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      auto Impl = [&](auto& func2_fx, auto& func2_jx) {
        this->func1.compute_jacobian(x, fx, jx);
        this->func2.compute_jacobian(x, func2_fx, func2_jx);
        if constexpr (DoDifference) {
          fx -= func2_fx;
          this->func2.accumulate_jacobian(jx, func2_jx, MinusEqualsAssignment());
        } else {
          fx += func2_fx;
          this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
        }
      };

      const int orows = this->func2.ORows();
      const int irows = this->func2.IRows();
      const int crit_size = std::max({orows, irows});
      using FType = Func2_Output<Scalar>;
      using JType = Func2_jacobian<Scalar>;
      MemoryManager::allocate_run(crit_size, Impl, TempSpec<FType>(orows, 1), TempSpec<JType>(orows, irows));
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

      //////////////////////////////////////////////


      auto Impl = [&](auto& func2_fx, auto& func2_jx, auto& func2_adjgrad, auto& func2_adjhess) {
        this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx_, jx_, adjgrad_, adjhess_, adjvars);

        this->func2.compute_jacobian_adjointgradient_adjointhessian(
            x, func2_fx, func2_jx, func2_adjgrad, func2_adjhess, adjvars);

        if constexpr (DoDifference) {
          fx -= func2_fx;
          this->func2.accumulate_jacobian(jx, func2_jx, MinusEqualsAssignment());
          this->func2.accumulate_gradient(adjgrad, func2_adjgrad, MinusEqualsAssignment());
          this->func2.accumulate_hessian(adjhess, func2_adjhess, MinusEqualsAssignment());

        } else {
          fx += func2_fx;
          this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
          this->func2.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());

          this->func2.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
        }
      };

      const int orows = this->func2.ORows();
      const int irows = this->func2.IRows();
      const int crit_size = std::max({orows, irows});

      using FType = Func2_Output<Scalar>;
      using JType = Func2_jacobian<Scalar>;
      using GType = Func2_gradient<Scalar>;
      using HType = Func2_hessian<Scalar>;
      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<FType>(orows, 1),
                                  TempSpec<JType>(orows, irows),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<HType>(irows, irows));
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      if constexpr (is_sum_of_segments) {
        this->func1.right_jacobian_product(target_, left, right, assign, aliased);
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
          this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(), aliased);
        } else {
          this->func2.right_jacobian_product(target_, left, right, assign, aliased);
        }
      } else if constexpr (func1_is_sumordiff && func2_is_segment) {
        if constexpr (Func1::is_sum_of_segments) {
          this->func1.right_jacobian_product(target_, left, right, assign, aliased);
          if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
            this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(), aliased);
          } else {
            this->func2.right_jacobian_product(target_, left, right, assign, aliased);
          }

        } else {
          Base::right_jacobian_product(target_, left, right, assign, aliased);
        }
      } else {
        Base::right_jacobian_product(target_, left, right, assign, aliased);
      }
    }
    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      if constexpr (is_sum_of_segments) {
        this->func1.accumulate_jacobian(target_, right, assign);
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
          this->func2.accumulate_jacobian(target_, right, PlusEqualsAssignment());
        } else {
          this->func2.accumulate_jacobian(target_, right, assign);
        }
      } else {
        Base::accumulate_jacobian(target_, right, assign);
      }
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////
  };

  template<class Derived, class Func1, class Func2, class... Funcs>
  struct MultiFunctionSum_Impl : VectorFunction<Derived,
                                                SZ_MAX<Func1::IRC, Func2::IRC, Funcs::IRC...>::value,
                                                SZ_MAX<Func1::ORC, Func2::ORC, Funcs::ORC...>::value> {
    using Base = VectorFunction<Derived,
                                SZ_MAX<Func1::IRC, Func2::IRC, Funcs::IRC...>::value,
                                SZ_MAX<Func1::ORC, Func2::ORC, Funcs::ORC...>::value>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func1);
    SUB_FUNCTION_IO_TYPES(Func2);

    Func1 func1;
    Func2 func2;
    std::tuple<Funcs...> funcs;

    static const bool IsLinearFunction = SZ_PROD<int(Func1::IsLinearFunction),
                                                 int(Func2::IsLinearFunction),
                                                 int(Funcs::IsLinearFunction)...>::value
                                         == 1;

    static const bool IsVectorizable =
        SZ_PROD<int(Func1::IsVectorizable), int(Func2::IsVectorizable), int(Funcs::IsVectorizable)...>::value
        == 1;

    static const bool IsSumofSegments =
        SZ_PROD<int(Is_Segment<Func1>::value || Is_ScaledSegment<Func1>::value),
                int(Is_Segment<Func2>::value || Is_ScaledSegment<Func2>::value),
                int(Is_Segment<Funcs>::value || Is_ScaledSegment<Funcs>::value)...>::value
        == 1;

    using INPUT_DOMAIN = CompositeDomain<Base::IRC,
                                         typename Func1::INPUT_DOMAIN,
                                         typename Func2::INPUT_DOMAIN,
                                         typename Funcs::INPUT_DOMAIN...>;

    MultiFunctionSum_Impl() {
    }
    MultiFunctionSum_Impl(Func1 f1, Func2 f2, Funcs... fs)
        : func1(std::move(f1)), func2(std::move(f2)), funcs(fs...) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());

      this->setIORows(irtemp, this->func1.ORows());
      setdmn();
    }
    MultiFunctionSum_Impl(Func1 f1, Func2 f2, std::tuple<Funcs...> fs)
        : func1(std::move(f1)), func2(std::move(f2)), funcs(fs) {
      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
      this->setIORows(irtemp, this->func1.ORows());
      setdmn();
    }

    MultiFunctionSum_Impl(std::tuple<Func1, Func2, Funcs...> fs) {
      this->func1 = std::get<0>(fs);
      this->func2 = std::get<1>(fs);
      ASSET::constexpr_for_loop(std::integral_constant<int, 0>(),
                                std::integral_constant<int, sizeof...(Funcs)>(),
                                [&](auto i) { std::get<i.value>(this->funcs) = std::get<i.value + 2>(fs); });

      int irtemp = std::max(this->func1.IRows(), this->func2.IRows());
      this->setIORows(irtemp, this->func1.ORows());
      setdmn();
    }

    void setdmn() {
      std::vector<DomainMatrix> tmp;
      tmp.push_back(func1.input_domain());
      tmp.push_back(func2.input_domain());

      if (this->func1.ORows() != this->func2.ORows()) {
        fmt::print(
            fmt::fg(fmt::color::red),
            "Math Error in MultiFunctionSum/+ method !!!\n"
            "Output Size of Func1 (ORows = {0:}) does not match Output Size of Func2 (ORows = {1:}).\n",
            this->func1.ORows(),
            this->func2.ORows());
        throw std::invalid_argument("");
      }
      if (this->func1.IRows() != this->func2.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in MultiFunctionSum/+ method !!!\n"
                   "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func2 (IRows = {1:}).\n",
                   this->func1.IRows(),
                   this->func2.IRows());
        throw std::invalid_argument("");
      }


      ASSET::constexpr_for_loop(
          std::integral_constant<int, 0>(), std::integral_constant<int, sizeof...(Funcs)>(), [&](auto i) {
            tmp.push_back(std::get<i.value>(this->funcs).input_domain());
            if (this->func1.ORows() != std::get<i.value>(this->funcs).ORows()) {
              fmt::print(fmt::fg(fmt::color::red),
                         "Math Error in MultiFunctionSum/+ method !!!\n"
                         "Output Size of Func1 (ORows = {0:}) does not match Output Size of Func{2:} (ORows "
                         "= {1:}).\n",
                         this->func1.ORows(),
                         std::get<i.value>(this->funcs).ORows(),
                         i.value);
              throw std::invalid_argument("");
            }
            if (this->func1.IRows() != std::get<i.value>(this->funcs).IRows()) {
              fmt::print(fmt::fg(fmt::color::red),
                         "Math Error in MultiFunctionSum/+ method !!!\n"
                         "Input Size of Func1 (IRows = {0:}) does not match Input Size of Func{2:} (IRows = "
                         "{1:}).\n",
                         this->func1.IRows(),
                         std::get<i.value>(this->funcs).IRows(),
                         i.value);
              throw std::invalid_argument("");
            }
          });
      this->set_input_domain(this->IRows(), tmp);
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      auto Impl = [&](auto& func2_fx) {
        this->func1.compute(x, fx);
        func2_fx.setZero();
        this->func2.compute(x, func2_fx);
        fx += func2_fx;

        ASSET::tuple_for_each(this->funcs, [&](const auto& funci) {
          func2_fx.setZero();
          funci.compute(x, func2_fx);
          fx += func2_fx;
        });
      };

      const int orows = this->func2.ORows();
      const int crit_size = orows;
      using FType = Func2_Output<Scalar>;
      MemoryManager::allocate_run(crit_size, Impl, TempSpec<FType>(orows, 1));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      auto Impl = [&](auto& func2_fx, auto& func2_jx) {
        this->func1.compute_jacobian(x, fx, jx);
        this->func2.compute_jacobian(x, func2_fx, func2_jx);
        fx += func2_fx;
        this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());

        ASSET::tuple_for_each(this->funcs, [&](const auto& funci) {
          func2_fx.setZero();

          typedef typename std::remove_reference<decltype(funci)>::type FunciType;
          if constexpr (FunciType::InputIsDynamic) {
            func2_jx.setZero();
          } else {
            constexpr int sds = FunciType::INPUT_DOMAIN::SubDomains.size();
            ASSET::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                  constexpr int Start1 = FunciType::INPUT_DOMAIN::SubDomains[i.value][0];
                  constexpr int Size1 = FunciType::INPUT_DOMAIN::SubDomains[i.value][1];
                  func2_jx.template middleCols<Size1>(Start1, Size1).setZero();
                });
          }

          funci.compute_jacobian(x, func2_fx, func2_jx);
          fx += func2_fx;
          funci.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
        });
      };


      const int orows = this->func2.ORows();
      const int irows = this->func2.IRows();
      const int crit_size = std::max({orows, irows});
      using FType = Func2_Output<Scalar>;
      using JType = Func2_jacobian<Scalar>;
      MemoryManager::allocate_run(crit_size, Impl, TempSpec<FType>(orows, 1), TempSpec<JType>(orows, irows));
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
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();

      //
      auto Impl = [&](auto& func2_fx, auto& func2_jx, auto& func2_adjgrad, auto& func2_adjhess) {
        this->func1.compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, adjgrad, adjhess, adjvars);

        this->func2.compute_jacobian_adjointgradient_adjointhessian(
            x, func2_fx, func2_jx, func2_adjgrad, func2_adjhess, adjvars);

        fx += func2_fx;

        this->func2.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
        this->func2.accumulate_gradient(adjgrad_, func2_adjgrad, PlusEqualsAssignment());
        this->func2.accumulate_hessian(adjhess_, func2_adjhess, PlusEqualsAssignment());

        ASSET::tuple_for_each(this->funcs, [&](const auto& funci) {
          func2_fx.setZero();
          func2_adjgrad.setZero();

          typedef typename std::remove_reference<decltype(funci)>::type FunciType;

          // func2_jx.setZero();
          funci.zero_matrix_domain(func2_jx);
          if constexpr (!FunciType::IsLinearFunction) {
            // func2_adjhess.setZero();
            funci.zero_matrix_domain(func2_adjhess);
          }

          funci.compute_jacobian_adjointgradient_adjointhessian(
              x, func2_fx, func2_jx, func2_adjgrad, func2_adjhess, adjvars);
          fx += func2_fx;
          funci.accumulate_jacobian(jx, func2_jx, PlusEqualsAssignment());
          funci.accumulate_gradient(adjgrad, func2_adjgrad, PlusEqualsAssignment());
          if constexpr (!FunciType::IsLinearFunction)
            funci.accumulate_hessian(adjhess, func2_adjhess, PlusEqualsAssignment());
        });
      };


      const int orows = this->func2.ORows();
      const int irows = this->func2.IRows();
      const int crit_size = std::max({orows, irows});

      using FType = Func2_Output<Scalar>;
      using JType = Func2_jacobian<Scalar>;
      using GType = Func2_gradient<Scalar>;
      using HType = Func2_hessian<Scalar>;
      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<FType>(orows, 1),
                                  TempSpec<JType>(orows, irows),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<HType>(irows, irows));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    template<class Target, class Left, class Right, class Assignment, bool Aliased>
    inline void right_jacobian_product(ConstMatrixBaseRef<Target> target_,
                                       ConstEigenBaseRef<Left> left,
                                       ConstEigenBaseRef<Right> right,
                                       Assignment assign,
                                       std::bool_constant<Aliased> aliased) const {
      if constexpr (IsSumofSegments) {
        this->func1.right_jacobian_product(target_, left, right, assign, aliased);
        if constexpr (std::is_same<Assignment, DirectAssignment>::value) {
          this->func2.right_jacobian_product(target_, left, right, PlusEqualsAssignment(), aliased);
          ASSET::tuple_for_each(this->funcs, [&](const auto& func) {
            func.right_jacobian_product(target_, left, right, PlusEqualsAssignment(), aliased);
          });
        } else {
          this->func2.right_jacobian_product(target_, left, right, assign, aliased);
          ASSET::tuple_for_each(this->funcs, [&](const auto& func) {
            func.right_jacobian_product(target_, left, right, assign, aliased);
          });
        }
      } else {
        Base::right_jacobian_product(target_, left, right, assign, aliased);
      }
    }
    ///////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET
