#pragma once

#include "Segment.h"
#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class OuterFunc, class InnerFunc>
  struct NestedFunction_Impl;

  template<class OuterFunc, class InnerFunc>
  struct NestedFunction : NestedFunction_Impl<NestedFunction<OuterFunc, InnerFunc>, OuterFunc, InnerFunc> {
    using Base = NestedFunction_Impl<NestedFunction<OuterFunc, InnerFunc>, OuterFunc, InnerFunc>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<NestedFunction<OuterFunc, InnerFunc>>(m, name);
      obj.def(py::init<>());
      obj.def(py::init<OuterFunc, InnerFunc>());
      Base::DenseBaseBuild(obj);
    }
  };

  //////////////////////////////////////////////////////////////////////
  template<class Derived, class OuterFunc, class InnerFunc>
  struct NestedFunction_Impl : VectorFunction<Derived, InnerFunc::IRC, OuterFunc::ORC> {
    using Base = VectorFunction<Derived, InnerFunc::IRC, OuterFunc::ORC>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(OuterFunc);
    SUB_FUNCTION_IO_TYPES(InnerFunc);

    OuterFunc outer_func;
    InnerFunc inner_func;

    using INPUT_DOMAIN = typename InnerFunc::INPUT_DOMAIN;
    static const bool IsLinearFunction = OuterFunc::IsLinearFunction && InnerFunc::IsLinearFunction;
    static const bool IsVectorizable = OuterFunc::IsVectorizable && InnerFunc::IsVectorizable;


    NestedFunction_Impl() {
    }
    NestedFunction_Impl(OuterFunc ofunc, InnerFunc ifunc)
        : outer_func(std::move(ofunc)), inner_func(std::move(ifunc)) {
      if (this->inner_func.ORows() != this->outer_func.IRows()) {

        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in NestedFunction/.eval method !!!\n"
                   "Output Size of InnerFunction (ORows = {0:}) does not match Input Size of OuterFunction "
                   "(IRows = {1:}).\n",
                   this->inner_func.ORows(),
                   this->outer_func.IRows());

        throw std::invalid_argument("");
      }

      this->setIORows(this->inner_func.IRows(), this->outer_func.ORows());
      this->set_input_domain(this->IRows(), {inner_func.input_domain()});
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();


      auto Impl = [&](auto& fx_inner) {
        if constexpr (Is_Segment<OuterFunc>::value) {
          this->inner_func.compute(x, fx_inner);
          fx = fx_inner.template segment<OuterFunc::ORC>(this->outer_func.SegStart, this->ORows());
        } else {
          this->inner_func.compute(x, fx_inner);
          this->outer_func.compute(fx_inner, fx);
        }
      };

      const int orows = this->inner_func.ORows();
      const int crit_size = orows;
      using FType = InnerFunc_Output<Scalar>;
      MemoryManager::allocate_run(crit_size, Impl, TempSpec<FType>(orows, 1));
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      // MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      if constexpr (Is_Segment<OuterFunc>::value) {

        auto Impl = [&](auto& fx_inner, auto& jx_inner) {
          this->inner_func.compute_jacobian(x, fx_inner, jx_inner);
          fx = fx_inner.template segment<OuterFunc::ORC>(this->outer_func.SegStart, this->ORows());
          this->inner_func.accumulate_matrix_domain(
              jx_,
              jx_inner.template middleRows<OuterFunc::ORC>(this->outer_func.SegStart, this->ORows()),
              PlusEqualsAssignment());
        };

        const int inner_OR = this->inner_func.ORows();
        const int inner_IR = this->inner_func.IRows();
        const int outer_OR = this->outer_func.ORows();
        const int outer_IR = this->outer_func.IRows();
        const int crit_size = std::max({inner_OR, inner_IR, outer_OR, outer_IR});

        using IFXType = InnerFunc_Output<Scalar>;
        using IJXType = InnerFunc_jacobian<Scalar>;
        using OJXType = OuterFunc_jacobian<Scalar>;
        MemoryManager::allocate_run(
            crit_size, Impl, TempSpec<IFXType>(inner_OR, 1), TempSpec<IJXType>(inner_OR, inner_IR));

      } else {
        auto Impl = [&](auto& fx_inner, auto& jx_inner, auto& jx_outer) {
          this->inner_func.compute_jacobian(x, fx_inner, jx_inner);
          this->outer_func.compute_jacobian(fx_inner, fx_, jx_outer);
          this->inner_func.right_jacobian_product(
              jx_, jx_outer, jx_inner, DirectAssignment(), std::bool_constant<false>());
        };


        const int inner_OR = this->inner_func.ORows();
        const int inner_IR = this->inner_func.IRows();
        const int outer_OR = this->outer_func.ORows();
        const int outer_IR = this->outer_func.IRows();
        const int crit_size = std::max({inner_OR, inner_IR, outer_OR, outer_IR});

        using IFXType = InnerFunc_Output<Scalar>;
        using IJXType = InnerFunc_jacobian<Scalar>;
        using OJXType = OuterFunc_jacobian<Scalar>;
        MemoryManager::allocate_run(crit_size,
                                    Impl,
                                    TempSpec<IFXType>(inner_OR, 1),
                                    TempSpec<IJXType>(inner_OR, inner_IR),
                                    TempSpec<OJXType>(outer_OR, outer_IR));
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
      // MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      // VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();


      if constexpr (Is_Segment<OuterFunc>::value) {
        auto Impl = [&](auto& fx_inner, auto& jx_inner, auto& gx_outer) {
          gx_outer.template segment<OuterFunc::ORC>(this->outer_func.SegStart, this->ORows()) = adjvars;
          this->inner_func.compute_jacobian_adjointgradient_adjointhessian(
              x, fx_inner, jx_inner, adjgrad_, adjhess_, gx_outer);

          fx = fx_inner.template segment<OuterFunc::ORC>(this->outer_func.SegStart, this->ORows());
          this->inner_func.accumulate_matrix_domain(
              jx_,
              jx_inner.template middleRows<OuterFunc::ORC>(this->outer_func.SegStart, this->ORows()),
              PlusEqualsAssignment());
        };


        const int inner_OR = this->inner_func.ORows();
        const int inner_IR = this->inner_func.IRows();
        const int outer_OR = this->outer_func.ORows();
        const int outer_IR = this->outer_func.IRows();
        const int crit_size = std::max({inner_OR, inner_IR, outer_OR, outer_IR});

        using IFXType = InnerFunc_Output<Scalar>;
        using IJXType = InnerFunc_jacobian<Scalar>;
        using OJXType = OuterFunc_jacobian<Scalar>;
        using OGXType = OuterFunc_gradient<Scalar>;
        using OHXType = OuterFunc_hessian<Scalar>;

        MemoryManager::allocate_run(crit_size,
                                    Impl,
                                    TempSpec<IFXType>(inner_OR, 1),
                                    TempSpec<IJXType>(inner_OR, inner_IR),
                                    TempSpec<OGXType>(outer_IR, 1));
      } else {
        auto Impl = [&](auto& fx_inner,
                        auto& jx_inner,
                        auto& jx_outer,
                        auto& gx_outer,
                        auto& hx_outer,
                        auto& Ht) {
          this->inner_func.compute(x, fx_inner);
          this->outer_func.compute_jacobian_adjointgradient_adjointhessian(
              fx_inner, fx_, jx_outer, gx_outer, hx_outer, adjvars);
          fx_inner.setZero();
          this->inner_func.compute_jacobian_adjointgradient_adjointhessian(
              x, fx_inner, jx_inner, adjgrad_, adjhess_, gx_outer);
          this->inner_func.right_jacobian_product(
              jx_, jx_outer, jx_inner, DirectAssignment(), std::bool_constant<false>());

          if constexpr (!OuterFunc::IsLinearFunction) {


            this->inner_func.right_jacobian_product(
                Ht, hx_outer, jx_inner, DirectAssignment(), std::bool_constant<false>());
            if constexpr (InnerFunc::InputIsDynamic) {
              int sds = this->inner_func.SubDomains.cols();
              if (sds == 0) {
                this->inner_func.right_jacobian_product(
                    adjhess, Ht.transpose(), jx_inner, PlusEqualsAssignment(), std::bool_constant<false>());
              } else {

                for (int i = 0; i < sds; i++) {
                  int start = this->inner_func.SubDomains(0, i);
                  int size = this->inner_func.SubDomains(1, i);
                  this->inner_func.right_jacobian_product(adjhess.middleRows(start, size),
                                                          Ht.middleCols(start, size).transpose(),
                                                          jx_inner,
                                                          PlusEqualsAssignment(),
                                                          std::bool_constant<false>());
                }
              }

            } else {
              Eigen::Matrix<Scalar, InnerFunc::IRC, InnerFunc::ORC> HTT = Ht.transpose();
              constexpr int sds = InnerFunc::INPUT_DOMAIN::SubDomains.size();
              ASSET::constexpr_for_loop(
                  std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                    constexpr int start = InnerFunc::INPUT_DOMAIN::SubDomains[i.value][0];
                    constexpr int size = InnerFunc::INPUT_DOMAIN::SubDomains[i.value][1];

                    this->inner_func.right_jacobian_product(adjhess.template middleRows<size>(start, size),
                                                            HTT.template middleRows<size>(start, size),
                                                            jx_inner,
                                                            PlusEqualsAssignment(),
                                                            std::bool_constant<false>());
                  });
            }
          }
        };


        const int inner_OR = this->inner_func.ORows();
        const int inner_IR = this->inner_func.IRows();
        const int outer_OR = this->outer_func.ORows();
        const int outer_IR = this->outer_func.IRows();
        const int crit_size = std::max({inner_OR, inner_IR, outer_OR, outer_IR});

        using IFXType = InnerFunc_Output<Scalar>;
        using IJXType = InnerFunc_jacobian<Scalar>;
        using OJXType = OuterFunc_jacobian<Scalar>;
        using OGXType = OuterFunc_gradient<Scalar>;
        using OHXType = OuterFunc_hessian<Scalar>;

        MemoryManager::allocate_run(crit_size,
                                    Impl,
                                    TempSpec<IFXType>(inner_OR, 1),
                                    TempSpec<IJXType>(inner_OR, inner_IR),
                                    TempSpec<OJXType>(outer_OR, outer_IR),
                                    TempSpec<OGXType>(outer_IR, 1),
                                    TempSpec<OHXType>(outer_IR, outer_IR),
                                    TempSpec<IJXType>(inner_OR, inner_IR));
      }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

  template<class Derived, class OuterFunc, int IR, int OR, int ST>
  struct NestedFunction_Impl<Derived, OuterFunc, Segment<IR, OR, ST>>
      : VectorFunction<Derived, IR, OuterFunc::ORC>, SegStartHolder<ST> {
    using Base = VectorFunction<Derived, IR, OuterFunc::ORC>;
    using Base::compute;
    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(OuterFunc);

    using InnerFunc = Segment<IR, OR, ST>;

    OuterFunc outer_func;

    using INPUT_DOMAIN = typename Segment<IR, OR, ST>::INPUT_DOMAIN;
    static const bool IsLinearFunction = OuterFunc::IsLinearFunction && InnerFunc::IsLinearFunction;
    static const bool IsVectorizable = OuterFunc::IsVectorizable && InnerFunc::IsVectorizable;

    NestedFunction_Impl() {
    }
    NestedFunction_Impl(OuterFunc ofunc, Segment<IR, OR, ST> ifunc) {
      this->outer_func = ofunc;
      this->setSegStart(ifunc.SegStart);

      if (ifunc.ORows() != this->outer_func.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in NestedFunction/.eval method !!!\n"
                   "Output Size of InnerFunction (ORows = {0:}) does not match Input Size of OuterFunction "
                   "(IRows = {1:}).\n",
                   ifunc.ORows(),
                   this->outer_func.IRows());
        throw std::invalid_argument("");
      }


      this->setIORows(ifunc.IRows(), this->outer_func.ORows());
      this->set_input_domain(this->IRows(), {ifunc.input_domain()});
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      // typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      const int size = this->outer_func.IRows();

      this->outer_func.compute(x.template segment<InnerFunc::ORC>(this->SegStart, size), fx);
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {

      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      const int size = this->outer_func.IRows();

      this->outer_func.compute_jacobian(x.template segment<InnerFunc::ORC>(this->SegStart, size),
                                        fx,
                                        jx.template middleCols<InnerFunc::ORC>(this->SegStart, size));
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

      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
      MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

      const int size = this->outer_func.IRows();

      this->outer_func.compute_jacobian_adjointgradient_adjointhessian(
          x.template segment<InnerFunc::ORC>(this->SegStart, size),
          fx,
          jx.template middleCols<InnerFunc::ORC>(this->SegStart, size),
          adjgrad.template segment<InnerFunc::ORC>(this->SegStart, size),
          adjhess.template block<InnerFunc::ORC, InnerFunc::ORC>(
              this->SegStart, this->SegStart, this->outer_func.IRows(), size),
          adjvars);
    }

    template<class Target, class JacType, class Assignment>
    inline void accumulate_jacobian(ConstMatrixBaseRef<Target> target_,
                                    ConstMatrixBaseRef<JacType> right,
                                    Assignment assign) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      MatrixBaseRef<JacType> right_ref = right.const_cast_derived();
      // typedef typename Target::Scalar Scalar;
      if constexpr (OR > 0 && IR > 0 && OuterFunc::IRC == -1) {
        Base::accumulate_jacobian(target_, right, assign);
      } else {
        const int size = this->outer_func.IRows();
        this->outer_func.accumulate_jacobian(
            target.template middleCols<InnerFunc::ORC>(this->SegStart, size),
            right_ref.template middleCols<InnerFunc::ORC>(this->SegStart, size),
            assign);
      }
    }

    template<class Target, class JacType, class Assignment>
    inline void accumulate_hessian(ConstMatrixBaseRef<Target> target_,
                                   ConstMatrixBaseRef<JacType> right,
                                   Assignment assign) const {
      MatrixBaseRef<Target> target = target_.const_cast_derived();
      MatrixBaseRef<JacType> right_ref = right.const_cast_derived();
      // typedef typename Target::Scalar Scalar;

      if constexpr (OR > 0 && IR > 0 && OuterFunc::IRC == -1) {
        Base::accumulate_hessian(target_, right, assign);
      } else {
        const int size = this->outer_func.IRows();
        this->outer_func.accumulate_hessian(
            target.template block<InnerFunc::ORC, InnerFunc::ORC>(this->SegStart, this->SegStart, size, size),
            right_ref.template block<InnerFunc::ORC, InnerFunc::ORC>(
                this->SegStart, this->SegStart, size, size),
            assign);
      }
    }
  };


  ////////////////////////////////////////////////////////////////////////
}  // namespace ASSET
