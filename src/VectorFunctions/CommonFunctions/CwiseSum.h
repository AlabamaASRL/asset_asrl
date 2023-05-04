#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class Func>
  struct CwiseSum_Impl;

  template<class Func>
  struct CwiseSum : CwiseSum_Impl<CwiseSum<Func>, Func> {
    using Base = CwiseSum_Impl<CwiseSum<Func>, Func>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
  };

  template<class Derived, class Func>
  struct CwiseSum_Impl : VectorFunction<Derived, Func::IRC, 1> {
    using Base = VectorFunction<Derived, Func::IRC, 1>;
    using Base::compute;

    template<class OtherFunc>
    using BaseTemplate = CwiseSum_Impl<CwiseSum<OtherFunc>, OtherFunc>;
    template<class... OtherFunc>
    using DerivedTemplate = CwiseSum<OtherFunc...>;

    using INPUT_DOMAIN = typename Func::INPUT_DOMAIN;
    static const bool IsLinearFunction = Func::IsLinearFunction;
    static const bool IsSegmentOp = Is_Segment<Func>::value || Is_Arguments<Func>::value;

    DENSE_FUNCTION_BASE_TYPES(Base);
    SUB_FUNCTION_IO_TYPES(Func);

    Func func;
    CwiseSum_Impl() {
    }
    CwiseSum_Impl(Func f) : func(std::move(f)) {
      this->setIORows(this->func.IRows(), 1);
      this->set_input_domain(this->IRows(), {this->func.input_domain()});
    }


    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      if constexpr (IsSegmentOp) {
        fx[0] = x.template segment<Func::ORC>(this->func.SegStart, this->func.ORows()).sum();
      } else {
        Func_Output<Scalar> fxv;
        if constexpr (Func::OutputIsDynamic) {
          fxv.resize(this->func.ORows());
        }

        this->func.compute(x, fxv);
        fx[0] = fxv.sum();
      }
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


      if constexpr (IsSegmentOp) {
        fx[0] = x.template segment<Func::ORC>(this->func.SegStart, this->func.ORows()).sum();
        jx.template middleCols<Func::ORC>(this->func.SegStart, this->func.ORows()).setOnes();
      } else {


        auto Impl = [&](auto& fxv, auto& jxv) {
          this->func.compute_jacobian(x, fxv, jxv);
          fx[0] = fxv.sum();

          if constexpr (Func::InputIsDynamic) {
            if (this->SubDomains.size() == 0) {
              jx = jxv.colwise().sum();
            } else {
              for (int i = 0; i < this->SubDomains.size(); i++) {
                int start = this->SubDomains(i, 0);
                int size = this->SubDomains(i, 1);
                jx.middleCols(start, size) = jxv.middleCols(start, size).colwise().sum().transpose();
              }
            }
          } else {

            constexpr int sds = Func::INPUT_DOMAIN::SubDomains.size();

            ASSET::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                  constexpr int start = Func::INPUT_DOMAIN::SubDomains[i.value][0];
                  constexpr int size = Func::INPUT_DOMAIN::SubDomains[i.value][1];
                  jx.middleCols(start, size) = jxv.middleCols(start, size).colwise().sum().transpose();
                });
          }
        };

        ASSET::MemoryManager::allocate_run(
            this->IRows(),
            Impl,
            TempSpec<Func_Output<Scalar>>(this->func.ORows(), 1),
            TempSpec<Func_jacobian<Scalar>>(this->func.ORows(), this->func.IRows()));
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

      if constexpr (IsSegmentOp) {
        fx[0] = x.template segment<Func::ORC>(this->func.SegStart, this->func.ORows()).sum();
        jx.template middleCols<Func::ORC>(this->func.SegStart, this->func.ORows()).setOnes();
        adjgrad.template segment<Func::ORC>(this->func.SegStart, this->func.ORows()).setConstant(adjvars[0]);
      } else {

        auto Impl = [&](auto& fxv, auto& jxv, auto& adjv) {
          adjv.setConstant(adjvars[0]);

          this->func.compute_jacobian_adjointgradient_adjointhessian(x, fxv, jxv, adjgrad, adjhess, adjv);
          fx[0] = fxv.sum();

          if constexpr (Func::InputIsDynamic) {
            if (this->SubDomains.size() == 0) {
              jx = jxv.colwise().sum();
            } else {
              for (int i = 0; i < this->SubDomains.size(); i++) {
                int start = this->SubDomains(i, 0);
                int size = this->SubDomains(i, 1);
                jx.middleCols(start, size) = jxv.middleCols(start, size).colwise().sum().transpose();
              }
            }
          } else {
            constexpr int sds = Func::INPUT_DOMAIN::SubDomains.size();
            ASSET::constexpr_for_loop(
                std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                  constexpr int start = Func::INPUT_DOMAIN::SubDomains[i.value][0];
                  constexpr int size = Func::INPUT_DOMAIN::SubDomains[i.value][1];
                  jx.middleCols(start, size) = jxv.middleCols(start, size).colwise().sum().transpose();
                });
          }
        };

        ASSET::MemoryManager::allocate_run(
            this->IRows(),
            Impl,
            TempSpec<Func_Output<Scalar>>(this->func.ORows(), 1),
            TempSpec<Func_jacobian<Scalar>>(this->func.ORows(), this->func.IRows()),
            TempSpec<Func_Output<Scalar>>(this->func.ORows(), 1));
      }
    }
  };

}  // namespace ASSET
