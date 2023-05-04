#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class VecFunc, class ScalFunc>
  struct VectorScalarFunctionProduct_Impl;

  template<class VecFunc, class ScalFunc>
  struct VectorScalarFunctionProduct
      : VectorScalarFunctionProduct_Impl<VectorScalarFunctionProduct<VecFunc, ScalFunc>, VecFunc, ScalFunc> {
    using Base =
        VectorScalarFunctionProduct_Impl<VectorScalarFunctionProduct<VecFunc, ScalFunc>, VecFunc, ScalFunc>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
  };

  template<class Derived, class VecFunc, class ScalFunc>
  struct VectorScalarFunctionProduct_Impl
      : VectorFunction<Derived, SZ_MAX<VecFunc::IRC, ScalFunc::IRC>::value, VecFunc::ORC> {
    using Base = VectorFunction<Derived, SZ_MAX<VecFunc::IRC, ScalFunc::IRC>::value, VecFunc::ORC>;
    DENSE_FUNCTION_BASE_TYPES(Base);

    SUB_FUNCTION_IO_TYPES(VecFunc);
    SUB_FUNCTION_IO_TYPES(ScalFunc);
    using Base::compute;

    VecFunc vectorfunc;
    ScalFunc scalarfunc;

    using INPUT_DOMAIN =
        CompositeDomain<Base::IRC, typename VecFunc::INPUT_DOMAIN, typename ScalFunc::INPUT_DOMAIN>;
    static const bool IsVectorizable = VecFunc::IsVectorizable && ScalFunc::IsVectorizable;

    VectorScalarFunctionProduct_Impl() {
    }
    VectorScalarFunctionProduct_Impl(VecFunc f1, ScalFunc f2)
        : vectorfunc(std::move(f1)), scalarfunc(std::move(f2)) {
      int irtemp = std::max(this->vectorfunc.IRows(), this->scalarfunc.IRows());
      this->setIORows(irtemp, this->vectorfunc.ORows());

      this->set_input_domain(this->IRows(), {scalarfunc.input_domain(), vectorfunc.input_domain()});

      if (this->scalarfunc.IRows() != this->vectorfunc.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in VectorScalarFunctionProduct/ * method !!!\n"
                   "Input Size of VectorFunc (IRows = {0:}) does not match Input Size of ScalarFunc (IRows = "
                   "{1:}).\n",
                   this->vectorfunc.IRows(),
                   this->scalarfunc.IRows());
        throw std::invalid_argument("");
      }
    }

    static const bool vectorfunc_is_segment = Is_Segment<VecFunc>::value || Is_ScaledSegment<VecFunc>::value;
    static const bool scalarfunc_is_segment =
        Is_Segment<ScalFunc>::value || Is_ScaledSegment<ScalFunc>::value;
    static const bool is_prod_of_segments = vectorfunc_is_segment && scalarfunc_is_segment;

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      this->vectorfunc.compute(x, fx_);
      Vector1<Scalar> fxs;
      this->scalarfunc.compute(x, fxs);
      fx *= fxs[0];
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      auto Impl = [&](auto& jxs) {
        Vector1<Scalar> fxs;
        this->vectorfunc.compute_jacobian(x, fx_, jx_);
        this->scalarfunc.compute_jacobian(x, fxs, jxs);
        this->vectorfunc.scale_jacobian(jx, fxs[0]);
        this->scalarfunc.right_jacobian_product(
            jx_, fx, jxs, PlusEqualsAssignment(), std::bool_constant<false>());
        fx *= fxs[0];
      };

      const int irows = this->scalarfunc.IRows();
      using JType = ScalFunc_jacobian<Scalar>;
      MemoryManager::allocate_run(irows, Impl, TempSpec<JType>(1, irows));
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

      //////////////////////////////////////////

      auto Impl = [&](auto& jxs, auto& gxs, auto& hxs, auto& adjtemp) {
        Vector1<Scalar> fxs;
        adjtemp = adjvars;
        if constexpr (ScalFunc::IsLinearFunction && false) {
          this->scalarfunc.compute(x, fxs);
          adjtemp *= fxs[0];
          fxs[0] = 0;
        }

        this->vectorfunc.compute_jacobian_adjointgradient_adjointhessian(
            x, fx, jx, adjgrad, adjhess, adjtemp);

        ScalFunc_Output<Scalar> ls;
        ls[0] = fx.dot(adjvars);

        this->scalarfunc.compute_jacobian_adjointgradient_adjointhessian(x, fxs, jxs, gxs, hxs, ls);

        if constexpr (ScalFunc::IsLinearFunction && false) {  // This is dangerous, might divide by zeros
          this->vectorfunc.scale_gradient(adjgrad, Scalar(1.0 / fxs[0]));
        } else {
          this->vectorfunc.scale_hessian(adjhess, fxs[0]);
        }

        this->scalarfunc.accumulate_hessian(adjhess, hxs, PlusEqualsAssignment());

        if constexpr (ScalFunc::InputIsDynamic) {
          const int sds = this->scalarfunc.SubDomains.cols();

          if (sds == 0) {
            hxs.setZero();
          } else {
            for (int i = 0; i < sds; i++) {
              int start = this->scalarfunc.SubDomains(0, i);
              int size = this->scalarfunc.SubDomains(1, i);
              hxs.middleCols(start, size).setZero();
            }
          }
        } else {
          constexpr int sds = ScalFunc::INPUT_DOMAIN::SubDomains.size();
          ASSET::constexpr_for_loop(
              std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                constexpr int start = ScalFunc::INPUT_DOMAIN::SubDomains[i.value][0];
                constexpr int size = ScalFunc::INPUT_DOMAIN::SubDomains[i.value][1];
                hxs.template middleCols<size>(start, size).setZero();
              });
        }

        this->scalarfunc.right_jacobian_product(
            hxs, adjgrad, jxs, DirectAssignment(), std::bool_constant<false>());

        if constexpr (ScalFunc::InputIsDynamic) {
          const int sds = this->scalarfunc.SubDomains.cols();

          if (sds == 0) {
            adjhess += hxs + hxs.transpose();

          } else {
            for (int i = 0; i < sds; i++) {
              int start = this->scalarfunc.SubDomains(0, i);
              int size = this->scalarfunc.SubDomains(1, i);
              adjhess.middleCols(start, size) += hxs.middleCols(start, size);
              adjhess.middleRows(start, size) += hxs.middleCols(start, size).transpose();
            }
          }

        } else {
          constexpr int sds = ScalFunc::INPUT_DOMAIN::SubDomains.size();

          ASSET::constexpr_for_loop(
              std::integral_constant<int, 0>(), std::integral_constant<int, sds>(), [&](auto i) {
                constexpr int start = ScalFunc::INPUT_DOMAIN::SubDomains[i.value][0];
                constexpr int size = ScalFunc::INPUT_DOMAIN::SubDomains[i.value][1];
                adjhess.template middleCols<size>(start, size) += hxs.template middleCols<size>(start, size);
                adjhess.template middleRows<size>(start, size) +=
                    hxs.template middleCols<size>(start, size).transpose();
              });
        }

        this->vectorfunc.scale_jacobian(jx, fxs[0]);

        this->vectorfunc.scale_gradient(adjgrad, fxs[0]);

        this->scalarfunc.accumulate_gradient(adjgrad, gxs, PlusEqualsAssignment());

        this->scalarfunc.right_jacobian_product(
            jx_, fx, jxs, PlusEqualsAssignment(), std::bool_constant<false>());
        fx *= fxs[0];
      };


      const int irows = this->scalarfunc.IRows();
      const int orows = this->ORows();
      const int crit_size = std::max(irows, orows);
      using JType = ScalFunc_jacobian<Scalar>;
      using GType = ScalFunc_gradient<Scalar>;
      using HType = ScalFunc_hessian<Scalar>;
      using AType = Output<Scalar>;

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<JType>(1, irows),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<HType>(irows, irows),
                                  TempSpec<AType>(orows, 1));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET
