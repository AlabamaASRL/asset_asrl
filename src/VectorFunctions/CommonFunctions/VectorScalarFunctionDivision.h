#pragma once
#include "VectorFunction.h"

namespace ASSET {

  template<class Derived, class VecFunc, class ScalFunc>
  struct VectorScalarFunctionDivision_Impl;

  template<class VecFunc, class ScalFunc>
  struct VectorScalarFunctionDivision
      : VectorScalarFunctionDivision_Impl<VectorScalarFunctionDivision<VecFunc, ScalFunc>,
                                          VecFunc,
                                          ScalFunc> {
    using Base =
        VectorScalarFunctionDivision_Impl<VectorScalarFunctionDivision<VecFunc, ScalFunc>, VecFunc, ScalFunc>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::Base;
  };

  template<class Derived, class VecFunc, class ScalFunc>
  struct VectorScalarFunctionDivision_Impl
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

    VectorScalarFunctionDivision_Impl() {
    }
    VectorScalarFunctionDivision_Impl(VecFunc f1, ScalFunc f2)
        : vectorfunc(std::move(f1)), scalarfunc(std::move(f2)) {
      int irtemp = std::max(this->vectorfunc.IRows(), this->scalarfunc.IRows());
      this->setIORows(irtemp, this->vectorfunc.ORows());

      this->set_input_domain(this->IRows(), {scalarfunc.input_domain(), vectorfunc.input_domain()});

      if (this->scalarfunc.IRows() != this->vectorfunc.IRows()) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Math Error in VectorScalarFunctionDivision/ * method !!!\n"
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

      Vector1<Scalar> fxs;
      this->vectorfunc.compute(x, fx);
      this->scalarfunc.compute(x, fxs);

      Scalar ifxs = 1.0 / fxs[0];

      fx *= ifxs;

      // f/g
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();

      auto Impl = [&](auto& fxt, auto& jxs) {
        // f'/g - fg'/g^2


        Vector1<Scalar> fxs;
        this->vectorfunc.compute_jacobian(x, fx, jx);
        this->scalarfunc.compute_jacobian(x, fxs, jxs);

        Scalar ifxs = 1.0 / fxs[0];


        this->vectorfunc.scale_jacobian(jx, ifxs);

        fxt = -fx * (ifxs * ifxs);
        fx *= ifxs;

        this->scalarfunc.right_jacobian_product(
            jx, fxt, jxs, PlusEqualsAssignment(), std::bool_constant<false>());
      };

      const int irows = this->scalarfunc.IRows();
      const int orows = this->ORows();

      using FType = Output<Scalar>;
      using JType = ScalFunc_jacobian<Scalar>;
      MemoryManager::allocate_run(irows, Impl, TempSpec<FType>(orows, 1), TempSpec<JType>(1, irows));
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

      auto Impl = [&](auto& fxt, auto& jxs, auto& gxs, auto& gtmp, auto& hxs) {
        this->vectorfunc.compute_jacobian_adjointgradient_adjointhessian(
            x, fx, jx, adjgrad, adjhess, adjvars);

        Vector1<Scalar> fxs;
        Vector1<Scalar> ls;
        ls[0] = -fx.dot(adjvars);

        this->scalarfunc.compute_jacobian_adjointgradient_adjointhessian(x, fxs, jxs, gxs, hxs, ls);

        Scalar ifxs = 1.0 / fxs[0];


        this->scalarfunc.scale_hessian(hxs, Scalar(ifxs * ifxs));
        this->scalarfunc.scale_gradient(gxs, Scalar(ifxs * ifxs));


        this->vectorfunc.scale_hessian(adjhess, ifxs);
        this->vectorfunc.scale_gradient(adjgrad, ifxs);

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


        gtmp = (gxs + adjgrad) * (-ifxs);


        this->scalarfunc.right_jacobian_product(
            hxs, gtmp, jxs, DirectAssignment(), std::bool_constant<false>());


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


        this->vectorfunc.scale_jacobian(jx, ifxs);
        this->scalarfunc.accumulate_gradient(adjgrad, gxs, PlusEqualsAssignment());

        fxt = -fx * (ifxs * ifxs);
        fx *= ifxs;

        this->scalarfunc.right_jacobian_product(
            jx, fxt, jxs, PlusEqualsAssignment(), std::bool_constant<false>());
      };


      const int irows = this->scalarfunc.IRows();
      const int orows = this->ORows();
      const int crit_size = std::max(irows, orows);

      using FType = Output<Scalar>;
      using JType = ScalFunc_jacobian<Scalar>;
      using GType = ScalFunc_gradient<Scalar>;
      using HType = ScalFunc_hessian<Scalar>;

      MemoryManager::allocate_run(crit_size,
                                  Impl,
                                  TempSpec<FType>(orows, 1),
                                  TempSpec<JType>(1, irows),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<GType>(irows, 1),
                                  TempSpec<HType>(irows, irows));
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  };

}  // namespace ASSET
