#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<int IR, int EL1, int... ELS>
  struct Elements : VectorFunction<Elements<IR, EL1, ELS...>, IR, 1 + sizeof...(ELS)> {
    using Base = VectorFunction<Elements<IR, EL1, ELS...>, IR, 1 + sizeof...(ELS)>;
    DENSE_FUNCTION_BASE_TYPES(Base);
    using Base::compute;
    static const std::tuple<std::integral_constant<int, EL1>, std::integral_constant<int, ELS>...> elements;
    static const int num_elements = 1 + sizeof...(ELS);

    Elements() {
    }
    Elements(int irows) {
      this->setIORows(irows, num_elements);
    }

    static const bool IsLinearFunction = true;
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      ASSET::tuple_for_loop(elements, [&](const auto& ele, auto i) { fx[i] = x[ele.value]; });
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
      ASSET::tuple_for_loop(elements, [&](const auto& ele, int i) {
        fx[i] = x[ele.value];
        jx(i, ele.value) = 1.0;
      });
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

      ASSET::tuple_for_loop(elements, [&](const auto& ele, int i) {
        fx[i] = x[ele.value];
        jx(i, ele.value) = 1.0;
        adjgrad[ele.value] = adjvars[i];
      });
    }
  };

}  // namespace ASSET
