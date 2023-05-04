#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<int OSZ>
  struct Value : VectorFunction<Value<OSZ>, OSZ, OSZ> {
    using Base = VectorFunction<Value<OSZ>, OSZ, OSZ>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    template<class Func>
    using REARGUMENT = Value<OSZ>;

    typename Base::template Output<double> value;

    Value() {
      this->value.setOnes();
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = this->value;
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();

      fx = this->value;
    }

    template<class Func, int FuncIRC, int ii>
    decltype(auto) rearged(const DenseFunctionBase<Func, FuncIRC, ii>& f) const {
      return Value<OSZ>();
    }
  };

}  // namespace ASSET
