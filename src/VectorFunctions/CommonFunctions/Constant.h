#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<int IR, int OR>
  struct Constant : VectorFunction<Constant<IR, OR>, IR, OR> {
    using Base = VectorFunction<Constant<IR, OR>, IR, OR>;
    DENSE_FUNCTION_BASE_TYPES(Base)


    static const bool IsLinearFunction = true;
    static const bool IsVectorizable = true;

    Output<double> value;

    Constant(int ir, Output<double> val) {
      this->setIORows(ir, val.size());
      value = val;

      DomainMatrix dmn(2, 1);
      dmn(0, 0) = 0;
      dmn(1, 0) = 0;
      this->set_input_domain(this->IRows(), {dmn});
    }

    Constant() {
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Constant<IR, OR>>(m, name);
      obj.def(py::init<int, Output<double>>());
      Base::DenseBaseBuild(obj);
    }

    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = this->value.template cast<Scalar>();
    }
    template<class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
                                      ConstVectorBaseRef<OutType> fx_,
                                      ConstMatrixBaseRef<JacType> jx_) const {
      typedef typename InType::Scalar Scalar;
      VectorBaseRef<OutType> fx = fx_.const_cast_derived();
      fx = this->value.template cast<Scalar>();
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
      fx = this->value.template cast<Scalar>();
    }
  };

}  // namespace ASSET
