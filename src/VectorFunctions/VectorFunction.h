#pragma once

#include "DenseDerivatives.h"
#include "DenseDifferentiation/DenseFDiffCentArray.h"
#include "DenseDifferentiation/DenseFDiffFwd.h"
//#include "DenseDifferentiation/DenseAutodiffFwd.h"
#include "Utils/FunctionReturnType.h"

namespace ASSET {

  template<class Derived,
           int IR,
           int OR,
           DenseDerivativeModes Jm = DenseDerivativeModes::Analytic,
           DenseDerivativeModes Hm = DenseDerivativeModes::Analytic>
  struct VectorFunction : DenseDerivatives<Derived, IR, OR, Jm, Hm> {
    using Base = DenseDerivatives<Derived, IR, OR, Jm, Hm>;
    DENSE_FUNCTION_BASE_TYPES(Base)
  };

  template<class Derived, class ExprImpl, class... Ts>
  struct VectorExpression : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    using Base = typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    VectorExpression(Ts... ts) : Base(ExprImpl::Definition(ts...)) {
    }
    VectorExpression() {};

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Derived>(m, name).def(py::init<Ts...>());
      Base::DenseBaseBuild(obj);
    }

    /////////////////////////////////////
    void InitExpression(Ts... ts) {
      *this = Base(ExprImpl::Definition(ts...));
    }
  };

  template<class Derived, class ExprImpl>
  struct VectorExpression<Derived, ExprImpl>
      : return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived> {
    using Base = typename return_type_t<decltype(&ExprImpl::Definition)>::template AsBaseClass<Derived>;

    using Base::Base;
    VectorExpression() : Base(ExprImpl::Definition()) {
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Derived>(m, name).def(py::init<>());
      Base::DenseBaseBuild(obj);
    }

    /////////////////////////////////////
    void InitExpression() {
      *this = Base(ExprImpl::Definition());
    }
  };

#define BUILD_FROM_EXPRESSION(NAME, IMPL, ...)              \
  struct NAME : VectorExpression<NAME, IMPL, __VA_ARGS__> { \
    using Base = VectorExpression<NAME, IMPL, __VA_ARGS__>; \
    using Base::Base;                                       \
  };

}  // namespace ASSET
