#pragma once
#include "CommonFunctions/ExpressionFwdDeclarations.h"

namespace ASSET {


  template<class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
  auto crossProduct(const DenseFunctionBase<Func1, IR1, OR1>& f1,
                    const DenseFunctionBase<Func2, IR2, OR2>& f2) {
    return FunctionCrossProduct<Func1, Func2>(f1.derived(), f2.derived());
  }

  template<class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
  auto quatProduct(const DenseFunctionBase<Func1, IR1, OR1>& f1,
                   const DenseFunctionBase<Func2, IR2, OR2>& f2) {
    return FunctionQuatProduct<Func1, Func2>(f1.derived(), f2.derived());
  }

  template<class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
  auto dotProduct(const DenseFunctionBase<Func1, IR1, OR1>& f1,
                  const DenseFunctionBase<Func2, IR2, OR2>& f2) {
    return FunctionDotProduct<Func1, Func2>(f1.derived(), f2.derived());
  }

  template<class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
  auto cwiseProduct(const DenseFunctionBase<Func1, IR1, OR1>& f1,
                    const DenseFunctionBase<Func2, IR2, OR2>& f2) {
    return CwiseFunctionProduct<Func1, Func2>(f1.derived(), f2.derived());
  }
  template<class Func1, int IR1, int OR1, class Func2, int IR2, int OR2>
  auto cwiseQuotient(const DenseFunctionBase<Func1, IR1, OR1>& f1,
                     const DenseFunctionBase<Func2, IR2, OR2>& f2) {
    return CwiseFunctionProduct<Func1, CwiseInverse<Func2>>(f1.derived(), CwiseInverse<Func2>(f2.derived()));
  }


}  // namespace ASSET