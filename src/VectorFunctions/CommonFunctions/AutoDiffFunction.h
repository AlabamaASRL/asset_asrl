#pragma once

#include "VectorFunction.h"

namespace ASSET {

  template<class Func>
  struct ADFun : VectorFunction<ADFun<Func>, Func::IRC, Func::ORC, FDiffCentArray, FDiffFwd> {
    using Base = VectorFunction<ADFun<Func>, Func::IRC, Func::ORC, FDiffCentArray, FDiffFwd>;
    DENSE_FUNCTION_BASE_TYPES(Base)

    Func func;
    ADFun(Func f) : func(std::move(f)) {
      this->setIORows(this->func.IRows(), this->func.ORows());
    }
    template<class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_) const {
      this->func.compute(x, fx_);
    }

    void test() {
      Input<double> x;
      x.setRandom();
      return TestDerivs(x);
    };
    void test(Input<double> x) {
      return TestDerivs(x);
    };
    void TestDerivs(Input<double> x) {
      Output<double> l;
      l.setOnes();

      std::cout << "Jacobian Error" << std::endl;
      std::cout << this->jacobian(x) - this->func.jacobian(x) << std::endl << std::endl;
      ;
      std::cout << "Hessian Error" << std::endl;
      std::cout << (this->adjointhessian(x, l) - this->func.adjointhessian(x, l)) << std::endl << std::endl;
      ;
    }
  };

}  // namespace ASSET
