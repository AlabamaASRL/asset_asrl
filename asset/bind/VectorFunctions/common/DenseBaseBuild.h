#pragma once

#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int OR, class PyClass>
  void DenseBaseBuild(PyClass& obj) {

    using Gen = GenericFunction<-1, -1>;

    obj.def("IRows", &Derived::IRows);
    obj.def("ORows", &Derived::ORows);
    obj.def("name", &Derived::name);
    obj.def("compute", [](const Derived& func, ConstEigenRef<typename Derived::template Input<double>> x) {
      if (x.size() != func.IRows())
        throw std::invalid_argument("Incorrectly sized input to function");
      typename Derived::template Output<double> fx(func.ORows());
      fx.setZero();
      func.derived().compute(x, fx);
      return fx;
    });
    obj.def("__call__", [](const Derived& func, ConstEigenRef<typename Derived::template Input<double>> x) {
      if (x.size() != func.IRows())
        throw std::invalid_argument("Incorrectly sized input to function");
      typename Derived::template Output<double> fx(func.ORows());
      fx.setZero();
      func.derived().compute(x, fx);
      return fx;
    });

    obj.def("jacobian", [](const Derived& func, ConstEigenRef<typename Derived::template Input<double>> x) {
      if (x.size() != func.IRows())
        throw std::invalid_argument("Incorrectly sized input to function");
      typename Derived::template Jacobian<double> jx(func.ORows(), func.IRows());
      jx.setZero();
      func.derived().jacobian(x, jx);
      return jx;
    });
    obj.def("adjointgradient",
            [](const Derived& func,
               ConstEigenRef<typename Derived::template Input<double>> x,
               ConstEigenRef<typename Derived::template Output<double>> lm) {
              if (x.size() != func.IRows())
                throw std::invalid_argument("Incorrectly sized input to function");
              if (lm.size() != func.ORows())
                throw std::invalid_argument("Incorrectly sized multiplier input to function");

              typename Derived::template Gradient<double> ax(func.IRows());
              ax.setZero();
              func.derived().adjointgradient(x, ax, lm);
              return ax;
            });
    obj.def("adjointhessian",
            [](const Derived& func,
               ConstEigenRef<typename Derived::template Input<double>> x,
               ConstEigenRef<typename Derived::template Output<double>> lm) {
              if (x.size() != func.IRows())
                throw std::invalid_argument("Incorrectly sized input to function");
              if (lm.size() != func.ORows())
                throw std::invalid_argument("Incorrectly sized multiplier input to function");

              typename Derived::template Hessian<double> hx(func.IRows(), func.IRows());
              hx.setZero();
              func.derived().adjointhessian(x, hx, lm);
              return hx;
            });

    obj.def("computeall",
            [](const Derived& func,
               ConstEigenRef<typename Derived::template Input<double>> x,
               ConstEigenRef<typename Derived::template Output<double>> lm) {
              if (x.size() != func.IRows())
                throw std::invalid_argument("Incorrectly sized input to function");
              if (lm.size() != func.ORows())
                throw std::invalid_argument("Incorrectly sized multiplier input to function");

              typename Derived::template Output<double> fx(func.ORows());

              typename Derived::template Jacobian<double> jx(func.ORows(), func.IRows());
              typename Derived::template Gradient<double> gx(func.IRows());
              typename Derived::template Hessian<double> hx(func.IRows(), func.IRows());

              fx.setZero();
              jx.setZero();
              gx.setZero();
              hx.setZero();

              func.derived().compute_jacobian_adjointgradient_adjointhessian(x, fx, jx, gx, hx, lm);
              return std::tuple {fx, jx, gx, hx};
            });

    obj.def("rpt", &Derived::rpt);
    obj.def("vf", &Derived::template MakeGeneric<GenericFunction<-1, -1>>);

    if constexpr (OR == 1) {
      obj.def("sf", &Derived::template MakeGeneric<GenericFunction<-1, 1>>);
    }
  }

}  // namespace ASSET
