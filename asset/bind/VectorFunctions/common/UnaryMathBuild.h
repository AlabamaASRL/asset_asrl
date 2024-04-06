#pragma once

#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void UnaryMathBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    if constexpr (OR != 1) {

      if constexpr (!std::is_same<PyClass, py::module>::value) {
        obj.def("sum", [](const Derived& a) { return GenS(a.sum()); });
      }

      obj.def("normalized_power3", [](const Derived& a, typename Derived::template Output<double> b) {
        return Gen((a + b).template normalized_power<3>());
      });
      obj.def("normalized_power3",
              [](const Derived& a, typename Derived::template Output<double> b, double s) {
                return Gen(((a + b).template normalized_power<3>()) * s);
              });
      ////////////////////////////////////////////////////////////////////
      if constexpr (OR > 0) {  // already constant size

        obj.def("norm", [](const Derived& a) { return GenS(a.norm()); });
        obj.def("squared_norm", [](const Derived& a) { return GenS(a.squared_norm()); });
        obj.def("cubed_norm", [](const Derived& a) { return GenS(a.template norm_power<3>()); });

        obj.def("inverse_norm", [](const Derived& a) { return GenS(a.inverse_norm()); });
        obj.def("inverse_squared_norm", [](const Derived& a) { return GenS(a.inverse_squared_norm()); });
        obj.def("inverse_cubed_norm",
                [](const Derived& a) { return GenS(a.template inverse_norm_power<3>()); });
        obj.def("inverse_four_norm",
                [](const Derived& a) { return GenS(a.template inverse_norm_power<4>()); });

        obj.def("normalized", [](const Derived& a) { return Gen(a.normalized()); });

        obj.def("normalized_power2", [](const Derived& a) { return Gen(a.template normalized_power<2>()); });

        obj.def("normalized_power4", [](const Derived& a) { return Gen(a.template normalized_power<4>()); });
        obj.def("normalized_power5", [](const Derived& a) { return Gen(a.template normalized_power<5>()); });

        obj.def("normalized_power3", [](const Derived& a) { return Gen(a.template normalized_power<3>()); });

      } else {
        /// Try to fit these to constant size 2,3 if possible

        auto SizeSwitch = [](const auto& fun, auto Lam) {
          int orr = fun.ORows();
          // if (orr == 2)     return Lam(fun, std::integral_constant<int, 2>());
          // else
          if (orr == 3)
            return Lam(fun, std::integral_constant<int, 3>());
          return Lam(fun, std::integral_constant<int, -1>());
        };

        obj.def("norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(Norm<size.value>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("squared_norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(SquaredNorm<size.value>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("cubed_norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(NormPower<size.value, 3>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("inverse_norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(InverseNorm<size.value>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("inverse_squared_norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(InverseSquaredNorm<size.value>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("inverse_cubed_norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(InverseNormPower<size.value, 3>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("inverse_four_norm", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return GenS(InverseNormPower<size.value, 4>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("normalized", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return Gen(Normalized<size.value>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("normalized_power2", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return Gen(NormalizedPower<size.value, 2>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("normalized_power3", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return Gen(NormalizedPower<size.value, 3>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("normalized_power4", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return Gen(NormalizedPower<size.value, 4>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
        obj.def("normalized_power5", [SizeSwitch](const Derived& fun) {
          auto Lam = [](const Derived& funt, auto size) {
            return Gen(NormalizedPower<size.value, 5>(funt.ORows()).eval(funt));
          };
          return SizeSwitch(fun, Lam);
        });
      }
    }

    if constexpr (OR == 1) {

      obj.def("sin", [](const Derived& a) { return GenS(a.Sin()); });
      obj.def("cos", [](const Derived& a) { return GenS(a.Cos()); });
      obj.def("tan", [](const Derived& a) { return GenS(a.Tan()); });
      obj.def("sqrt", [](const Derived& a) { return GenS(a.Sqrt()); });
      obj.def("exp", [](const Derived& a) { return GenS(a.Exp()); });
      obj.def("log", [](const Derived& e) { return GenS(CwiseLog<Derived>(e)); });
      obj.def("squared", [](const Derived& a) { return GenS(a.Square()); });
      obj.def("arcsin", [](const Derived& e) { return GenS(CwiseArcSin<Derived>(e)); });
      obj.def("arccos", [](const Derived& e) { return GenS(CwiseArcCos<Derived>(e)); });
      obj.def("arctan", [](const Derived& e) { return GenS(CwiseArcTan<Derived>(e)); });

      obj.def("sinh", [](const Derived& e) { return GenS(CwiseSinH<Derived>(e)); });
      obj.def("cosh", [](const Derived& e) { return GenS(CwiseCosH<Derived>(e)); });
      obj.def("tanh", [](const Derived& e) { return GenS(CwiseTanH<Derived>(e)); });

      obj.def("arcsinh", [](const Derived& e) { return GenS(CwiseArcSinH<Derived>(e)); });
      obj.def("arccosh", [](const Derived& e) { return GenS(CwiseArcCosH<Derived>(e)); });
      obj.def("arctanh", [](const Derived& e) { return GenS(CwiseArcTanH<Derived>(e)); });

      obj.def("__abs__", [](const Derived& e) { return GenS(CwiseAbs<Derived>(e)); });
      obj.def("sign", [](const Derived& e) { return GenS(SignFunction<Derived>(e)); });

      obj.def("pow", [](const Derived& e, double power) { return GenS(CwisePow<Derived>(e, power)); });

      if constexpr (!std::is_same<PyClass, py::module>::value) {
        obj.def(
            "__pow__",
            [](const Derived& a, double b) { return GenS(CwisePow<Derived>(a, b)); },
            py::is_operator());
        obj.def(
            "__pow__",
            [](const Derived& a, int b) {
              if (b == 1)
                return GenS(a);
              else if (b == 2)
                return GenS(a.Square());
              return GenS(CwisePow<Derived>(a, b));
            },
            py::is_operator());
      }
    }
  }

}  // namespace ASSET
