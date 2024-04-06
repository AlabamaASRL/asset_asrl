#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/CommonFunctions.h>
#include <bind/VectorFunctions/CommonFunctions/PythonFunctions.h>
#include <bind/VectorFunctions/common/DenseBaseBuild.h>
#include <bind/VectorFunctions/common/IfElseBuild.h>
#include <bind/VectorFunctions/common/MinMaxBuild.h>
#include <bind/VectorFunctions/common/SegmentBuild.h>
#include <bind/pch.h>

namespace ASSET {

  // Declaration /////////////////////////////////////////////////////////////////////////////////////////////

  template<class Func>
  struct FunctionBinder {
    static void Bind(py::module&, const char*);
  };

  // Specializations /////////////////////////////////////////////////////////////////////////////////////////

  // Constant ================================================================================================

  template<int IR, int OR>
  struct FunctionBinder<Constant<IR, OR>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = Constant<IR, OR>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<int, typename Fun::template Output<double>>());
      DenseBaseBuild<Fun, OR, decltype(obj)>(obj);
    }
  };

  // DotProduct ==============================================================================================

  template<class Func1, class Func2>
  struct FunctionBinder<FunctionDotProduct<Func1, Func2>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = FunctionDotProduct<Func1, Func2>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<Func1, Func2>());
      DenseBaseBuild<Fun, Fun::OR, decltype(obj)>(obj);
    }
  };

  // NestedFunction ==========================================================================================

  template<class OuterFunc, class InnerFunc>
  struct FunctionBinder<NestedFunction<OuterFunc, InnerFunc>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = NestedFunction<OuterFunc, InnerFunc>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<>());
      obj.def(py::init<OuterFunc, InnerFunc>());
      DenseBaseBuild<Fun, OuterFunc::OR, decltype(obj)>(obj);
    }
  };

  // NormalizedPower =========================================================================================

  template<int IR, int PW>
  struct FunctionBinder<NormalizedPower<IR, PW>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = NormalizedPower<IR, PW>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<int>());
      if constexpr (IR > 0) {
        obj.def(py::init<>());
      }
      DenseBaseBuild<Fun, 1, decltype(obj)>(obj);
    }
  };

  // Normalized ==============================================================================================

  template<int IR>
  struct FunctionBinder<Normalized<IR>> {
    static void Bind(py::module& m, const char* name) {
      FunctionBinder<NormalizedPower<IR, 1>>::Bind(m, name);
    }
  };

  // Norms ===================================================================================================

  template<int IR, int PW>
  struct FunctionBinder<NormPower<IR, PW>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = NormPower<IR, PW>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<int>());
      if constexpr (IR > 0) {
        obj.def(py::init<>());
      }
      DenseBaseBuild<Fun, 1, decltype(obj)>(obj);
    }
  };

  template<int IR>
  struct FunctionBinder<Norm<IR>> {
    static void Bind(py::module& m, const char* name) {
      FunctionBinder<NormPower<IR, 1>>::Bind(m, name);
    }
  };

  template<int IR>
  struct FunctionBinder<SquaredNorm<IR>> {
    static void Bind(py::module& m, const char* name) {
      FunctionBinder<NormPower<IR, 2>>::Bind(m, name);
    }
  };

  template<int IR>
  struct FunctionBinder<InverseNorm<IR>> {
    static void Bind(py::module& m, const char* name) {
      FunctionBinder<NormPower<IR, -1>>::Bind(m, name);
    }
  };

  template<int IR>
  struct FunctionBinder<InverseSquaredNorm<IR>> {
    static void Bind(py::module& m, const char* name) {
      FunctionBinder<NormPower<IR, -2>>::Bind(m, name);
    }
  };

  template<int IR, int PW>
  struct FunctionBinder<InverseNormPower<IR, PW>> {
    static void Bind(py::module& m, const char* name) {
      FunctionBinder<NormPower<IR, -PW>>::Bind(m, name);
    }
  };

  // ParsedInput =============================================================================================

  template<class Func, int IR, int OR>
  struct FunctionBinder<ParsedInput<Func, IR, OR>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = ParsedInput<Func, IR, OR>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<Func, const typename Fun::template Func_Input<int>&, int>());
      DenseBaseBuild<Fun, OR, decltype(obj)>(obj);
    }
  };

  // Arguments ===============================================================================================

  template<int IR_OR>
  struct FunctionBinder<Arguments<IR_OR>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = Arguments<IR_OR>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<int>());
      obj.def("Constant", [](const Fun& a, Eigen::VectorXd v) {
        return GenericFunction<-1, -1>(Constant<-1, -1>(a.IRows(), v));
      });
      obj.def("Constant", [](const Fun& a, double v) {
        Eigen::Matrix<double, 1, 1> vv;
        vv[0] = v;
        return GenericFunction<-1, 1>(Constant<-1, 1>(a.IRows(), vv));
      });

      DenseBaseBuild<Fun, IR_OR, decltype(obj)>(obj);
      SegmentBuild<Fun, IR_OR, IR_OR, decltype(obj)>(obj);
    }
  };

  // Segment =================================================================================================

  template<int IR, int OR, int ST>
  struct FunctionBinder<Segment<IR, OR, ST>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = Segment<IR, OR, ST>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<int, int, int>());
      DenseBaseBuild<Fun, OR, decltype(obj)>(obj);
      SegmentBuild<Fun, IR, OR, decltype(obj)>(obj);
    }
  };

  // TwoFunctionSum ==========================================================================================

  template<class Func1, class Func2>
  struct FunctionBinder<TwoFunctionSum<Func1, Func2>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = TwoFunctionSum<Func1, Func2>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<Func1, Func2>());
      DenseBaseBuild<Fun, Fun::OR, decltype(obj)>(obj);
    }
  };

  // FunctionDifference ======================================================================================

  template<class Func1, class Func2>
  struct FunctionBinder<FunctionDifference<Func1, Func2>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = FunctionDifference<Func1, Func2>;

      auto obj = py::class_<Fun>(m, name);
      obj.def(py::init<Func1, Func2>());
      DenseBaseBuild<Fun, Fun::OR, decltype(obj)>(obj);
    }
  };

  // Python ==================================================================================================

  template<int IR, int OR>
  struct FunctionBinder<PyVectorFunction<IR, OR>> {
    static void Bind(py::module& m, const char* name) {
      using Fun = PyVectorFunction<IR, OR>;
      using FType = std::function<typename Fun::template Output<double>(
          ConstEigenRef<typename Fun::template Input<double>>, py::detail::args_proxy)>;

      auto obj = py::class_<Fun>(m, name);

      if constexpr (OR != 1) {
        obj.def(py::init<int, int, const FType&, double, double, py::tuple>(),
                py::arg("IRows"),
                py::arg("ORows"),
                py::arg("Func"),
                py::arg("Jstepsize") = 1.0e-6,
                py::arg("Hstepsize") = 1.0e-4,
                py::arg("args") = py::tuple());
      } else {
        obj.def(py::init<int, const FType&, double, double, py::tuple>(),
                py::arg("IRows"),
                py::arg("Func"),
                py::arg("Jstepsize") = 1.0e-6,
                py::arg("Hstepsize") = 1.0e-4,
                py::arg("args") = py::tuple());
      }

      DenseBaseBuild<Fun, OR, decltype(obj)>(obj);
    }
  };

  // GenericConditional ======================================================================================

  template<int IR>
  struct FunctionBinder<GenericConditional<IR>> {
    static void Bind(py::module& m, const char* name) {

      using GenCon = GenericConditional<IR>;

      auto obj = py::class_<GenCon>(m, name);

      obj.def("compute", [](const GenCon& a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); });

      obj.def(
          "__and__",
          [](const GenCon& a, const GenCon& b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ANDFlag, b));
          },
          py::is_operator());

      obj.def(
          "__or__",
          [](const GenCon& a, const GenCon& b) {
            return GenCon(ConditionalStatement<GenCon, GenCon>(a, ConditionalFlags::ORFlag, b));
          },
          py::is_operator());

      IfElseBuild(obj);
    }
  };

  // GenericComparative ======================================================================================

  template<int IR>
  struct FunctionBinder<GenericComparative<IR>> {
    static void Bind(py::module& m, const char* name) {
      using GenComp = GenericComparative<IR>;

      auto obj = py::class_<GenComp>(m, "Comparative");

      obj.def("compute", [](const GenComp& a, ConstEigenRef<Eigen::VectorXd> x) { return a.compute(x); });

      MinMaxBuild(obj);
    }
  };

}  // namespace ASSET
