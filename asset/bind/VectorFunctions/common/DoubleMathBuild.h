#pragma once

#include <ASSET/VectorFunctions/OperatorOverloads.h>
#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void DoubleMathBuild(PyClass& obj) {

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    // Prevents numpy from overriding __radd__ and __rmul__
    obj.attr("__array_ufunc__") = py::none();

    obj.def(
        "__add__",
        [](const Derived& a, typename Derived::template Output<double> b) { return BinGen(a + b); },
        py::is_operator());
    obj.def(
        "__radd__",
        [](const Derived& a, typename Derived::template Output<double> b) { return BinGen(a + b); },
        py::is_operator());
    obj.def(
        "__sub__",
        [](const Derived& a, typename Derived::template Output<double> b) { return BinGen(a - b); },
        py::is_operator());
    obj.def(
        "__rsub__",
        [](const Derived& a, typename Derived::template Output<double> b) { return BinGen(b - a); },
        py::is_operator());

    obj.def("__mul__", [](const Derived& a, double b) { return BinGen(a * b); }, py::is_operator());
    obj.def("__rmul__", [](const Derived& a, double b) { return BinGen(a * b); }, py::is_operator());

    obj.def("__neg__", [](const Derived& a) { return BinGen(a * (-1.0)); }, py::is_operator());

    obj.def(
        "__truediv__", [](const Derived& a, double b) { return BinGen(a * (1.0 / b)); }, py::is_operator());
    obj.def(
        "__truediv__",
        [](const Derived& a, const Segment<-1, 1, -1>& b) { return BinGen(a / b); },
        py::is_operator());

    if constexpr (OR == 1) {  // Scalars
      obj.def("__add__", [](const Derived& a, double b) { return BinGen(a + b); }, py::is_operator());
      obj.def("__radd__", [](const Derived& a, double b) { return BinGen(a + b); }, py::is_operator());
      obj.def("__sub__", [](const Derived& a, double b) { return BinGen(a - b); }, py::is_operator());
      obj.def("__rsub__", [](const Derived& a, double b) { return BinGen(b - a); }, py::is_operator());
      obj.def("__rtruediv__", [](const Derived& a, double b) { return BinGen(b / a); }, py::is_operator());

      obj.def(
          "__rmul__",
          [](const Derived& a, const Eigen::VectorXd& b) { return Gen(MatrixScaled<Derived, -1>(a, b)); },
          py::is_operator());

      obj.def(
          "__mul__",
          [](const Derived& a, const Eigen::VectorXd& b) { return Gen(MatrixScaled<Derived, -1>(a, b)); },
          py::is_operator());
    }
  }

}  // namespace ASSET
