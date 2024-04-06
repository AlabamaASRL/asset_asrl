#pragma once

#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void BinaryOperatorsBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using ARGS = Arguments<-1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    constexpr bool is_seg = std::is_same<Derived, SEG>::value;
    constexpr bool is_seg2 = std::is_same<Derived, SEG2>::value;
    constexpr bool is_seg3 = std::is_same<Derived, SEG3>::value;

    constexpr bool is_elem = std::is_same<Derived, ELEM>::value;
    constexpr bool is_gen = std::is_same<Derived, Gen>::value;
    constexpr bool is_gens = std::is_same<Derived, GenS>::value;

    constexpr bool is_arglike = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

    if constexpr (!is_arglike) {

      obj.def("eval", [](const Derived& a, const ELEM& b) { return BinGen(a.eval(b)); });
      obj.def("eval", [](const Derived& a, const SEG& b) { return BinGen(a.eval(b)); });
      obj.def("eval", [](const Derived& a, const SEG2& b) { return BinGen(a.eval(b)); });
      obj.def("eval", [](const Derived& a, const SEG3& b) { return BinGen(a.eval(b)); });
      obj.def("eval", [](const Derived& a, const Gen& b) { return BinGen(a.eval(b)); });

      /////////////////////////////////////////////////////
      obj.def(
          "__call__", [](const Derived& a, const Gen& b) { return BinGen(a.eval(b)); }, py::is_operator());
      obj.def(
          "__call__", [](const Derived& a, const ELEM& b) { return BinGen(a.eval(b)); }, py::is_operator());
      obj.def("__call__", [](const Derived& a, const SEG& b) { return BinGen(a.eval(b)); });
      obj.def(
          "__call__", [](const Derived& a, const SEG2& b) { return BinGen(a.eval(b)); }, py::is_operator());
      obj.def(
          "__call__", [](const Derived& a, const SEG3& b) { return BinGen(a.eval(b)); }, py::is_operator());
      ///////////////////////////////////////////////////////////////
      obj.def("eval", [](const Derived& a, int ir, Eigen::VectorXi v) {
        return BinGen(ParsedInput<Derived, -1, OR>(a, v, ir));
      });
    }

    obj.def(
        "apply",
        [](const Derived& a, const Gen& b) {
          // return Gen(b.eval(a));
          return Gen(NestedFunction<Derived, Gen>(a, b));
        },
        py::is_operator());
    obj.def(
        "apply",
        [](const Derived& a, const GenS& b) {
          // return GenS(b.eval(a));
          return Gen(NestedFunction<Derived, GenS>(a, b));
        },
        py::is_operator());

    obj.def("__add__", [](const Derived& a, const Derived& b) { return BinGen(a + b); }, py::is_operator());
    obj.def("__sub__", [](const Derived& a, const Derived& b) { return BinGen(a - b); }, py::is_operator());

    obj.def("__mul__", [](const Derived& a, const ELEM& b) { return BinGen(a * b); }, py::is_operator());
    obj.def("__mul__", [](const Derived& a, const GenS& b) { return BinGen(a * b); }, py::is_operator());

    obj.def("__truediv__", [](const Derived& a, const ELEM& b) { return BinGen(a / b); }, py::is_operator());

    obj.def("__truediv__", [](const Derived& a, const GenS& b) { return BinGen(a / b); }, py::is_operator());

    if constexpr (is_seg || is_elem || is_seg2 || is_seg3) {  //
      obj.def(
          "__add__",
          [](const Derived& a, const BinGen& b) { return BinGen(TwoFunctionSum<Derived, BinGen>(a, b)); },
          py::is_operator());
      obj.def(
          "__sub__",
          [](const Derived& a, const BinGen& b) { return BinGen(FunctionDifference<Derived, BinGen>(a, b)); },
          py::is_operator());
    }

    if constexpr (OR != 1) {

      obj.def("__rmul__", [](const Derived& a, const ELEM& b) { return BinGen(a * b); }, py::is_operator());

      obj.def("__rmul__", [](const Derived& a, const GenS& b) { return BinGen(a * b); }, py::is_operator());
    }
  }

}  // namespace ASSET
