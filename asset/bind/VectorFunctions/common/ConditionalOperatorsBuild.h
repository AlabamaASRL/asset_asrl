#pragma once

#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h>
#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void ConditionalOperatorsBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<IR>;

    if constexpr (OR == 1) {
      obj.def("__lt__", [](const Derived& a, double b) { return GenCon(a < b); }, py::is_operator());
      obj.def("__gt__", [](const Derived& a, double b) { return GenCon(a > b); }, py::is_operator());
      obj.def("__rlt__", [](const Derived& a, double b) { return GenCon(a > b); }, py::is_operator());
      obj.def(
          "__rgt__",
          [](const Derived& a, double b) {
            Vector1<double> tmp;
            tmp[0] = b;
            Constant<IR, 1> bfunc(a.IRows(), tmp);
            return GenCon(a < b);
          },
          py::is_operator());
      obj.def("__le__", [](const Derived& a, double b) { return GenCon(a <= b); }, py::is_operator());
      obj.def("__ge__", [](const Derived& a, double b) { return GenCon(a >= b); }, py::is_operator());

      obj.def("__lt__", [](const Derived& a, const Derived& b) { return GenCon(a < b); }, py::is_operator());
      obj.def("__gt__", [](const Derived& a, const Derived& b) { return GenCon(a > b); }, py::is_operator());
      obj.def("__le__", [](const Derived& a, const Derived& b) { return GenCon(a <= b); }, py::is_operator());
      obj.def("__ge__", [](const Derived& a, const Derived& b) { return GenCon(a >= b); }, py::is_operator());

      if constexpr (std::is_same<Derived, ELEM>::value) {
        obj.def("__lt__", [](const Derived& a, const GenS& b) { return GenCon(a < b); }, py::is_operator());
        obj.def("__gt__", [](const Derived& a, const GenS& b) { return GenCon(a > b); }, py::is_operator());
        obj.def("__le__", [](const Derived& a, const GenS& b) { return GenCon(a <= b); }, py::is_operator());
        obj.def("__ge__", [](const Derived& a, const GenS& b) { return GenCon(a >= b); }, py::is_operator());
      }
    }
  }

}  // namespace ASSET
