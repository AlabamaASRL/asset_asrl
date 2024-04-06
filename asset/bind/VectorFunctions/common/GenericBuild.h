#pragma once

#include "DenseBaseBuild.h"

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void GenericBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    obj.def(py::init<const GenericFunction<IR, OR>&>());
    if constexpr (OR == -1 && IR == -1) {
      obj.def(py::init(&Derived::template PyCopy<GenericFunction<IR, 1>>));
    }

    obj.def("input_domain", &Derived::input_domain);
    obj.def("is_linear", &Derived::is_linear);
    obj.def("SuperTest", &Derived::SuperTest);
    obj.def("SpeedTest", &Derived::SpeedTest);

    DenseBaseBuild<Derived, OR, decltype(obj)>(obj);
  }

}  // namespace ASSET
