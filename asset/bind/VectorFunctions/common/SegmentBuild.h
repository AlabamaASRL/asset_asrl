#pragma once

#include <bind/pch.h>

#include "BinaryMathBuild.h"
#include "BinaryOperatorsBuild.h"
#include "ConditionalOperatorsBuild.h"
#include "DoubleMathBuild.h"
#include "FunctionIndexingBuild.h"
#include "UnaryMathBuild.h"

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void SegmentBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    DoubleMathBuild<Derived, IR, OR, PyClass>(obj);
    UnaryMathBuild<Derived, IR, OR, PyClass>(obj);
    BinaryMathBuild<Derived, IR, OR, PyClass>(obj);
    BinaryOperatorsBuild<Derived, IR, OR, PyClass>(obj);
    FunctionIndexingBuild<Derived, IR, OR, PyClass>(obj);
    ConditionalOperatorsBuild<Derived, IR, OR, PyClass>(obj);

    obj.def("tolist", [](const Derived& func) {
      using ELEM = Segment<-1, 1, -1>;
      std::vector<ELEM> elems;
      for (int i = 0; i < func.ORows(); i++) {
        elems.push_back(func.coeff(i));
      }
      return elems;
    });

    obj.def("tolist", [](const Derived& func, std::vector<int> coeffs) {
      using ELEM = Segment<-1, 1, -1>;
      std::vector<ELEM> elems;
      for (const auto& coeff: coeffs) {
        elems.push_back(func.coeff(coeff));
      }
      return elems;
    });

    obj.def("tolist", [](const Derived& func, std::vector<std::tuple<int, int>> seglist) {
      using ELEM = Segment<-1, 1, -1>;
      using SEG2 = Segment<-1, 2, -1>;
      using SEG3 = Segment<-1, 3, -1>;
      using SEG = Segment<-1, -1, -1>;

      std::vector<py::object> segs;
      for (const auto& seg: seglist) {

        int start = std::get<0>(seg);
        int size = std::get<1>(seg);
        py::object pyfun;
        if (size == 1) {
          auto f = func.coeff(start);
          pyfun = py::cast(f);
        } else if (size == 2) {
          auto f = func.template segment<2>(start);
          pyfun = py::cast(f);
        } else if (size == 3) {
          auto f = func.template segment<3>(start);
          pyfun = py::cast(f);
        } else {
          auto f = func.segment(start, size);
          pyfun = py::cast(f);
        }

        segs.push_back(pyfun);
      }
      return segs;
    });
  }

}  // namespace ASSET
