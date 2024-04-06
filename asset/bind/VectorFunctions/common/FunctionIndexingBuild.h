#pragma once

#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void FunctionIndexingBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    obj.def("padded_lower", [](const Derived& a, int lpad) { return Gen(a.padded_lower(lpad)); });
    obj.def("padded_upper", [](const Derived& a, int upad) { return Gen(a.padded_upper(upad)); });
    obj.def("padded", [](const Derived& a, int upad, int lpad) { return Gen(a.padded(upad, lpad)); });

    constexpr bool is_seg = Is_Segment<Derived>::value || Is_Arguments<Derived>::value;

    using SegRetType = typename std::conditional<is_seg, SEG, GenericFunction<-1, -1>>::type;
    using ElemRetType = typename std::conditional<is_seg, ELEM, GenericFunction<-1, 1>>::type;

    obj.def("segment",
            [](const Derived& a, int start, int size) { return SegRetType(a.segment(start, size)); });
    obj.def("head", [](const Derived& a, int size) { return SegRetType(a.head(size)); });
    obj.def("tail", [](const Derived& a, int size) { return SegRetType(a.tail(size)); });

    obj.def("coeff", [](const Derived& a, int elem) { return ElemRetType(a.coeff(elem)); });
    obj.def(
        "__getitem__",
        [](const Derived& a, int elem) { return ElemRetType(a.coeff(elem)); },
        py::is_operator());
    obj.def(
        "__getitem__",
        [](const Derived& a, const py::slice& slice) {
          size_t start, stop, step, slicelength;
          if (!slice.compute(a.ORows(), &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
          }

          if (step != 1) {
            throw std::invalid_argument("Non continous slices not supported");
          }
          if (start >= a.ORows()) {
            throw std::invalid_argument("Segment index out of bounds.");
          }
          if (start > stop) {
            throw std::invalid_argument("Backward indexing not supported.");
          }
          if (slicelength <= 0) {
            throw std::invalid_argument("Slice length must be greater than 0.");
          }

          int start_ = start;
          int size_ = stop - start;

          return SegRetType(a.segment(start_, size_));
        },
        py::is_operator());

    // if constexpr (OR != 1) {
    //     obj.def("colmatrix", [](const Derived& a, int rows, int cols) {
    //         Gen agen(a);
    //         return agen.colmatrix(rows, cols);
    //         });
    //     obj.def("rowmatrix", [](const Derived& a, int rows, int cols) {
    //         Gen agen(a);
    //         return agen.rowmatrix(rows, cols);
    //         });
    // }

    if constexpr (OR < 0 || OR > 2) {
      using Seg2RetType = typename std::conditional<is_seg, SEG2, GenericFunction<-1, -1>>::type;

      auto seg2 = [](const Derived& a, int start) { return Seg2RetType(a.template segment<2>(start)); };
      auto head2 = [](const Derived& a) { return Seg2RetType(a.template segment<2>(0)); };
      auto tail2 = [](const Derived& a) { return Seg2RetType(a.template segment<2>(a.ORows() - 2)); };

      obj.def("segment_2", seg2);
      obj.def("head_2", head2);
      obj.def("tail_2", tail2);
      obj.def("segment2", seg2);
      obj.def("head2", head2);
      obj.def("tail2", tail2);
    }
    if constexpr (OR < 0 || OR > 3) {
      using Seg3RetType = typename std::conditional<is_seg, SEG3, GenericFunction<-1, -1>>::type;
      auto seg3 = [](const Derived& a, int start) { return Seg3RetType(a.template segment<3>(start)); };
      auto head3 = [](const Derived& a) { return Seg3RetType(a.template segment<3>(0)); };
      auto tail3 = [](const Derived& a) { return Seg3RetType(a.template segment<3>(a.ORows() - 3)); };
      obj.def("segment_3", seg3);
      obj.def("head_3", head3);
      obj.def("tail_3", tail3);
      obj.def("segment3", seg3);
      obj.def("head3", head3);
      obj.def("tail3", tail3);
    }
  }

}  // namespace ASSET
