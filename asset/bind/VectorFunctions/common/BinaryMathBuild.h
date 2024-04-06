#pragma once

#include <bind/pch.h>

namespace ASSET {

  template<class Derived, int IR, int OR, class PyClass>
  void BinaryMathBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using BinGen = typename std::conditional<OR == 1, GenS, Gen>::type;

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

    if constexpr (OR == 3 || OR == -1) {

      if constexpr (!is_seg)
        obj.def("cross", [](const Derived& seg1, const SEG& seg2) { return Gen(crossProduct(seg1, seg2)); });

      if constexpr (!is_seg3)
        obj.def("cross", [](const Derived& seg1, const SEG3& seg2) { return Gen(crossProduct(seg1, seg2)); });
      if constexpr (!is_gen)
        obj.def("cross", [](const Derived& seg1, const Gen& seg2) { return Gen(crossProduct(seg1, seg2)); });

      obj.def("cross", [](const Derived& seg1, const Vector3<double>& seg2) {
        return Gen(crossProduct(seg1, Constant<-1, 3>(seg1.IRows(), seg2)));
      });

      obj.def("cross",
              [](const Derived& seg1, const Derived& seg2) { return Gen(crossProduct(seg1, seg2)); });
    }

    if constexpr (OR != 1) {
      if constexpr (!is_seg)
        obj.def("dot", [](const Derived& seg1, const SEG& seg2) { return GenS(dotProduct(seg1, seg2)); });

      if constexpr (!is_gen)
        obj.def("dot", [](const Derived& seg1, const Gen& seg2) { return GenS(dotProduct(seg1, seg2)); });

      ///////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////

      if constexpr (!is_seg)
        obj.def("cwiseProduct", [](const Derived& seg1, const SEG& seg2) {
          return Gen(CwiseFunctionProduct<Derived, SEG>(seg1, seg2));
        });
      if constexpr (!is_gen)
        obj.def("cwiseProduct", [](const Derived& seg1, const Gen& seg2) {
          return Gen(CwiseFunctionProduct<Derived, Gen>(seg1, seg2));
        });

      ///////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////////////////////////////////////////////

      if constexpr (!is_seg)
        obj.def("cwiseQuotient",
                [](const Derived& seg1, const SEG& seg2) { return Gen(cwiseQuotient(seg1, seg2)); });
      if constexpr (!is_gen)
        obj.def("cwiseQuotient",
                [](const Derived& seg1, const Gen& seg2) { return Gen(cwiseQuotient(seg1, seg2)); });
    }

    obj.def("dot", [](const Derived& seg1, const Derived& seg2) { return GenS(dotProduct(seg1, seg2)); });
    obj.def("cwiseProduct", [](const Derived& seg1, const Derived& seg2) {
      return BinGen(CwiseFunctionProduct<Derived, Derived>(seg1, seg2));
    });

    obj.def("cwiseQuotient",
            [](const Derived& seg1, const Derived& seg2) { return BinGen(cwiseQuotient(seg1, seg2)); });

    //////////////////////////////////////
    obj.def("dot", [](const Derived& seg1, const typename Derived::template Output<double>& seg2) {
      return GenS(dotProduct(seg1, Constant<-1, Derived::ORC>(seg1.IRows(), seg2)));
    });

    obj.def("cwiseProduct", [](const Derived& seg1, const typename Derived::template Output<double>& seg2) {
      return BinGen(RowScaled<Derived>(seg1, seg2));
    });

    obj.def("cwiseQuotient", [](const Derived& seg1, const typename Derived::template Output<double>& seg2) {
      typename Derived::template Output<double> seg2i = seg2.cwiseInverse();
      return BinGen(RowScaled<Derived>(seg1, seg2i));
    });
  }

}  // namespace ASSET
