#pragma once

#include <bind/pch.h>

namespace ASSET {

  template<class PyClass>
  void MinMaxBuild(PyClass& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using GenComp = GenericComparative<-1>;

    // =========================================================================
    // ONE ARG BINDINGS

    // =========================================================================
    // TWO ARG BINDINGS
    // 1D Output Maximization Bindings
    obj.def("max", [](const GenS& f1, const GenS& f2) {
      return GenS(ComparativeFunction<GenS, GenS>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](double v1, const GenS& f2) {
      Vector1<double> v;
      v[0] = v1;
      Constant<-1, 1> f1(f2.IRows(), v);
      return GenS(ComparativeFunction<Constant<-1, 1>, GenS>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](const GenS& f1, double v2) {
      Vector1<double> v;
      v[0] = v2;
      Constant<-1, 1> f2(f1.IRows(), v);
      return GenS(ComparativeFunction<GenS, Constant<-1, 1>>(ComparativeFlags::MaxFlag, f1, f2));
    });

    // ND Output Maximization Bindings
    obj.def("max", [](const Gen& f1, const Gen& f2) {
      return Gen(ComparativeFunction<Gen, Gen>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](Eigen::VectorXd v1, const Gen& f2) {
      Constant<-1, -1> f1(f2.IRows(), v1);
      return Gen(ComparativeFunction<Constant<-1, -1>, Gen>(ComparativeFlags::MaxFlag, f1, f2));
    });
    obj.def("max", [](const Gen& f1, Eigen::VectorXd v2) {
      Constant<-1, -1> f2(f1.IRows(), v2);
      return Gen(ComparativeFunction<Gen, Constant<-1, -1>>(ComparativeFlags::MaxFlag, f1, f2));
    });

    // 1D Output Minimization Bindings
    obj.def("min", [](const GenS& f1, const GenS& f2) {
      return GenS(ComparativeFunction<GenS, GenS>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](double v1, const GenS& f2) {
      Vector1<double> v;
      v[0] = v1;
      Constant<-1, 1> f1(f2.IRows(), v);
      return GenS(ComparativeFunction<Constant<-1, 1>, GenS>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](const GenS& f1, double v2) {
      Vector1<double> v;
      v[0] = v2;
      Constant<-1, 1> f2(f1.IRows(), v);
      return GenS(ComparativeFunction<GenS, Constant<-1, 1>>(ComparativeFlags::MinFlag, f1, f2));
    });

    // ND Output Maximization Bindings
    obj.def("min", [](const Gen& f1, const Gen& f2) {
      return Gen(ComparativeFunction<Gen, Gen>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](Eigen::VectorXd v1, const Gen& f2) {
      Constant<-1, -1> f1(f2.IRows(), v1);
      return Gen(ComparativeFunction<Constant<-1, -1>, Gen>(ComparativeFlags::MinFlag, f1, f2));
    });
    obj.def("min", [](const Gen& f1, Eigen::VectorXd v2) {
      Constant<-1, -1> f2(f1.IRows(), v2);
      return Gen(ComparativeFunction<Gen, Constant<-1, -1>>(ComparativeFlags::MinFlag, f1, f2));
    });

    // =========================================================================
    // THREE ARG BINDINGS
    // TODO: 3-argument min-max bindings
  }

}  // namespace ASSET
