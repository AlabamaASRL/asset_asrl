#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/Segment.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>
#include <bind/pch.h>

namespace ASSET {

  template<class PyClass>
  void IfElseBuild(PyClass& obj) {

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using ELEM = Segment<-1, 1, -1>;
    using GenCon = GenericConditional<-1>;

    obj.def("ifelse", [](const GenCon& test, const GenS& tf, const GenS& ff) {
      return GenS(IfElseFunction {test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon& test, double tfv, const GenS& ff) {
      Vector1<double> v;
      v[0] = tfv;
      Constant<-1, 1> tf(test.IRows(), v);
      return GenS(IfElseFunction {test, tf, ff});
    });
    obj.def("ifelse", [](const GenCon& test, const GenS& tf, double ffv) {
      Vector1<double> v;
      v[0] = ffv;
      Constant<-1, 1> ff(test.IRows(), v);
      return GenS(IfElseFunction {test, tf, ff});
    });
    obj.def("ifelse", [](const GenCon& test, double tfv, double ffv) {
      Vector1<double> v1;
      v1[0] = tfv;
      Constant<-1, 1> tf(test.IRows(), v1);
      Vector1<double> v2;
      v2[0] = ffv;
      Constant<-1, 1> ff(test.IRows(), v2);
      return GenS(IfElseFunction {test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon& test, const Gen& tf, const Gen& ff) {
      return Gen(IfElseFunction {test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon& test, Eigen::VectorXd v, const Gen& ff) {
      Constant<-1, -1> tf(test.IRows(), v);
      return Gen(IfElseFunction {test, tf, ff});
    });
    obj.def("ifelse", [](const GenCon& test, const Gen& tf, Eigen::VectorXd v) {
      Constant<-1, -1> ff(test.IRows(), v);
      return Gen(IfElseFunction {test, tf, ff});
    });

    obj.def("ifelse", [](const GenCon& test, Eigen::VectorXd v1, Eigen::VectorXd v2) {
      Constant<-1, -1> tf(test.IRows(), v1);
      Constant<-1, -1> ff(test.IRows(), v2);
      return Gen(IfElseFunction {test, tf, ff});
    });
  }

}  // namespace ASSET
