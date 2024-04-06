#include <bind/VectorFunctions/BindVectorFunction2.h>

namespace ASSET {

  template<class FType>
  void DefineListEval(py::class_<FType>& obj) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    obj.def("__call__", [](const FType& fun, const std::vector<GenS>& funcs) {
      return FType(fun.eval(DynamicStack(funcs)));
    });
    obj.def("__call__", [](const FType& fun, const std::vector<Gen>& funcs) {
      return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType& fun, const Gen& first, py::args x) {
      auto funcs = std::vector {first};
      auto funcsrest = ParsePythonArgs(x, first.IRows());
      for (const auto& f: funcsrest)
        funcs.push_back(f);
      return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType& fun, double first, py::args x) {
      auto funcsrest = ParsePythonArgs(x);
      Vector1<double> val;
      val[0] = first;
      auto funcs = std::vector {Gen(Constant<-1, 1>(funcsrest[0].IRows(), val))};
      for (const auto& f: funcsrest)
        funcs.push_back(f);
      return FType(fun.eval(DynamicStack(funcs)));
    });

    obj.def("__call__", [](const FType& fun, Eigen::VectorXd first, py::args x) {
      auto funcsrest = ParsePythonArgs(x);
      auto funcs = std::vector {Gen(Constant<-1, -1>(funcsrest[0].IRows(), first))};
      for (const auto& f: funcsrest)
        funcs.push_back(f);
      return FType(fun.eval(DynamicStack(funcs)));
    });
  }

}  // namespace ASSET

void ASSET::BindVectorFunction2(FunctionRegistry& reg, py::module& m) {
  using Gen = GenericFunction<-1, -1>;
  using GenS = GenericFunction<-1, 1>;

  DoubleMathBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);
  UnaryMathBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);
  BinaryMathBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);
  BinaryOperatorsBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);
  FunctionIndexingBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);
  ConditionalOperatorsBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);

  ///////////////////////////////////////
  DoubleMathBuild<Gen, -1, -1, decltype(reg.vfuncx)>(reg.vfuncx);
  FunctionIndexingBuild<Gen, -1, -1, decltype(reg.vfuncx)>(reg.vfuncx);
  BinaryOperatorsBuild<Gen, -1, -1, decltype(reg.vfuncx)>(reg.vfuncx);
  ///////////////////////////////////////

  DefineListEval(reg.vfuncx);
  DefineListEval(reg.sfuncx);
}
