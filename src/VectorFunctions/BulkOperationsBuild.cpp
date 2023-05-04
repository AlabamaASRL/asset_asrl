#include "ASSET_VectorFunctions.h"

namespace ASSET {


  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>>& elems) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_stack<Gen, Gen>(elems);
  }
  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>>& elems) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_stack<Gen, GenS>(elems);
  }
  GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>>& elems) {
    using Gen = GenericFunction<-1, -1>;
    return make_dynamic_sum<Gen, Gen>(elems);
  }
  GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>>& elems) {
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_sum<GenS, GenS>(elems);
  }


}  // namespace ASSET


void ASSET::BulkOperationsBuild(FunctionRegistry& reg, py::module& m) {
  using Gen = GenericFunction<-1, -1>;
  using GenS = GenericFunction<-1, 1>;
  using SEG = Segment<-1, -1, -1>;
  using SEG2 = Segment<-1, 2, -1>;
  using SEG3 = Segment<-1, 3, -1>;
  using SEG4 = Segment<-1, 4, -1>;
  using ELEM = Segment<-1, 1, -1>;

  m.def("stack", [](const std::vector<GenS>& elems) { return DynamicStack(elems); });
  m.def("stack", [](const std::vector<Gen>& elems) { return DynamicStack(elems); });
  m.def("stack", [](const Gen& first, py::args x) {
    auto funcs = std::vector {first};
    auto funcsrest = ParsePythonArgs(x, first.IRows());
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicStack(funcs);
  });
  m.def("stack", [](double first, py::args x) {
    auto funcsrest = ParsePythonArgs(x);
    Vector1<double> val;
    val[0] = first;
    auto funcs = std::vector {Gen(Constant<-1, 1>(funcsrest[0].IRows(), val))};
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicStack(funcs);
  });
  m.def("stack", [](Eigen::VectorXd first, py::args x) {
    auto funcsrest = ParsePythonArgs(x);
    auto funcs = std::vector {Gen(Constant<-1, -1>(funcsrest[0].IRows(), first))};
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicStack(funcs);
  });
  ///////////////////////////////////////////////////////
  m.def("stack_scalar", [](const std::vector<GenS>& elems) { return DynamicStack(elems); });
  m.def("stack_scalar", [](const GenS& first, py::args x) {
    auto funcs = std::vector {first};
    auto funcsrest = ParsePythonArgsScalar(x, first.IRows());
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicStack(funcs);
  });
  m.def("stack_scalar", [](double first, py::args x) {
    auto funcsrest = ParsePythonArgsScalar(x);
    Vector1<double> val;
    val[0] = first;
    auto funcs = std::vector {GenS(Constant<-1, 1>(funcsrest[0].IRows(), val))};
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicStack(funcs);
  });
  ////////////////////////////////////////////////////////

  m.def("sum", [](const std::vector<GenS>& elems) { return DynamicSum(elems); });
  m.def("sum", [](const std::vector<Gen>& elems) { return DynamicSum(elems); });

  m.def("sum", [](const GenS& first, py::args x) {
    auto funcs = std::vector {first};
    auto funcsrest = ParsePythonArgsScalar(x);
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicSum(funcs);
  });
  m.def("sum", [](double first, py::args x) {
    auto funcsrest = ParsePythonArgsScalar(x);
    Vector1<double> val;
    val[0] = first;
    auto funcs = std::vector {GenS(Constant<-1, 1>(funcsrest[0].IRows(), val))};
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicSum(funcs);
  });

  m.def("sum", [](const Gen& first, py::args x) {
    auto funcs = std::vector {first};
    auto funcsrest = ParsePythonArgs(x);
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicSum(funcs);
  });

  m.def("sum", [](Eigen::VectorXd first, py::args x) {
    auto funcsrest = ParsePythonArgs(x);
    auto funcs = std::vector {Gen(Constant<-1, -1>(funcsrest[0].IRows(), first))};
    for (const auto& f: funcsrest)
      funcs.push_back(f);
    return DynamicSum(funcs);
  });


  //////////////////////////////////////////////////////////

  m.def("Stack", [](const std::vector<GenS>& elems) { return DynamicStack(elems); });
  m.def("StackScalar", [](const std::vector<GenS>& elems) { return DynamicStack(elems); });
  m.def("Stack", [](const std::vector<Gen>& elems) { return DynamicStack(elems); });


  m.def("Sum", [](const std::vector<GenS>& elems) { return DynamicSum(elems); });
  m.def("SumScalar", [](const std::vector<GenS>& elems) { return DynamicSum(elems); });
  m.def("Sum", [](const std::vector<Gen>& elems) { return DynamicSum(elems); });

  m.def("SumElems", &make_dynamic_sum<GenS, ELEM>);
  m.def("SumElems", [](const std::vector<ELEM>& elems, const std::vector<double>& scales) {
    std::vector<Scaled<ELEM>> selems;
    for (int i = 0; i < elems.size(); i++) {
      selems.emplace_back(elems[i] * scales[i]);
    };
    return make_dynamic_sum<GenS, Scaled<ELEM>>(selems);
  });
}