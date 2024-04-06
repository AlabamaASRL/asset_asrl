#include <bind/VectorFunctions/BindVectorFunctions.h>

void ASSET::BindVectorFunctions(FunctionRegistry& reg, py::module& m) {

  auto& mod = reg.getVectorFunctionsModule();

  using Gen = GenericFunction<-1, -1>;
  using GenS = GenericFunction<-1, 1>;
  using SEG = Segment<-1, -1, -1>;
  using SEG2 = Segment<-1, 2, -1>;
  using SEG3 = Segment<-1, 3, -1>;
  using ELEM = Segment<-1, 1, -1>;

  //////////////////////////////////
  GenericBuild<Gen, -1, -1, decltype(reg.vfuncx)>(reg.vfuncx);
  GenericBuild<GenS, -1, 1, decltype(reg.sfuncx)>(reg.sfuncx);
  py::implicitly_convertible<GenS, Gen>();
  BindVectorFunction1(reg, mod);
  BindVectorFunction2(reg, mod);
  ////////////////////////////////////

  BindArgSeg1(reg, mod);
  BindArgSeg2(reg, mod);
  BindArgSeg3(reg, mod);
  BindArgSeg4(reg, mod);
  BindArgSeg5(reg, mod);

  BindBulkOperations(reg, mod);
  BindFreeFunctions(reg, mod);
  BindMatrixFunction(mod);

  FunctionBinder<GenericConditional<-1>>::Bind(mod, "Conditional");
  FunctionBinder<GenericComparative<-1>>::Bind(mod, "Comparative");

  reg.Build_Register<PyVectorFunction<-1, -1>>(mod, "PyVectorFunction");
  reg.Build_Register<PyVectorFunction<-1, 1>>(mod, "PyScalarFunction");

  reg.Build_Register<Constant<-1, -1>>(mod, "ConstantVector");
  reg.Build_Register<Constant<-1, 1>>(mod, "ConstantScalar");

  BindInterpTable1D(mod);
  BindInterpTable2D(mod);
  BindInterpTable3D(mod);
  BindInterpTable4D(mod);

  mod.def("ScalarDynamicStackTest", [](const std::vector<GenericFunction<-1, 1>>& funcs) {
    return GenericFunction<-1, -1>(DynamicStackedOutputs {funcs});
  });

  mod.def("DynamicStackTest", [](const std::vector<GenericFunction<-1, -1>>& funcs) {
    return GenericFunction<-1, -1>(DynamicStackedOutputs {funcs});
  });
}
