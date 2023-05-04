#include "ASSET_VectorFunctions.h"

#include "CommonFunctions/InterpTable1D.h"
#include "CommonFunctions/InterpTable2D.h"
#include "CommonFunctions/InterpTable3D.h"
#include "Utils/fmtlib.h"

namespace ASSET {}  // namespace ASSET

void ASSET::VectorFunctionBuild(FunctionRegistry& reg, py::module& m) {
  auto& mod = reg.getVectorFunctionsModule();

  using Gen = GenericFunction<-1, -1>;
  using GenS = GenericFunction<-1, 1>;
  using SEG = Segment<-1, -1, -1>;
  using SEG2 = Segment<-1, 2, -1>;
  using SEG3 = Segment<-1, 3, -1>;
  using ELEM = Segment<-1, 1, -1>;

  //////////////////////////////////
  Gen::GenericBuild(reg.vfuncx);
  GenS::GenericBuild(reg.sfuncx);
  py::implicitly_convertible<GenS, Gen>();
  VectorFunctionBuildPart1(reg, mod);
  VectorFunctionBuildPart2(reg, mod);
  ////////////////////////////////////

  ArgsSegBuildPart1(reg, mod);
  ArgsSegBuildPart2(reg, mod);
  ArgsSegBuildPart3(reg, mod);
  ArgsSegBuildPart4(reg, mod);
  ArgsSegBuildPart5(reg, mod);

  BulkOperationsBuild(reg, mod);
  FreeFunctionsBuild(reg, mod);
  MatrixFunctionBuild(mod);
  GenericConditional<-1>::ConditionalBuild(mod);
  GenericComparative<-1>::ComparativeBuild(mod);

  reg.Build_Register<PyVectorFunction<-1, -1>>(mod, "PyVectorFunction");
  reg.Build_Register<PyVectorFunction<-1, 1>>(mod, "PyScalarFunction");

  reg.Build_Register<Constant<-1, -1>>(mod, "ConstantVector");
  reg.Build_Register<Constant<-1, 1>>(mod, "ConstantScalar");


  InterpTable1DBuild(mod);
  InterpTable2DBuild(mod);
  InterpTable3DBuild(mod);

  mod.def("ScalarDynamicStackTest", [](const std::vector<GenericFunction<-1, 1>>& funcs) {
    return GenericFunction<-1, -1>(DynamicStackedOutputs {funcs});
  });

  mod.def("DynamicStackTest", [](const std::vector<GenericFunction<-1, -1>>& funcs) {
    return GenericFunction<-1, -1>(DynamicStackedOutputs {funcs});
  });
}
