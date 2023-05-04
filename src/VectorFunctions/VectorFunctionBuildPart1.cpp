#include "ASSET_VectorFunctions.h"

namespace ASSET {
  void VectorFunctionBuildPart1(FunctionRegistry& reg, py::module& m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    Gen::UnaryMathBuild(reg.vfuncx);
    Gen::BinaryMathBuild(reg.vfuncx);
  }

}  // namespace ASSET