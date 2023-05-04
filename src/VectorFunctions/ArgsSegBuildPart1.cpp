#include "ASSET_VectorFunctions.h"

namespace ASSET {
  void ArgsSegBuildPart1(FunctionRegistry& reg, py::module& m) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using ELEM = Segment<-1, 1, -1>;
    reg.Build_Register<SEG>(m, "Segment");
  }

}  // namespace ASSET