#include <bind/VectorFunctions/BindArgSeg1.h>

void ASSET::BindArgSeg1(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<Segment<-1, -1, -1>>(m, "Segment");
}
