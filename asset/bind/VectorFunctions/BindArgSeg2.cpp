#include <bind/VectorFunctions/BindArgSeg2.h>

void ASSET::BindArgSeg2(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<Segment<-1, 1, -1>>(m, "Element");
}
