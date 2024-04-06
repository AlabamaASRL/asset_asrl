#include <bind/VectorFunctions/BindArgSeg4.h>

void ASSET::BindArgSeg4(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<Segment<-1, 2, -1>>(m, "Segment2");
}
