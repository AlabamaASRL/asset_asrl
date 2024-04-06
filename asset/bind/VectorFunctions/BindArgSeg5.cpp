#include <bind/VectorFunctions/BindArgSeg5.h>

void ASSET::BindArgSeg5(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<Segment<-1, 3, -1>>(m, "Segment3");
}
