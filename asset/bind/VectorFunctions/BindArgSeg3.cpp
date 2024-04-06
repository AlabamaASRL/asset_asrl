#include <bind/VectorFunctions/BindArgSeg3.h>

void ASSET::BindArgSeg3(FunctionRegistry& reg, py::module& m) {
  reg.Build_Register<Arguments<-1>>(m, "Arguments");
}
