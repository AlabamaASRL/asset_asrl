#include <bind/VectorFunctions/BindVectorFunction1.h>

void ASSET::BindVectorFunction1(FunctionRegistry& reg, py::module& m) {
  using Gen = GenericFunction<-1, -1>;

  UnaryMathBuild<Gen, -1, -1, decltype(reg.vfuncx)>(reg.vfuncx);
  BinaryMathBuild<Gen, -1, -1, decltype(reg.vfuncx)>(reg.vfuncx);
}
