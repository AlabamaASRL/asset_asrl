#include <ASSET/VectorFunctions/DynamicStack.h>

namespace ASSET {

  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>>& elems) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_stack<Gen, Gen>(elems);
  }

  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>>& elems) {
    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_stack<Gen, GenS>(elems);
  }

}  // namespace ASSET
