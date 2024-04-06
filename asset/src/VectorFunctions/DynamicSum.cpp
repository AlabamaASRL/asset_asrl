#include <ASSET/VectorFunctions/DynamicSum.h>

namespace ASSET {

  GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>>& elems) {
    using Gen = GenericFunction<-1, -1>;
    return make_dynamic_sum<Gen, Gen>(elems);
  }

  GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>>& elems) {
    using GenS = GenericFunction<-1, 1>;
    return make_dynamic_sum<GenS, GenS>(elems);
  }

}  // namespace ASSET
