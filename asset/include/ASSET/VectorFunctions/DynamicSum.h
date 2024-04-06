#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/Summation.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>

namespace ASSET {

  GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>>& elems);
  GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>>& elems);

}  // namespace ASSET
