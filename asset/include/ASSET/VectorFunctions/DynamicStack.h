#pragma once

#include <ASSET/VectorFunctions/CommonFunctions/Stacked.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericFunction.h>

namespace ASSET {

  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>>& elems);
  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>>& elems);

}  // namespace ASSET
