#pragma once

#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericComparative.h>
#include <ASSET/VectorFunctions/VectorFunctionTypeErasure/GenericConditional.h>
#include <bind/pch.h>

#include "BindArgSeg1.h"
#include "BindArgSeg2.h"
#include "BindArgSeg3.h"
#include "BindArgSeg4.h"
#include "BindArgSeg5.h"
#include "BindBulkOperations.h"
#include "BindFreeFunctions.h"
#include "BindMatrixFunction.h"
#include "BindVectorFunction1.h"
#include "BindVectorFunction2.h"
#include "CommonFunctions/BindInterpTable1D.h"
#include "CommonFunctions/BindInterpTable2D.h"
#include "CommonFunctions/BindInterpTable3D.h"
#include "CommonFunctions/BindInterpTable4D.h"
#include "CommonFunctions/PythonFunctions.h"
#include "FunctionRegistry.h"
#include "common/GenericBuild.h"

namespace ASSET {

  void BindVectorFunctions(FunctionRegistry&, py::module&);

}
