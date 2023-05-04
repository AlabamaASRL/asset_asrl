#pragma once
#include "CommonFunctions/CommonFunctions.h"
#include "FunctionRegistry.h"
#include "MathOverloads.h"
#include "OperatorOverloads.h"
#include "PythonArgParsing.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericComparative.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"

namespace ASSET {

  /// <summary>
  /// Builds All of vector Functions module, Calls all of other build functions
  /// </summary>
  /// <param name="reg"></param>
  /// <param name="m"></param>
  void VectorFunctionBuild(FunctionRegistry& reg, py::module& m);

  void VectorFunctionBuildPart1(FunctionRegistry& reg, py::module& m);
  void VectorFunctionBuildPart2(FunctionRegistry& reg, py::module& m);
  void ArgsSegBuildPart1(FunctionRegistry& reg, py::module& m);
  void ArgsSegBuildPart2(FunctionRegistry& reg, py::module& m);
  void ArgsSegBuildPart3(FunctionRegistry& reg, py::module& m);
  void ArgsSegBuildPart4(FunctionRegistry& reg, py::module& m);
  void ArgsSegBuildPart5(FunctionRegistry& reg, py::module& m);

  void BulkOperationsBuild(FunctionRegistry& reg, py::module& m);

  // std::vector<GenericFunction<-1, -1>> ParsePythonArgs(py::args x,int irows=0);
  // std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(py::args x,int irows=0);

  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, -1>>& elems);
  GenericFunction<-1, -1> DynamicStack(const std::vector<GenericFunction<-1, 1>>& elems);
  GenericFunction<-1, -1> DynamicSum(const std::vector<GenericFunction<-1, -1>>& elems);
  GenericFunction<-1, 1> DynamicSum(const std::vector<GenericFunction<-1, 1>>& elems);

  void FreeFunctionsBuild(FunctionRegistry& reg, py::module& m);
  void MatrixFunctionBuild(py::module& m);

}  // namespace ASSET
