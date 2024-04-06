#pragma once

#include <bind/Solvers/BindJet.h>
#include <bind/Solvers/BindOptimizationProblem.h>
#include <bind/Solvers/BindOptimizationProblemBase.h>
#include <bind/Solvers/BindPSIOPT.h>
#include <bind/VectorFunctions/FunctionRegistry.h>
#include <bind/pch.h>
#include <mkl.h>

namespace ASSET {

  void BindSolvers(FunctionRegistry&, py::module&);

}
