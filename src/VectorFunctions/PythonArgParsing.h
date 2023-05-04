#pragma once

#include "CommonFunctions/ExpressionFwdDeclarations.h"
#include "pch.h"


namespace ASSET {

  /*
   * Converts list of python objects into a vector of dynamically sized GenericFunctions.
   * Can accept any of the fundamental types exposed to python as well as Python and Numpy
   * vectors and scalars.
   */
  std::vector<GenericFunction<-1, -1>> ParsePythonArgs(py::args x, int irows = 0);

  /*
   * Converts list of python objects into a vector of dynamically sized scalar GenericFunctions.
   * Can accept any of the fundamental scalar types exposed to python as well as Python and Numpy
   * scalars.
   */
  std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(py::args x, int irows = 0);


}  // namespace ASSET