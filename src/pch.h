#ifndef PCH_H
#define PCH_H

#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "TypeDefs/EigenTypes.h"
#include "Utils/EigenSTL.h"
#include "Utils/GetCoreCount.h"
#include "Utils/LambdaJumpTable.h"
#include "Utils/MathFunctions.h"
#include "Utils/STDExtensions.h"
#include "Utils/ThreadPool.h"
#include "Utils/Timer.h"
#include "Utils/TupleIterator.h"
#include "Utils/TypeErasure.h"
#include "Utils/TypeName.h"
#include "Utils/fmtlib.h"


////////////////////////////////////
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
//#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;


/////////////////////////////////

#endif  // PCH_H
