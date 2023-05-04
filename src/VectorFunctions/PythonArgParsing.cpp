#include "PythonArgParsing.h"

#include "CommonFunctions/CommonFunctions.h"
#include "FunctionRegistry.h"
#include "MathOverloads.h"
#include "OperatorOverloads.h"
#include "VectorFunction.h"
#include "VectorFunctionTypeErasure/GenericComparative.h"
#include "VectorFunctionTypeErasure/GenericConditional.h"
#include "VectorFunctionTypeErasure/GenericFunction.h"

namespace ASSET {


  std::vector<GenericFunction<-1, -1>> ParsePythonArgs(py::args x, int irows) {


    using std::cin;
    using std::cout;
    using std::endl;


    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    using Rtype = Gen;

    py::object vftype = (py::object) py::module::import("asset.VectorFunctions").attr("VectorFunction");
    py::object sftype = (py::object) py::module::import("asset.VectorFunctions").attr("ScalarFunction");
    py::object elemtype = (py::object) py::module::import("asset.VectorFunctions").attr("Element");
    py::object segtype = (py::object) py::module::import("asset.VectorFunctions").attr("Segment");
    py::object seg2type = (py::object) py::module::import("asset.VectorFunctions").attr("Segment2");
    py::object seg3type = (py::object) py::module::import("asset.VectorFunctions").attr("Segment3");
    py::object argtype = (py::object) py::module::import("asset.VectorFunctions").attr("Arguments");

    py::module builtins = py::module::import("builtins");
    py::object py_int = builtins.attr("int");
    py::object py_float = builtins.attr("float");
    py::object py_list = builtins.attr("list");
    py::object np_array = (py::object) py::module::import("numpy").attr("ndarray");
    py::object np_float = (py::object) py::module::import("numpy").attr("float64");
    py::object np_int = (py::object) py::module::import("numpy").attr("int32");

    int i = 0;
    // int irows = 0;
    for (auto xi = x.begin(); xi != x.end(); ++xi) {
      // py::print(py::str(xi->get_type()));
      if (xi->get_type().is(vftype) || xi->get_type().is(sftype) || xi->get_type().is(elemtype)
          || xi->get_type().is(segtype) || xi->get_type().is(seg2type) || xi->get_type().is(seg3type)
          || xi->get_type().is(argtype)) {
        int irowstmp = xi->attr("IRows")().cast<int>();
        if (irows == 0) {
          irows = irowstmp;
        } else if (irowstmp != irows) {
          throw std::invalid_argument("Asset functions in list must have same input size");
        }

      } else if (xi->get_type().is(py_float) || xi->get_type().is(py_int) || xi->get_type().is(np_int)
                 || xi->get_type().is(np_float)) {

        // Good to go
      } else if (xi->get_type().is(py_list) || xi->get_type().is(np_array)) {
        // Loop over and check that these are arrays of doubles or ints
        int lenvec = xi->attr("__len__")().cast<int>();
        for (int j = 0; j < lenvec; j++) {
          auto elemj = xi->attr("__getitem__")(py::int_(j)).get_type();
          if (!(elemj.is(py_float) || elemj.is(py_int) || elemj.is(np_int) || elemj.is(np_float))) {
            py::print(py::str(elemj));
            throw std::invalid_argument("Vectors and lists must only contain doubles or floats");
          }
        }
      }

      else {
        py::print(py::str(xi->get_type()));
        throw std::invalid_argument("Argument cannot be converted to Asset function");
      }

      i++;
    }

    if (irows == 0) {
      throw std::invalid_argument("Argument list must contain at least one asset function.");
    }

    std::vector<Rtype> funs;
    int Elem = 0;
    for (auto xi = x.begin(); xi != x.end(); ++xi) {
      if (xi->get_type().is(vftype)) {
        funs.emplace_back(Rtype(xi->cast<Gen>()));
      } else if (xi->get_type().is(sftype)) {
        funs.emplace_back(Rtype(xi->cast<GenS>()));
      } else if (xi->get_type().is(elemtype)) {
        funs.emplace_back(Rtype(xi->cast<ELEM>()));
      } else if (xi->get_type().is(segtype)) {
        funs.emplace_back(Rtype(xi->cast<SEG>()));
      } else if (xi->get_type().is(seg2type)) {
        funs.emplace_back(Rtype(xi->cast<SEG2>()));
      } else if (xi->get_type().is(seg3type)) {
        funs.emplace_back(Rtype(xi->cast<SEG3>()));
      } else if (xi->get_type().is(argtype)) {
        funs.emplace_back(Rtype(xi->cast<Arguments<-1>>()));
      } else if (xi->get_type().is(py_float) || xi->get_type().is(py_int) || xi->get_type().is(np_int)
                 || xi->get_type().is(np_float)) {
        Vector1<double> val;
        val[0] = xi->cast<double>();
        funs.emplace_back(Constant<-1, 1>(irows, val));
      } else if (xi->get_type().is(py_list) || xi->get_type().is(np_array)) {
        int lenvec = xi->attr("__len__")().cast<int>();
        VectorX<double> val(lenvec);
        for (int j = 0; j < lenvec; j++) {
          auto elemj = xi->attr("__getitem__")(py::int_(j)).get_type();
          val[j] = xi->attr("__getitem__")(py::int_(j)).cast<double>();
        }
        funs.emplace_back(Constant<-1, -1>(irows, val));
      } else {

        throw std::invalid_argument("Unrecognized Argument.");
      }
      Elem++;
    }

    return funs;
  }


  std::vector<GenericFunction<-1, 1>> ParsePythonArgsScalar(py::args x, int irows) {


    using std::cin;
    using std::cout;
    using std::endl;


    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    using Rtype = GenS;

    py::object sftype = (py::object) py::module::import("asset.VectorFunctions").attr("ScalarFunction");
    py::object elemtype = (py::object) py::module::import("asset.VectorFunctions").attr("Element");

    py::module builtins = py::module::import("builtins");
    py::object py_int = builtins.attr("int");
    py::object py_float = builtins.attr("float");
    py::object np_float = (py::object) py::module::import("numpy").attr("float64");
    py::object np_int = (py::object) py::module::import("numpy").attr("int32");

    int i = 0;
    // int irows = 0;
    for (auto xi = x.begin(); xi != x.end(); ++xi) {
      if (xi->get_type().is(sftype) || xi->get_type().is(elemtype)) {
        int irowstmp = xi->attr("IRows")().cast<int>();
        if (irows == 0) {
          irows = irowstmp;
        } else if (irowstmp != irows) {
          throw std::invalid_argument("Asset functions in list must have same input size");
        }

      } else if (xi->get_type().is(py_float) || xi->get_type().is(py_int) || xi->get_type().is(np_int)
                 || xi->get_type().is(np_float)) {
        // Good to go
      } else {
        py::print(py::str(xi->get_type()));
        throw std::invalid_argument("Argument cannot be converted to Asset function");
      }

      i++;
    }
    if (irows == 0) {
      throw std::invalid_argument("Argument list must contain at least one asset function.");
    }
    std::vector<Rtype> funs;
    int Elem = 0;
    for (auto xi = x.begin(); xi != x.end(); ++xi) {
      if (xi->get_type().is(sftype)) {
        funs.emplace_back(Rtype(xi->cast<GenS>()));
      } else if (xi->get_type().is(elemtype)) {
        funs.emplace_back(Rtype(xi->cast<ELEM>()));
      } else if (xi->get_type().is(py_float) || xi->get_type().is(py_int) || xi->get_type().is(np_float)
                 || xi->get_type().is(np_int)) {
        Vector1<double> val;
        val[0] = xi->cast<double>();
        funs.emplace_back(Constant<-1, 1>(irows, val));
      } else {
        throw std::invalid_argument("Unrecognized Argument.");
      }
      Elem++;
    }

    return funs;
  }
}  // namespace ASSET
