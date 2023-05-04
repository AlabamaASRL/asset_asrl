#include "ASSET_VectorFunctions.h"
#include "CommonFunctions/RootFinder.h"
namespace ASSET {


  template<class T>
  void UnaryVectorOpBuild(py::module& m) {

    auto UnaryOpLam = [](const T& cfun, const char* name) {
      py::object fun = py::cast(cfun);
      return fun.attr(name)();
    };
    m.def("norm", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "norm"); });
    m.def("squared_norm", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "squared_norm"); });
    m.def("cubed_norm", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "cubed_norm"); });

    m.def("inverse_norm", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "inverse_norm"); });
    m.def("inverse_squared_norm",
          [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "inverse_squared_norm"); });
    m.def("inverse_cubed_norm", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "inverse_cubed_norm"); });
    m.def("inverse_four_norm", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "inverse_four_norm"); });


    m.def("normalize", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "normalized"); });
    m.def("normalized", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "normalized"); });
    m.def("normalized_power2", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "normalized_power2"); });
    m.def("normalized_power3", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "normalized_power3"); });
    m.def("normalized_power4", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "normalized_power4"); });
    m.def("normalized_power5", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "normalized_power5"); });
  }

  template<class T>
  void UnaryScalarOpBuild(py::module& m) {

    auto UnaryOpLam = [](const T& cfun, const char* name) {
      py::object fun = py::cast(cfun);
      return fun.attr(name)();
    };
    m.def("sin", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "sin"); });
    m.def("cos", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "cos"); });
    m.def("tan", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "tan"); });

    m.def("sqrt", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "sqrt"); });
    m.def("exp", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "exp"); });
    m.def("log", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "log"); });
    m.def("squared", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "squared"); });

    m.def("arcsin", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "arcsin"); });
    m.def("arccos", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "arccos"); });
    m.def("arctan", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "arctan"); });

    m.def("sinh", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "sinh"); });
    m.def("cosh", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "cosh"); });
    m.def("tanh", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "tanh"); });

    m.def("pow", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "pow"); });

    m.def("arcsinh", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "arcsinh"); });
    m.def("arccosh", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "arccosh"); });
    m.def("arctanh", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "arctanh"); });

    m.def("sign", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "sign"); });
    m.def("abs", [UnaryOpLam](const T& fun) { return UnaryOpLam(fun, "__abs__"); });
  }

  void ProductBuild(py::module& m) {

    using Gen = GenericFunction<-1, -1>;
    using GenS = GenericFunction<-1, 1>;

    using ARGS = Arguments<-1>;
    using SEG = Segment<-1, -1, -1>;
    using SEG2 = Segment<-1, 2, -1>;
    using SEG3 = Segment<-1, 3, -1>;
    using SEG4 = Segment<-1, 4, -1>;
    using ELEM = Segment<-1, 1, -1>;

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////Cross Product///////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////


    auto CrossOpLam = [](const auto& f1, const auto& f2) {
      py::object fun1 = py::cast(f1);
      py::object fun2 = py::cast(f2);
      return fun1.attr("cross")(fun2);
    };

    m.def("cross", [CrossOpLam](const SEG3& f1, const Vector3<double>& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG& f1, const Vector3<double>& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const Gen& f1, const Vector3<double>& f2) { return CrossOpLam(f1, f2); });

    m.def("cross", [CrossOpLam](const Vector3<double>& f2, const SEG3& f1) {
      Vector3<double> f2tmp = -1.0 * f2;
      return CrossOpLam(f1, f2tmp);
    });
    m.def("cross", [CrossOpLam](const Vector3<double>& f2, const SEG& f1) {
      Vector3<double> f2tmp = -1.0 * f2;
      return CrossOpLam(f1, f2tmp);
    });
    m.def("cross", [CrossOpLam](const Vector3<double>& f2, const Gen& f1) {
      Vector3<double> f2tmp = -1.0 * f2;
      return CrossOpLam(f1, f2tmp);
    });


    m.def("cross", [CrossOpLam](const SEG3& f1, const SEG3& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG3& f1, const SEG& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG3& f1, const Gen& f2) { return CrossOpLam(f1, f2); });

    m.def("cross", [CrossOpLam](const SEG& f1, const SEG& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG& f1, const SEG3& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const SEG& f1, const Gen& f2) { return CrossOpLam(f1, f2); });

    m.def("cross", [CrossOpLam](const Gen& f1, const SEG3& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const Gen& f1, const SEG& f2) { return CrossOpLam(f1, f2); });
    m.def("cross", [CrossOpLam](const Gen& f1, const Gen& f2) { return CrossOpLam(f1, f2); });


    m.def("doublecross", [](const SEG3& seg1, const SEG3& seg2, const SEG3& seg3) {
      auto f1 = FunctionCrossProduct<SEG3, SEG3>(seg1, seg2);
      return Gen(FunctionCrossProduct<decltype(f1), SEG3>(f1, seg3));
    });
    m.def("doublecross", [](const Gen& seg1, const Gen& seg2, const Gen& seg3) {
      auto f1 = FunctionCrossProduct<Gen, Gen>(seg1, seg2);
      return Gen(FunctionCrossProduct<decltype(f1), Gen>(f1, seg3));
    });

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////cwiseProduct////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto cwiseProductOpLam = [](const auto& f1, const auto& f2) {
      py::object fun1 = py::cast(f1);
      py::object fun2 = py::cast(f2);
      return fun1.attr("cwiseProduct")(fun2);
    };

    m.def("cwiseProduct", [cwiseProductOpLam](const SEG2& f1, const Vector2<double>& f2) {
      return cwiseProductOpLam(f1, f2);
    });
    m.def("cwiseProduct", [cwiseProductOpLam](const SEG3& f1, const Vector3<double>& f2) {
      return cwiseProductOpLam(f1, f2);
    });
    m.def("cwiseProduct", [cwiseProductOpLam](const SEG& f1, const Vector3<double>& f2) {
      return cwiseProductOpLam(f1, f2);
    });
    m.def("cwiseProduct", [cwiseProductOpLam](const Gen& f1, const Vector3<double>& f2) {
      return cwiseProductOpLam(f1, f2);
    });

    m.def("cwiseProduct", [cwiseProductOpLam](const Vector2<double>& f2, const SEG2& f1) {
      return cwiseProductOpLam(f1, f2);
    });
    m.def("cwiseProduct", [cwiseProductOpLam](const Vector3<double>& f2, const SEG3& f1) {
      return cwiseProductOpLam(f1, f2);
    });
    m.def("cwiseProduct", [cwiseProductOpLam](const VectorX<double>& f2, const SEG& f1) {
      return cwiseProductOpLam(f1, f2);
    });
    m.def("cwiseProduct", [cwiseProductOpLam](const VectorX<double>& f2, const Gen& f1) {
      return cwiseProductOpLam(f1, f2);
    });

    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG2& f1, const SEG2& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG2& f1, const SEG& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG2& f1, const Gen& f2) { return cwiseProductOpLam(f1, f2); });

    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG3& f1, const SEG3& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG3& f1, const SEG& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG3& f1, const Gen& f2) { return cwiseProductOpLam(f1, f2); });

    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG& f1, const SEG& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG& f1, const SEG2& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG& f1, const SEG3& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const SEG& f1, const Gen& f2) { return cwiseProductOpLam(f1, f2); });

    m.def("cwiseProduct",
          [cwiseProductOpLam](const Gen& f1, const SEG2& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const Gen& f1, const SEG3& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const Gen& f1, const SEG& f2) { return cwiseProductOpLam(f1, f2); });
    m.def("cwiseProduct",
          [cwiseProductOpLam](const Gen& f1, const Gen& f2) { return cwiseProductOpLam(f1, f2); });

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////cwiseProduct////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto cwiseQuotientOpLam = [](const auto& f1, const auto& f2) {
      py::object fun1 = py::cast(f1);
      py::object fun2 = py::cast(f2);
      return fun1.attr("cwiseQuotient")(fun2);
    };

    m.def("cwiseQuotient", [cwiseQuotientOpLam](const SEG2& f1, const Vector2<double>& f2) {
      return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwiseQuotient", [cwiseQuotientOpLam](const SEG3& f1, const Vector3<double>& f2) {
      return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwiseQuotient", [cwiseQuotientOpLam](const SEG& f1, const Vector3<double>& f2) {
      return cwiseQuotientOpLam(f1, f2);
    });
    m.def("cwiseQuotient", [cwiseQuotientOpLam](const Gen& f1, const Vector3<double>& f2) {
      return cwiseQuotientOpLam(f1, f2);
    });

    m.def("cwiseQuotient", [cwiseQuotientOpLam](const Vector2<double>& f1, const SEG2& f2) {
      Constant<-1, -1> f1tmp(f2.IRows(), f1);
      return cwiseQuotientOpLam(f1tmp, f2);
    });
    m.def("cwiseQuotient", [cwiseQuotientOpLam](const Vector3<double>& f1, const SEG3& f2) {
      Constant<-1, -1> f1tmp(f2.IRows(), f1);
      return cwiseQuotientOpLam(f1tmp, f2);
    });
    m.def("cwiseQuotient", [cwiseQuotientOpLam](const VectorX<double>& f1, const SEG& f2) {
      Constant<-1, -1> f1tmp(f2.IRows(), f1);
      return cwiseQuotientOpLam(f1tmp, f2);
    });
    m.def("cwiseQuotient", [cwiseQuotientOpLam](const VectorX<double>& f1, const Gen& f2) {
      Constant<-1, -1> f1tmp(f2.IRows(), f1);
      return cwiseQuotientOpLam(f1tmp, f2);
    });

    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG2& f1, const SEG2& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG2& f1, const SEG& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG2& f1, const Gen& f2) { return cwiseQuotientOpLam(f1, f2); });

    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG3& f1, const SEG3& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG3& f1, const SEG& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG3& f1, const Gen& f2) { return cwiseQuotientOpLam(f1, f2); });

    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG& f1, const SEG& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG& f1, const SEG2& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG& f1, const SEG3& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const SEG& f1, const Gen& f2) { return cwiseQuotientOpLam(f1, f2); });

    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const Gen& f1, const SEG2& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const Gen& f1, const SEG3& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const Gen& f1, const SEG& f2) { return cwiseQuotientOpLam(f1, f2); });
    m.def("cwiseQuotient",
          [cwiseQuotientOpLam](const Gen& f1, const Gen& f2) { return cwiseQuotientOpLam(f1, f2); });


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////Dot  Product////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    auto dotOpLam = [](const auto& f1, const auto& f2) {
      py::object fun1 = py::cast(f1);
      py::object fun2 = py::cast(f2);
      return fun1.attr("dot")(fun2);
    };


    m.def("dot", [dotOpLam](const SEG2& f1, const Vector2<double>& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG3& f1, const Vector3<double>& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG& f1, const Vector3<double>& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen& f1, const Vector3<double>& f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const Vector2<double>& f2, const SEG2& f1) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Vector3<double>& f2, const SEG3& f1) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const VectorX<double>& f2, const SEG& f1) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const VectorX<double>& f2, const Gen& f1) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const SEG2& f1, const SEG2& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG2& f1, const SEG& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG2& f1, const Gen& f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const SEG3& f1, const SEG3& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG3& f1, const SEG& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG3& f1, const Gen& f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const SEG& f1, const SEG& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG& f1, const SEG2& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG& f1, const SEG3& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const SEG& f1, const Gen& f2) { return dotOpLam(f1, f2); });

    m.def("dot", [dotOpLam](const Gen& f1, const SEG2& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen& f1, const SEG3& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen& f1, const SEG& f2) { return dotOpLam(f1, f2); });
    m.def("dot", [dotOpLam](const Gen& f1, const Gen& f2) { return dotOpLam(f1, f2); });
  }

  struct QuatRotation {

    static auto Definition() {
      auto args = Arguments<7>();
      auto q = args.head<4>();
      auto qinv = StackedOutputs {q.head<3>() * (-1.0), q.coeff<3>()};
      auto vq = args.tail<3>().padded_lower<1>();
      auto qvq = FunctionQuatProduct<decltype(q), decltype(vq)> {q, vq};
      auto expr = FunctionQuatProduct<decltype(qvq), decltype(qinv)> {qvq, qinv}.head<3>();
      return expr;
    }
  };


}  // namespace ASSET


void ASSET::FreeFunctionsBuild(FunctionRegistry& reg, py::module& m) {
  using Gen = GenericFunction<-1, -1>;
  using GenS = GenericFunction<-1, 1>;

  using ARGS = Arguments<-1>;
  using SEG = Segment<-1, -1, -1>;
  using SEG2 = Segment<-1, 2, -1>;
  using SEG3 = Segment<-1, 3, -1>;
  using SEG4 = Segment<-1, 4, -1>;
  using ELEM = Segment<-1, 1, -1>;
  using GenCon = GenericConditional<-1>;

  /////////////////////////////////////////////////////////////////////

  m.def("normalize", [](const VectorX<double>& x) { return x.normalized(); });
  m.def("normalized", [](const VectorX<double>& x) { return x.normalized(); });


  /*
   These methods are already bound member functions for all valid asset vector functions in their build
   functions. Im simply calling them here thru pybind to prevent rebuilding those objects
  */


  UnaryVectorOpBuild<ARGS>(m);
  UnaryVectorOpBuild<SEG>(m);
  UnaryVectorOpBuild<SEG2>(m);
  UnaryVectorOpBuild<SEG3>(m);
  UnaryVectorOpBuild<Gen>(m);

  ProductBuild(m);


  ////////////////////////////////////////////////////////////////////////

  m.def("ScalarRootFinder", [](const GenS& fx, const GenS& dfx, int iter, double tol) {
    return GenS(ScalarRootFinder<GenS, GenS> {fx, dfx, iter, tol});
  });
  m.def("ScalarRootFinder", [](const GenS& fx, int iter, double tol) {
    return GenS(ScalarRootFinder<GenS, std::false_type> {fx, std::false_type(), iter, tol});
  });
  ////////////////////////////////////////////////////////////////////////


  m.def("quatProduct",
        [](const SEG& seg1, const SEG& seg2) { return Gen(FunctionQuatProduct<SEG, SEG>(seg1, seg2)); });
  m.def("quatProduct",
        [](const SEG& seg1, const Gen& seg2) { return Gen(FunctionQuatProduct<SEG, Gen>(seg1, seg2)); });
  m.def("quatProduct",
        [](const Gen& seg1, const SEG& seg2) { return Gen(FunctionQuatProduct<Gen, SEG>(seg1, seg2)); });
  m.def("quatProduct",
        [](const Gen& seg1, const Gen& seg2) { return Gen(FunctionQuatProduct<Gen, Gen>(seg1, seg2)); });

  m.def("quatRotate",
        [](const Gen& q, const Gen& v) { return Gen(QuatRotation::Definition().eval(stack(q, v))); });
  m.def("quatRotate",
        [](const Gen& q, const SEG3& v) { return Gen(QuatRotation::Definition().eval(stack(q, v))); });
  m.def("quatRotate",
        [](const Gen& q, const SEG& v) { return Gen(QuatRotation::Definition().eval(stack(q, v))); });
  m.def("quatRotate", [](const Gen& q, const Vector3<double>& vec) {
    return Gen(QuatRotation::Definition().eval(stack(q, Constant<-1, 3>(q.IRows(), vec))));
  });

  /////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////


  UnaryScalarOpBuild<ELEM>(m);
  UnaryScalarOpBuild<GenS>(m);

  GenCon::IfElseBuild(m);


  m.def("arctan2", [](const GenS& y, const GenS& x) {
    return GenS(ArcTan2Op().eval(StackedOutputs {y, x}));
  });
  m.def("arctan2", [](const ELEM& y, const ELEM& x) {
    return GenS(ArcTan2Op().eval(StackedOutputs {y, x}));
  });
  m.def("arctan2", [](const ELEM& y, const GenS& x) {
    return GenS(ArcTan2Op().eval(StackedOutputs {y, x}));
  });
  m.def("arctan2", [](const GenS& y, const ELEM& x) {
    return GenS(ArcTan2Op().eval(StackedOutputs {y, x}));
  });


  m.def("divtest", [](const Gen& a, const GenS& b) {
    return Gen(VectorScalarFunctionDivision<Gen, GenS> {a, b});
  });
  m.def("divtest", [](const Gen& a, const ELEM& b) {
    return Gen(VectorScalarFunctionDivision<Gen, ELEM> {a, b});
  });

  ////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////
}