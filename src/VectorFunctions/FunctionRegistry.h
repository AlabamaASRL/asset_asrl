#pragma once

#include "VectorFunctionTypeErasure/GenericFunction.h"
#include "pch.h"

namespace ASSET {

  template<class T>
  struct FuncPack {
    using type = T;
    std::string name;
    FuncPack(const T& t, std::string nm) : name(nm) {
    }
  };

  struct FunctionRegistry {
    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    py::module& mod;
    py::module vfmod;
    py::module ocmod;
    py::module solmod;

    py::class_<VectorFunctionalX> vfuncx;
    py::class_<ScalarFunctionalX> sfuncx;

    FunctionRegistry(py::module& m)
        : mod(m),
          vfmod(m.def_submodule("VectorFunctions",
                                "SubModule Containing Vector and Scalar Function Types and Functions")),
          ocmod(m.def_submodule("OptimalControl",
                                "SubModule Containing Optimal Control ODEs, Phases, and Utilities")),
          solmod(m.def_submodule("Solvers", "SubModule Containing PSIOPT,NLP, and Solver Flags")),

          vfuncx(py::class_<VectorFunctionalX>(this->vfmod, "VectorFunction")),
          sfuncx(py::class_<ScalarFunctionalX>(this->vfmod, "ScalarFunction")) {
    }

    py::module& getVectorFunctionsModule() {
      return this->vfmod;
    }
    py::module& getOptimalControlModule() {
      return this->ocmod;
    }
    py::module& getSolversModule() {
      return this->solmod;
    }

    template<int IR, int OR>
    struct RegSelector {
      template<class Derived>
      static void Register(FunctionRegistry* reg) {
        py::implicitly_convertible<Derived, VectorFunctionalX>();
        reg->vfuncx.def(py::init<Derived>());
      }
    };
    template<int IR>
    struct RegSelector<IR, 1> {
      template<class Derived>
      static void Register(FunctionRegistry* reg) {
        py::implicitly_convertible<Derived, VectorFunctionalX>();
        reg->vfuncx.def(py::init<Derived>());
        py::implicitly_convertible<Derived, ScalarFunctionalX>();
        reg->sfuncx.def(py::init<Derived>());
      }
    };

    template<class Derived>
    void Register() {
      RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template<class Derived>
    void Build_Register(py::module& m) {
      Derived::Build(m);
      RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template<class Derived>
    void Build_Register(const char* name) {
      Derived::Build(this->mod, name);
      RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
    template<class Derived>
    void Build_Register(py::module& m, const char* name) {
      Derived::Build(m, name);
      RegSelector<Derived::IRC, Derived::ORC>::template Register<Derived>(this);
    }
  };

}  // namespace ASSET
