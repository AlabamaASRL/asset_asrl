#pragma once
#include "ODESizes.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {

  template<int _XV, int _UV, int _PV>
  struct ODEArguments : Arguments<ODESize<_XV, _UV, _PV>::XtUPV>, ODESize<_XV, _UV, _PV> {


    using Base = Arguments<ODESize<_XV, _UV, _PV>::XtUPV>;

    ODEArguments(int Xv, int Uv, int Pv) : Base(Xv + Uv + Pv + 1) {
      this->setXtUPVars(Xv, Uv, Pv);
    }
    ODEArguments(int Xv, int Uv) : ODEArguments(Xv, Uv, 0) {
    }
    ODEArguments(int Xv) : ODEArguments(Xv, 0) {
    }


    static void Build(py::module& m, const char* name) {
      using Derived = ODEArguments<_XV, _UV, _PV>;
      auto obj = py::class_<ODEArguments<_XV, _UV, _PV>, Base>(m, name);
      obj.def(py::init<int, int, int>());
      obj.def(py::init<int, int>());
      obj.def(py::init<int>());

      Base::DenseBaseBuild(obj);

      obj.def("XVec", [](const Derived& a) { return a.segment(0, a.XVars()); });
      obj.def("XVar", [](const Derived& a, int i) { return a.segment(0, a.XVars()).coeff(i); });
      obj.def("XtVec", [](const Derived& a) { return a.segment(0, a.XtVars()); });
      obj.def("TVar", [](const Derived& a) { return a.coeff(a.TVar()); });
      obj.def("UVec", [](const Derived& a) { return a.segment(a.XtVars(), a.UVars()); });
      obj.def("UVar", [](const Derived& a, int i) { return a.segment(a.XtVars(), a.UVars()).coeff(i); });

      obj.def("PVec", [](const Derived& a) { return a.segment(a.XtUVars(), a.PVars()); });
      obj.def("PVar", [](const Derived& a, int i) { return a.segment(a.XtUVars(), a.PVars()).coeff(i); });
    }
  };


}  // namespace ASSET
