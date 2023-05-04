#pragma once

#include "ODEPhase.h"
#include "ODESizes.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"

namespace ASSET {

  template<class BaseType, class Derived, int _XV, int _UV, int _PV>
  struct ODEBase;

  template<class Derived,
           int _XV,
           int _UV,
           int _PV,
           DenseDerivativeModes Jm = DenseDerivativeModes::Analytic,
           DenseDerivativeModes Hm = DenseDerivativeModes::Analytic>
  struct ODE : ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                       Derived,
                       _XV,
                       _UV,
                       _PV> {
    using Base = ODEBase<VectorFunction<Derived, SZ_SUM<_XV, 1, _UV, _PV>::value, _XV, Jm, Hm>,
                         Derived,
                         _XV,
                         _UV,
                         _PV>;

    void setODESize(int xv, int uv, int pv) {
      this->setXVars(xv);
      this->setUVars(uv);
      this->setPVars(pv);
      this->setInputRows(this->XtUPVars());
      this->setOutputRows(this->XVars());
    }
  };

  template<class Derived, class ExprImpl, class... Ts>
  struct ODE_Expression : ODEBase<VectorExpression<Derived, ExprImpl, Ts...>,
                                  Derived,
                                  ExprImpl::XV,
                                  ExprImpl::UV,
                                  ExprImpl::PV> {
    using Base = ODEBase<VectorExpression<Derived, ExprImpl, Ts...>,
                         Derived,
                         ExprImpl::XV,
                         ExprImpl::UV,
                         ExprImpl::PV>;
    using Base::Base;

    void setODESize(int xv, int uv, int pv) {
      this->setXVars(xv);
      this->setUVars(uv);
      this->setPVars(pv);
    }

    static void Build(py::module& m, const char* name) {
      auto obj = py::class_<Derived>(m, name).def(py::init<Ts...>());
      Base::DenseBaseBuild(obj);
      obj.def("phase", [](const Derived& od, TranscriptionModes Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
      });
      Integrator<Derived>::BuildConstructors(obj);
    }
  };

#define BUILD_ODE_FROM_EXPRESSION(NAME, IMPL, ...)        \
  struct NAME : ODE_Expression<NAME, IMPL, __VA_ARGS__> { \
    using Base = ODE_Expression<NAME, IMPL, __VA_ARGS__>; \
    using Base::Base;                                     \
  };

  template<class BaseType, class Derived, int _XV, int _UV, int _PV>
  struct ODEBase : BaseType, ODESize<_XV, _UV, _PV> {
    using Base = BaseType;
    using Base::Base;
    static const bool IsGenericODE = false;

    Integrator<Derived> integrator(double dstep) const {
      return Integrator<Derived>(this->derived(), dstep);
    }

    static void BuildODEModule(const char* name, py::module& mod, FunctionRegistry& reg) {
      auto odemod = mod.def_submodule(name);
      reg.template Build_Register<Derived>(odemod, "ode");
      reg.template Build_Register<Integrator<Derived>>(odemod, "integrator");
      ODEPhase<Derived>::Build(odemod);
    }

    static void BuildODEModule(const char* name, FunctionRegistry& reg) {
      BuildODEModule(name, reg.mod, reg);
    }
  };

  template<class BaseType, int _XV, int _UV, int _PV>
  struct GenericODE
      : FunctionHolder<GenericODE<BaseType, _XV, _UV, _PV>, BaseType, SZ_SUM<_XV, _UV, _PV, 1>::value, _XV>,
        ODESize<_XV, _UV, _PV> {
    using Base =
        FunctionHolder<GenericODE<BaseType, _XV, _UV, _PV>, BaseType, SZ_SUM<_XV, _UV, _PV, 1>::value, _XV>;
    using Base::Base;

    static const bool IsGenericODE = true;

    GenericODE(BaseType f, int xv, int uv, int pv) : Base(f) {
      this->setXVars(xv);
      this->setUVars(uv);
      this->setPVars(pv);

      if (this->ORows() != xv) {
        throw std::invalid_argument("Output Size of Generic ODE Expression does not match the specified "
                                    "model size");
      }
      if (this->IRows() != (xv + uv + pv + 1)) {
        throw std::invalid_argument("Input Size of Generic ODE Expression does not match the specified "
                                    "model size");
      }
    }

    GenericODE(BaseType f, int xv, int uv) : GenericODE(f, xv, uv, 0) {
    }

    GenericODE(BaseType f, int xv) : GenericODE(f, xv, 0, 0) {
    }
    GenericODE(BaseType f) : GenericODE(f, _XV, _UV, _PV) {
    }

    static void BuildGenODEModule(const char* name, py::module& mod, FunctionRegistry& reg) {
      using Derived = GenericODE<BaseType, _XV, _UV, _PV>;
      auto odemod = mod.def_submodule(name);

      auto obj = py::class_<Derived>(odemod, "ode");
      obj.def(py::init<BaseType, int, int, int>());
      obj.def(py::init<BaseType, int, int>());
      obj.def(py::init<BaseType, int>());
      obj.def(py::init<BaseType>());
      ODEPhase<Derived>::Build(odemod);
      obj.def("phase", [](const Derived& od, TranscriptionModes Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
      });
      obj.def("phase",
              [](const Derived& od,
                 TranscriptionModes Tmode,
                 const std::vector<Eigen::VectorXd>& Traj,
                 int numdef) { return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef); });

      obj.def("phase", [](const Derived& od, std::string Tmode) {
        return std::make_shared<ODEPhase<Derived>>(od, Tmode);
      });
      obj.def("phase",
              [](const Derived& od, std::string Tmode, const std::vector<Eigen::VectorXd>& Traj, int numdef) {
                return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef);
              });
      obj.def(
          "phase",
          [](const Derived& od,
             std::string Tmode,
             const std::vector<Eigen::VectorXd>& Traj,
             int numdef,
             bool LerpIG) { return std::make_shared<ODEPhase<Derived>>(od, Tmode, Traj, numdef, LerpIG); });


      py::implicitly_convertible<Derived, GenericFunction<-1, -1>>();
      reg.vfuncx.def(
          py::init([](const Derived& ode) { return std::make_unique<GenericFunction<-1, -1>>(ode.func); }));

      reg.template Build_Register<Integrator<Derived>>(odemod, "integrator");

      Integrator<Derived>::BuildConstructors(obj);

      ODESize<_XV, _UV, _PV>::template BuildODESizeMembers<decltype(obj), Derived>(obj);


      obj.def("vf", [](const Derived& od) { return od.func; });


      obj.def("shooting_defect", [](const Derived& ode, const Integrator<Derived>& integ) {
        auto shooter = CentralShootingDefect<Derived, Integrator<Derived>>(ode, integ);
        return GenericFunction<-1, -1>(shooter);
      });
    }
  };


  template<int XV, int UV, int PV>
  using PythonGenericODE = GenericODE<GenericFunction<-1, -1>, XV, UV, PV>;


}  // namespace ASSET
