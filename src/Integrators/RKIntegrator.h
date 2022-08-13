#pragma once

#include "IntegratorBase.h"
#include "RKCoeffs.h"
#include "RKSteppers.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "OptimalControl/LGLInterpTable.h"
#include "OptimalControl/LGLInterpFunctions.h"

namespace ASSET {


    template <class BaseType, int _XV, int _UV, int _PV>
    struct GenericODE;
    template<int OR>
    struct InterpFunction;

template <class DODE>
struct StepperWrapper {
  using type = typename std::conditional<
      DODE::UV == 0 && DODE::PV == 0,
      GenericFunction<SZ_SUM<DODE::IRC, 1>::value, DODE::IRC>,
      GenericFunction<-1, DODE::IRC>>::type;
};


template <class DODE>
struct RKIntegrator : IntegratorBase<RKIntegrator<DODE>, DODE,
                                     typename StepperWrapper<DODE>::type> {
  using Base = IntegratorBase<RKIntegrator<DODE>, DODE,
                              typename StepperWrapper<DODE>::type>;
  using Base::Base;
  using Base::init;

  using StepperType = typename StepperWrapper<DODE>::type;

  template <class D, RKOptions Op>
  using BaseStepper = RKStepper_NEW<D, Op>;
    //  typename std::conditional<DODE::IsGenericODE, RKStepper2<D, Op>,
    //                            RKStepper<D, Op>>::type;




  RKIntegrator(const DODE& ode, double ds)
      : Base(ode, BaseStepper<DODE, RK4Classic>(ode), ds){};
  RKIntegrator(const DODE& ode, double ds, RKOptions rkop) {
    switch (rkop) {
      case RK4Classic:
        this->init(ode, BaseStepper<DODE, RK4Classic>(ode), ds);
        break;
      case DOPRI54:
        this->init(ode, BaseStepper<DODE, RK4Classic>(ode), ds);
        break;
      default:
        this->init(ode, BaseStepper<DODE, RK4Classic>(ode), ds);
    }
  };
  RKIntegrator(const DODE& ode, double ds, const GenericFunction<-1, -1>& ucon,
               const Eigen::VectorXi& varlocs, const Eigen::VectorXi& plocs) {
    if constexpr (DODE::UV != 0) {
      Eigen::VectorXi empty;
      empty.resize(0);
      this->init(ode, MakeStepper<RK4Classic>(ode, false, ucon, varlocs, plocs),
                 ds);
    }
  }
  RKIntegrator(const DODE& ode, double ds, const GenericFunction<-1, -1>& ucon,
               const Eigen::VectorXi& varlocs) {
    if constexpr (DODE::UV != 0) {
      Eigen::VectorXi empty;
      empty.resize(0);
      this->init(ode, MakeStepper<RK4Classic>(ode, false, ucon, varlocs, empty),
                 ds);
    }
  }
  RKIntegrator(const DODE& ode, double ds,std::shared_ptr<LGLInterpTable> tab,
      const Eigen::VectorXi& ulocs) {
      if constexpr (DODE::UV !=0) {
          Eigen::VectorXi empty;
          empty.resize(0);
          Eigen::VectorXi tloc(1);
          tloc[0] = ode.TVar();
          GenericFunction<-1, -1> ucon = InterpFunction<-1>(tab, ulocs);
          this->init(ode, MakeStepper<RK4Classic>(ode, false, ucon, tloc, empty),
              ds);
      }
  }
  RKIntegrator(const DODE& ode, double ds, const Eigen::VectorXd& v) {
      if constexpr (DODE::UV != 0) {
          Eigen::VectorXi empty;
          empty.resize(0);
          Eigen::VectorXi tloc(1);
          tloc[0] = ode.TVar();
          GenericFunction<-1, -1> ucon = Constant<-1, -1>(1, v);
          this->init(ode, MakeStepper<RK4Classic>(ode, false, ucon, tloc, empty),
              ds);
      }
  }



  template <RKOptions RKOp>
  static auto MakeStepper(const DODE& ode, bool nocontrol,
                          const GenericFunction<-1, -1>& ucon,
                          const Eigen::VectorXi& varlocs,
                          const Eigen::VectorXi& plocs) {
    auto Stepper = BaseStepper<DODE, RKOp>(ode);
    constexpr int IRC = decltype(Stepper)::IRC;
    constexpr int DUV = (DODE::UV == 1) ? -1 : DODE::UV;
    if constexpr (DODE::UV == 0) {
      return StepperType(Stepper);

    } else {
      if (nocontrol) {
        return StepperType(Stepper);
      } 
      else {

        if (ucon.ORows() != ode.UVars()) {
            throw std::invalid_argument("Controller output size does not match number of ode control variables");
        }
        
        if (plocs.size() == 0) {

            if (ucon.IRows() != varlocs.size()) {
                throw std::invalid_argument("Controller input size is inconsistent with specified number of input state variables");
            }


          if constexpr (DODE::PV == 0) {
              Arguments<IRC> stepperargs(Stepper.IRows());
              Arguments<DODE::IRC> odeargs(ode.IRows());

              ParsedInput<GenericFunction<-1, -1>, IRC, DUV> stepfunc(ucon, varlocs, Stepper.IRows());
              ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> odefunc(ucon, varlocs, ode.IRows());

              auto ODEargs = StackedOutputs{ odeargs.template head<DODE::XtV>(ode.XtVars()),odefunc };
              auto StepArgs = StackedOutputs{ stepperargs.template head<DODE::XtV>(ode.XtVars()),stepfunc, stepperargs.template tail<1>() };

              auto ODEexpr = NestedFunction<DODE, decltype(ODEargs)>(ode, ODEargs);
              auto GenOde = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(ODEexpr, ode.XVars(), ode.UVars(), ode.PVars());
              auto Stepper2 = StepperType(BaseStepper<decltype(GenOde), RKOp>(GenOde)).eval(StepArgs);

              auto Stepper3 = StepperType(ODEargs.eval(BaseStepper<decltype(GenOde), RKOp>(GenOde)));


              return StepperType(Stepper3);
          }
          else {


              Arguments<IRC> stepperargs(Stepper.IRows());
              Arguments<DODE::IRC> odeargs(ode.IRows());

              ParsedInput<GenericFunction<-1, -1>, IRC, DUV> stepfunc(ucon, varlocs, Stepper.IRows());
              ParsedInput<GenericFunction<-1, -1>, DODE::IRC, DUV> odefunc(ucon, varlocs, ode.IRows());

              auto ODEargs  = StackedOutputs{ odeargs.template head<DODE::XtV>(ode.XtVars()),odefunc,odeargs.template tail<-1>(ode.PVars()) };
              auto StepArgs = StackedOutputs{ stepperargs.template head<DODE::XtV>(ode.XtVars()),stepfunc, stepperargs.template tail<-1>(ode.PVars()+1) };

              auto ODEexpr = NestedFunction<DODE, decltype(ODEargs)>(ode, ODEargs);
              auto GenOde = GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>(ODEexpr, ode.XVars(), ode.UVars(), ode.PVars());
              auto Stepper2 = StepperType(BaseStepper<decltype(GenOde), RKOp>(GenOde)).eval(StepArgs);

              return StepperType(Stepper2);

          }
          
        } 
        else {
         
          return StepperType(Stepper);
        }
      }
    }
  }
  static void Build(py::module& m, const char* name) {
    auto obj =
        py::class_<RKIntegrator>(m, name).def(py::init<const DODE&, double>());
    obj.def(py::init<const DODE&, double, RKOptions>());
    if constexpr (DODE::UV != 0) {
      obj.def(py::init<const DODE&, double, const GenericFunction<-1, -1>&,
                       const Eigen::VectorXi&>());
      obj.def(py::init<const DODE&, double, std::shared_ptr<LGLInterpTable>,
          const Eigen::VectorXi&>());
      obj.def(py::init<const DODE&, double, const Eigen::VectorXd&>());
    }

    Base::DenseBaseBuild(obj);
    Base::IntegratorAPIBuild(obj);
  }
};

template <class DODE>
struct InlineRK4
    : IntegratorBase<InlineRK4<DODE>, DODE, RKStepper<DODE, RK4Classic>> {
  using Base =
      IntegratorBase<InlineRK4<DODE>, DODE, RKStepper<DODE, RK4Classic>>;
  InlineRK4() {}
  InlineRK4(const DODE& ode, double ds)
      : Base(ode, RKStepper<DODE, RK4Classic>(ode), ds) {}
};
}  // namespace ASSET
