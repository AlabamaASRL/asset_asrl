#pragma once

#include "OptimalControlFlags.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "pch.h"

namespace ASSET {

template <class DODE, class Integrator>
struct ShootingDefect_Impl {
  static auto Definition(const DODE& ode, const Integrator& integ) {
    constexpr int IRC = SZ_SUM<SZ_PROD<DODE::XtUV , 2>::value , DODE::PV>::value;
    int input_rows = ode.XtUVars() * 2 + ode.PVars();

    auto args = Arguments<IRC>(input_rows);
    // Input[x1,t1,u1,x2,t2,u2,pv]

    auto x1 = args.template head<DODE::XtUV>(ode.XtUVars());
    auto t1 = x1.template coeff<DODE::XV>(ode.XVars());
    auto x2 = args.template segment<DODE::XtUV, DODE::XtUV>(ode.XtUVars(),
                                                            ode.XtUVars());
    auto t2 = x2.template coeff<DODE::XV>(ode.XVars());

    auto tm = 0.5 * (t1 + t2);

    auto pvars = args.template tail<DODE::PV>(ode.PVars());

    auto make_state = [&](auto xx) {
        if constexpr (DODE::PV == 0) {
            return StackedOutputs{ xx, tm };
        }
        else {
            return StackedOutputs{ xx, tm, pvars };
        }
    };

    auto Arc1Input = make_state(x1);
    auto Arc2Input = make_state(x2);

    auto defect = integ.eval(Arc1Input).template head<DODE::XV>(ode.XVars()) -
                  integ.eval(Arc2Input).template head<DODE::XV>(ode.XVars());

    return defect;
  }
};

template <class DODE, class Integrator>
struct ShootingDefect : VectorExpression<ShootingDefect<DODE, Integrator>,
                                         ShootingDefect_Impl<DODE, Integrator>,
                                         const DODE&, const Integrator&> {
  using Base = VectorExpression<ShootingDefect<DODE, Integrator>,
                                ShootingDefect_Impl<DODE, Integrator>,
                                const DODE&, const Integrator&>;
  // using Base::Base;
  ShootingDefect() {}
  ShootingDefect(const DODE& ode, const Integrator& integ) : Base(ode, integ) {}
  bool EnableHessianSparsity = false;
};

}  // namespace ASSET
