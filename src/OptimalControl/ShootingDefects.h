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
            return StackedOutputs{ xx,pvars,tm};
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




template<class DODE,class Integrator>
struct CentralShootingDefect:
    VectorFunction< CentralShootingDefect<DODE,Integrator>, SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value,DODE::XV>{

    using Base = VectorFunction< CentralShootingDefect<DODE, Integrator>, SZ_SUM<SZ_PROD<DODE::XtUV, 2>::value, DODE::PV>::value, DODE::XV>;

    DENSE_FUNCTION_BASE_TYPES(Base);


    template <class Scalar>
    using ODEState = typename DODE::template Input<Scalar>;
    template <class Scalar>
    using ODEDeriv = typename DODE::template Output<Scalar>;
    bool EnableHessianSparsity = false;

    DODE ode;
    Integrator integ;


    CentralShootingDefect(const DODE& ode, const Integrator& integ):ode(ode),integ(integ) {

        this->setIORows(2 * this->ode.XtUVars() + this->ode.PVars(), this->ode.XVars());

    }

    std::vector<Output<double>> compute_impl_vectorized(const std::vector<Input<double>> & X1X2s) {


        std::vector<ODEState<double>> Xs(2 * X1X2s.size());

        for (int i = 0; i < X1X2s.size(); i++) {

            Xs[2 * i].resize(this->ode.IRows());
            Xs[2 * i + 1].resize(this->ode.IRows());


        }



    }



    template <class InType, class OutType>
    inline void compute_impl(ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();

    }
    template <class InType, class OutType, class JacType>
    inline void compute_jacobian_impl(ConstVectorBaseRef<InType> x,
        ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();


    }
    template <class InType, class OutType, class JacType, class AdjGradType,
        class AdjHessType, class AdjVarType>
    inline void compute_jacobian_adjointgradient_adjointhessian_impl(
        ConstVectorBaseRef<InType> x, ConstVectorBaseRef<OutType> fx_,
        ConstMatrixBaseRef<JacType> jx_, ConstVectorBaseRef<AdjGradType> adjgrad_,
        ConstMatrixBaseRef<AdjHessType> adjhess_,
        ConstVectorBaseRef<AdjVarType> adjvars) const {
        typedef typename InType::Scalar Scalar;
        VectorBaseRef<OutType> fx = fx_.const_cast_derived();
        MatrixBaseRef<JacType> jx = jx_.const_cast_derived();
        VectorBaseRef<AdjGradType> adjgrad = adjgrad_.const_cast_derived();
        MatrixBaseRef<AdjHessType> adjhess = adjhess_.const_cast_derived();

    }



    CentralShootingDefect(){
    
    }


};





}  // namespace ASSET
