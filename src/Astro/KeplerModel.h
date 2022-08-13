#pragma once
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "OptimalControl/ODE.h"
#include "OptimalControl/ODEPhase.h"
#include "KeplerPropagator.h"


namespace ASSET {

    struct Kepler_Impl : ODESize<6, 0, 0> {
        static auto Definition(double mu) {
            auto args = Arguments<7>();
            auto X = args.head<3>();
            auto V = args.segment<3, 3>();
            auto rvec = X;
            auto acc =  (-mu) * rvec.normalized_power<3>();
            auto ode = StackedOutputs{ V, acc };
            return ode;
        }
    };
    struct KeplerPhase;


    struct Kepler : ODE_Expression<Kepler, Kepler_Impl, double> {
        using Base = ODE_Expression<Kepler, Kepler_Impl, double>;
        using Base::Base;
        double mu = 1.0;
        Kepler(double mu) :Base(mu) {
            this->mu = mu;
        }

        static void Build(py::module& m, const char* name) {
            auto obj = py::class_<Kepler>(m, name).def(py::init<double>());
            Base::DenseBaseBuild(obj);
            obj.def("phase", [](const Kepler& od, TranscriptionModes Tmode) {
                return std::make_shared<KeplerPhase>(od, Tmode);
                });
            obj.def("phase", [](const Kepler& od, TranscriptionModes Tmode,
                        const std::vector<Eigen::VectorXd>& Traj, int numdef) {
                    return std::make_shared<KeplerPhase>(od, Tmode, Traj, numdef);
                });
            obj.def("integrator", [](const Kepler& od, double ds) {
                return RKIntegrator<Kepler>(od, ds);
                });
           

        }

    };



    struct KeplerPhase :ODEPhase<Kepler> {
        using Base = ODEPhase<Kepler>;
        using Base::Base;
        bool UseKeplerPropagator = true;


        ASSET::ConstraintInterface make_shooter() {
            if (UseKeplerPropagator) {
                auto kprop = KeplerPropagator(this->ode.mu);
                auto Args = Arguments<14>();
                auto X1 = Args.head<6>();
                auto X2 = Args.segment<6, 7>();
                auto t1 = Args.coeff<6>();
                auto t2 = Args.coeff<13>();

              

                auto Xk1 = StackedOutputs{ X1,(t2 - t1) / 2.0 };
                auto Xk2 = StackedOutputs{ X2,(t1 - t2) / 2.0 };

                auto shooter = kprop.eval(Xk1) - kprop.eval(Xk2);
                
                return ASSET::ConstraintInterface(shooter);
            }
            else {
                return Base::make_shooter();
            }

        }

        static void Build(py::module& m) {
            auto phase = py::class_<KeplerPhase, std::shared_ptr<KeplerPhase>,
                ODEPhaseBase>(m, "phase");
            phase.def(py::init<Kepler, TranscriptionModes>());
            phase.def(py::init<Kepler, TranscriptionModes,
                const std::vector<Eigen::VectorXd>&, int>());
            phase.def_readwrite("integrator", &KeplerPhase::integrator);
            phase.def_readwrite("UseKeplerPropagator", &KeplerPhase::UseKeplerPropagator);

        }


    };


    static void BuildKeplerMod(FunctionRegistry& reg, py::module& m) {
        auto odemod = m.def_submodule("Kepler");
        reg.template Build_Register<Kepler>(odemod, "ode");
        reg.template Build_Register<RKIntegrator<Kepler>>(odemod, "integrator");
        reg.Build_Register<KeplerPropagator>(odemod, "KeplerPropagator");
        KeplerPhase::Build(odemod);
    }


}