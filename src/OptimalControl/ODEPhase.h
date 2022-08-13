#pragma once

#include "Blocked_ODE_Wrapper.h"
#include "Integrators/RKIntegrator.h"
#include "LGLDefects.h"
#include "ODEPhaseBase.h"
#include "ShootingDefects.h"
#include "TrapezoidalDefects.h"
#include "pch.h"

namespace ASSET {

template <class DODE>
struct ODEPhase : ODEPhaseBase {
  using VectorXi = Eigen::VectorXi;
  using MatrixXi = Eigen::MatrixXi;

  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;

  using VectorFunctionalX = GenericFunction<-1, -1>;
  using ScalarFunctionalX = GenericFunction<-1, 1>;

  using StateConstraint = StateFunction<VectorFunctionalX>;
  using StateObjective = StateFunction<ScalarFunctionalX>;

  DODE ode;
  RKIntegrator<DODE> integrator;
  bool EnableHessianSparsity = false;

  template<class ODET,int CS>
  using LGLType = LGLDefects<ODET, CS>;


  ODEPhase(const DODE& ode, TranscriptionModes Tmode)
      : ODEPhaseBase(ode.XVars(), ode.UVars(), ode.PVars()) {
    this->ode = ode;
    this->integrator = RKIntegrator<DODE>(ode, .01);
    this->setTranscriptionMode(Tmode);
  }
  ODEPhase(const DODE& ode, TranscriptionModes Tmode,
           const std::vector<Eigen::VectorXd>& Traj, int numdef)
      : ODEPhase(ode, Tmode) {
    this->setTraj(Traj, numdef);
  }

  ODEPhase(const DODE& ode, std::string Tmode)
      : ODEPhase(ode, strto_TranscriptionMode(Tmode)) {
      
  }
  ODEPhase(const DODE& ode, std::string Tmode,
      const std::vector<Eigen::VectorXd>& Traj, int numdef)
      :ODEPhase(ode, strto_TranscriptionMode(Tmode),Traj,  numdef) {
     
  }

  virtual void setTranscriptionMode(TranscriptionModes m) {
    this->resetTranscription();

    this->TranscriptionMode = m;

    VectorFunctionalX odetemp;

    if constexpr (DODE::IsGenericODE) {
      odetemp = this->ode.func;
    } else {
      odetemp = this->ode;
    }

    switch (this->TranscriptionMode) {
      case TranscriptionModes::LGL7:
        this->Table = LGLInterpTable(odetemp, this->XVars(),
                                     this->UVars() + this->PVars(), m);
        this->numTranCardStates = 4;
        break;
      case TranscriptionModes::LGL5:
        this->Table = LGLInterpTable(odetemp, this->XVars(),
                                     this->UVars() + this->PVars(), m);
        this->numTranCardStates = 3;
        break;
      case TranscriptionModes::LGL3:
        this->Table = LGLInterpTable(odetemp, this->XVars(),
                                     this->UVars() + this->PVars(), m);
        this->numTranCardStates = 2;
        break;
      case TranscriptionModes::Trapezoidal:
        this->Table = LGLInterpTable(odetemp, this->XVars(),
                                     this->UVars() + this->PVars(),
                                     TranscriptionModes::LGL3);
        this->numTranCardStates = 2;
        break;
      case TranscriptionModes::CentralShooting:
        this->Table = LGLInterpTable(odetemp, this->XVars(),
                                     this->UVars() + this->PVars(),
                                     TranscriptionModes::LGL3);
       // throw std::invalid_argument("Central Shooting has been disabled for the initial release");

        this->numTranCardStates = 2;
        break;
      default: {
        throw std::invalid_argument("Invalid Transcription Method");
        break;
      }
    }
  }

  virtual void transcribe_dynamics() {
    VectorXi StateT(this->ode.XtUVars());
    for (int i = 0; i < this->ode.XtUVars(); i++) StateT[i] = i;
    VectorXi OParT(this->ode.PVars());
    for (int i = 0; i < this->ode.PVars(); i++) OParT[i] = i;
    VectorXi empty(0);
    empty.resize(0);


    auto lgldef = [&](auto cs) {
        if (this->ControlMode == BlockConstant) {

            if constexpr (DODE::UV == 0 && DODE::PV == 0) {
                LGLType<DODE, cs.value> lgl(this->ode);
                lgl.EnableVectorization = this->EnableVectorization;
                this->DynamicsFuncIndex = this->indexer.addEquality(
                    lgl, PhaseRegionFlags::DefectPath, StateT, OParT, empty,
                    ThreadingFlags::ByApplication);
            }
            else {
                LGLType<Blocked_ODE_Wrapper<DODE>, cs.value> lgl(
                    Blocked_ODE_Wrapper<DODE>(this->ode));
                lgl.EnableVectorization = this->EnableVectorization;
                this->DynamicsFuncIndex = this->indexer.addEquality(
                    lgl, PhaseRegionFlags::BlockDefectPath, StateT, OParT, empty,
                    ThreadingFlags::ByApplication);
            }
        }
        else {
            LGLType<DODE, cs.value> lgl(this->ode);
            lgl.EnableVectorization = this->EnableVectorization;
            this->DynamicsFuncIndex = this->indexer.addEquality(
                lgl, PhaseRegionFlags::DefectPath, StateT, OParT, empty,
                ThreadingFlags::ByApplication);
        }
    };


    switch (this->TranscriptionMode) {
      case TranscriptionModes::LGL7: {
        lgldef(int_const<4>());
        break;
      }
      case TranscriptionModes::LGL5: {
        lgldef(int_const<3>());
        break;
      }
      case TranscriptionModes::LGL3: {
        lgldef(int_const<2>());
        break;
      }
      case TranscriptionModes::Trapezoidal: {
          if (this->ControlMode == BlockConstant) {

              if constexpr (DODE::UV == 0 && DODE::PV == 0) {
                  TrapezoidalDefects<DODE> trap(this->ode);
                  trap.EnableVectorization = this->EnableVectorization;
                 // trap.EnableHessianSparsity = this->EnableHessianSparsity;
                  this->DynamicsFuncIndex =
                      this->indexer.addEquality(trap, PhaseRegionFlags::DefectPath, StateT,
                          OParT, empty,
                          ThreadingFlags::ByApplication);
              }
              else {
                  TrapezoidalDefects<Blocked_ODE_Wrapper<DODE>> trap(
                      Blocked_ODE_Wrapper<DODE>(this->ode));
                  trap.EnableVectorization = this->EnableVectorization;
                  trap.EnableHessianSparsity = this->EnableHessianSparsity;
                  this->DynamicsFuncIndex = this->indexer.addEquality(
                      trap, PhaseRegionFlags::BlockDefectPath, StateT, OParT, empty,
                      ThreadingFlags::ByApplication);

              }

              
          }
          else {
              TrapezoidalDefects<DODE> trap(this->ode);
              trap.EnableVectorization = this->EnableVectorization;
              //trap.EnableHessianSparsity = this->EnableHessianSparsity;
              this->DynamicsFuncIndex =
                  this->indexer.addEquality(trap, PhaseRegionFlags::DefectPath, StateT,
                      OParT, empty,
                      ThreadingFlags::ByApplication);
          }
        break;
      }
      case TranscriptionModes::CentralShooting: {
       
          auto shooter = this->make_shooter();
          this->indexer.addEquality(shooter, PhaseRegionFlags::DefectPath,
                                    StateT, OParT, empty,
                                    ThreadingFlags::ByApplication);

        
        break;
      }
      default:throw std::invalid_argument("Invalid Transcription Method");
    }
  }

  virtual ASSET::ConstraintInterface make_shooter() {
      auto Integ = RKIntegrator<DODE>{ this->ode, this->integrator.DefStepSize };
      Integ.Adaptive = this->integrator.Adaptive;
      Integ.FastAdaptiveSTM = this->integrator.FastAdaptiveSTM;
      Integ.AbsTols = this->integrator.AbsTols;
      Integ.MinStepSize = this->integrator.MinStepSize;
      Integ.MaxStepSize = this->integrator.MaxStepSize;
      Integ.EnableVectorization = this->EnableVectorization;

      auto shooter = ShootingDefect{ this->ode, Integ };
      shooter.EnableHessianSparsity = this->EnableHessianSparsity;

      return ASSET::ConstraintInterface(shooter);
  }


  static void Build(py::module& m) {
    auto phase = py::class_<ODEPhase<DODE>, std::shared_ptr<ODEPhase<DODE>>,
                            ODEPhaseBase>(m, "phase");
    phase.def(py::init<DODE, TranscriptionModes>());
    phase.def(py::init<DODE, TranscriptionModes,
                       const std::vector<Eigen::VectorXd>&, int>());
    phase.def(py::init<DODE, std::string>());
    phase.def(py::init<DODE, std::string,
        const std::vector<Eigen::VectorXd>&, int>());

    phase.def_readwrite("integrator", &ODEPhase<DODE>::integrator);
    phase.def_readwrite("EnableHessianSparsity",  &ODEPhase<DODE>::EnableHessianSparsity);
  }
};

}  // namespace ASSET
