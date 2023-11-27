#pragma once

#include "Blocked_ODE_Wrapper.h"
#include "Integrators/Integrator.h"
#include "LGLDefects.h"
#include "ODEPhaseBase.h"
#include "ShootingDefects.h"
#include "TrapezoidalDefects.h"
#include "pch.h"

namespace ASSET {

  template<class DODE>
  struct ODEPhase : ODEPhaseBase {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    using StateConstraint = StateFunction<VectorFunctionalX>;
    using StateObjective = StateFunction<ScalarFunctionalX>;

    template<class ODET, int CS>
    using LGLType = LGLDefects<ODET, CS>;

    template<class Scalar>
    using ODEState = typename DODE::template Input<Scalar>;
    template<class Scalar>
    using ODEDeriv = typename DODE::template Output<Scalar>;

    using ScaledODE =  GenericODE<GenericFunction<-1, -1>, DODE::XV, DODE::UV, DODE::PV>;


    DODE ode;
    ScaledODE ode_scaled;


    Integrator<DODE> integrator;
    bool EnableHessianSparsity = false;
    bool OldShootingDefect = false;

    ODEPhase(const DODE& ode, TranscriptionModes Tmode)
        : ODEPhaseBase(ode.XVars(), ode.UVars(), ode.PVars()) {
      this->ode = ode;
      this->integrator = Integrator<DODE>(ode, .01);
      this->set_idxs(this->ode.get_idxs());

      this->setTranscriptionMode(Tmode);
    }
    ODEPhase(const DODE& ode, TranscriptionModes Tmode, const std::vector<Eigen::VectorXd>& Traj, int numdef)
        : ODEPhase(ode, Tmode) {
      this->setTraj(Traj, numdef);
    }
    ODEPhase(const DODE& ode,
             TranscriptionModes Tmode,
             const std::vector<Eigen::VectorXd>& Traj,
             int numdef,
             bool LerpIG)
        : ODEPhase(ode, Tmode) {
      this->setTraj(Traj, numdef, LerpIG);
    }

    ODEPhase(const DODE& ode, std::string Tmode) : ODEPhase(ode, strto_TranscriptionMode(Tmode)) {
    }
    ODEPhase(const DODE& ode, std::string Tmode, const std::vector<Eigen::VectorXd>& Traj, int numdef)
        : ODEPhase(ode, strto_TranscriptionMode(Tmode), Traj, numdef) {
    }
    ODEPhase(
        const DODE& ode, std::string Tmode, const std::vector<Eigen::VectorXd>& Traj, int numdef, bool LerpIG)
        : ODEPhase(ode, strto_TranscriptionMode(Tmode), Traj, numdef, LerpIG) {
    }

    void setUnits(const Eigen::VectorXd& XtUPUnits_) {
        this->XtUPUnits = XtUPUnits_;
        VectorXd output_scales = XtUPUnits.head(this->XVars()).cwiseInverse()*this->XtUPUnits[this->XVars()];
        VectorFunctionalX odetemp;

        if constexpr (DODE::IsGenericODE) {
            odetemp = this->ode.func;
        }
        else {
            odetemp = this->ode;
        }

        auto tmp = IOScaled<VectorFunctionalX>(odetemp, this->XtUPUnits, output_scales);

        this->ode_scaled = ScaledODE(tmp, this->XVars(), this->UVars(), this->PVars());


    }

    virtual Integrator<ScaledODE> make_scaled_reintegrator() const {
        Integrator<ScaledODE> Integ;
        if (this->UVars() == 0 || this->ControlMode == BlockConstant) {
            Integ = Integrator<ScaledODE>{ this->ode_scaled, this->integrator.DefStepSize };
        }
        else {
            Eigen::VectorXi ulocs;
            ulocs.setLinSpaced(this->UVars(), this->TVar() + 1, this->TVar() + this->UVars());

            Eigen::VectorXi varlocs(1);
            varlocs[0] = this->TVar();

            double tscale = this->XtUPUnits[this->TVar()];
            auto tabcon = InterpFunction<-1>(std::make_shared<LGLInterpTable>(this->Table), ulocs).eval(Arguments<1>(1) * tscale);
            VectorXd Uscales = this->XtUPUnits.segment(this->TVar() + 1, this->UVars()).cwiseInverse();
            auto ucon = RowScaled<decltype(tabcon)>(tabcon, Uscales);
            Integ = Integrator<ScaledODE>{
                this->ode_scaled, this->integrator.DefStepSize, ucon, varlocs };
        }

        Integ.Adaptive = this->integrator.Adaptive;
        Integ.FastAdaptiveSTM = this->integrator.FastAdaptiveSTM;
        Integ.AbsTols = this->integrator.AbsTols;
        Integ.RelTols = this->integrator.RelTols;
        Integ.MinStepSize = this->integrator.MinStepSize;
        Integ.MaxStepSize = this->integrator.MaxStepSize;
        Integ.EnableVectorization = this->EnableVectorization;

        return Integ;
    }

    virtual Integrator<DODE> make_reintegrator() const{
        Integrator<DODE> Integ;

        if (this->UVars() == 0 || this->ControlMode == BlockConstant) {
            Integ = Integrator<DODE>{ this->ode, this->integrator.DefStepSize };
        }
        else {
            Integ = Integrator<DODE>{
                this->ode, this->integrator.DefStepSize, std::make_shared<LGLInterpTable>(this->Table) };
        }

        Integ.Adaptive = this->integrator.Adaptive;
        Integ.FastAdaptiveSTM = this->integrator.FastAdaptiveSTM;
        Integ.AbsTols = this->integrator.AbsTols;
        Integ.RelTols = this->integrator.RelTols;
        Integ.MinStepSize = this->integrator.MinStepSize;
        Integ.MaxStepSize = this->integrator.MaxStepSize;
        Integ.EnableVectorization = this->EnableVectorization;

        return Integ;
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
          this->Table = LGLInterpTable(odetemp, this->XVars(), this->UVars() + this->PVars(), m);
          this->Order = 7.0;
          this->numTranCardStates = 4;
          break;
        case TranscriptionModes::LGL5:
          this->Table = LGLInterpTable(odetemp, this->XVars(), this->UVars() + this->PVars(), m);
          this->Order = 5.0;
          this->numTranCardStates = 3;
          break;
        case TranscriptionModes::LGL3:
          this->Table = LGLInterpTable(odetemp, this->XVars(), this->UVars() + this->PVars(), m);

          this->Order = 3.0;
          this->numTranCardStates = 2;
          break;
        case TranscriptionModes::Trapezoidal:
          this->Table =
              LGLInterpTable(odetemp, this->XVars(), this->UVars() + this->PVars(), TranscriptionModes::LGL3);

          this->Order = 2.0;
          this->numTranCardStates = 2;
          break;
        case TranscriptionModes::CentralShooting:
          this->Table =
              LGLInterpTable(odetemp, this->XVars(), this->UVars() + this->PVars(), TranscriptionModes::LGL3);
          this->Order = 7.0;
          this->numTranCardStates = 2;
          // Default Central Shooting to BlockConstant!!!
          this->setControlMode(BlockConstant);

          break;
        default: {
          throw std::invalid_argument("Invalid Transcription Method");
          break;
        }
      }
    }

    virtual void transcribe_dynamics() {
      VectorXi StateT(this->ode.XtUVars());
      for (int i = 0; i < this->ode.XtUVars(); i++)
        StateT[i] = i;
      VectorXi OParT(this->ode.PVars());
      for (int i = 0; i < this->ode.PVars(); i++)
        OParT[i] = i;
      VectorXi empty(0);
      empty.resize(0);


      auto lgldef = [&](auto cs, auto ode_t) {
        if (this->ControlMode == BlockConstant) {

          if constexpr (DODE::UV == 0 && DODE::PV == 0) {
            LGLType<decltype(ode_t), cs.value> lgl(ode_t);
            lgl.EnableVectorization = this->EnableVectorization;
            this->DynamicsFuncIndex = this->indexer.addEquality(
                lgl, PhaseRegionFlags::DefectPath, StateT, OParT, empty, ThreadingFlags::ByApplication);
          } else {

            using BlockedODE =  Blocked_ODE_Wrapper<decltype(ode_t)>;

            auto lgl = LGLDefects<BlockedODE, cs.value>(BlockedODE(ode_t));
            lgl.EnableVectorization = this->EnableVectorization;
            this->DynamicsFuncIndex = this->indexer.addEquality(
                lgl, PhaseRegionFlags::BlockDefectPath, StateT, OParT, empty, ThreadingFlags::ByApplication);
          }
        } else {
          LGLType<decltype(ode_t), cs.value> lgl(ode_t);
          lgl.EnableVectorization = this->EnableVectorization;
          this->DynamicsFuncIndex = this->indexer.addEquality(
              lgl, PhaseRegionFlags::DefectPath, StateT, OParT, empty, ThreadingFlags::ByApplication);
        }
      };

      auto trapdef = [&](auto ode_t) {

          if (this->ControlMode == BlockConstant) {

              if constexpr (DODE::UV == 0 && DODE::PV == 0) {
                  TrapezoidalDefects<decltype(ode_t)> trap(ode_t);
                  trap.EnableVectorization = this->EnableVectorization;
                  this->DynamicsFuncIndex = this->indexer.addEquality(
                      trap, PhaseRegionFlags::DefectPath, StateT, OParT, empty, ThreadingFlags::ByApplication);
              }
              else {
                  using BlockedODE =  Blocked_ODE_Wrapper<decltype(ode_t)>;

                  auto trap = TrapezoidalDefects<BlockedODE>(BlockedODE(ode_t));
                  trap.EnableVectorization = this->EnableVectorization;
                  //trap.EnableHessianSparsity = this->EnableHessianSparsity;
                  this->DynamicsFuncIndex = this->indexer.addEquality(trap,
                      PhaseRegionFlags::BlockDefectPath,
                      StateT,
                      OParT,
                      empty,
                      ThreadingFlags::ByApplication);
              }


          }
          else {
              TrapezoidalDefects<decltype(ode_t)> trap(ode_t);
              trap.EnableVectorization = this->EnableVectorization;
              this->DynamicsFuncIndex = this->indexer.addEquality(
                  trap, PhaseRegionFlags::DefectPath, StateT, OParT, empty, ThreadingFlags::ByApplication);
          }

      };


      switch (this->TranscriptionMode) {
        case TranscriptionModes::LGL7: {

          if (this->AutoScaling) {
              lgldef(int_const<4>(),this->ode_scaled);
          }
          else {
              lgldef(int_const<4>(),this->ode);
          }

          break;
        }
        case TranscriptionModes::LGL5: {
            if (this->AutoScaling) {
                lgldef(int_const<3>(), this->ode_scaled);
            }
            else {
                lgldef(int_const<3>(), this->ode);
            }
          break;
        }
        case TranscriptionModes::LGL3: {
            if (this->AutoScaling) {
                lgldef(int_const<2>(), this->ode_scaled);
            }
            else {
                lgldef(int_const<2>(), this->ode);
            }
          break;
        }
        case TranscriptionModes::Trapezoidal: {
            if (this->AutoScaling) {
                trapdef(this->ode_scaled);
            }
            else {
                trapdef(this->ode);
            }
          break;
        }
        case TranscriptionModes::CentralShooting: {

          auto shooter = this->make_shooter();
          this->DynamicsFuncIndex = this->indexer.addEquality(
              shooter, PhaseRegionFlags::DefectPath, StateT, OParT, empty, ThreadingFlags::ByApplication);


          break;
        }
        default:
          throw std::invalid_argument("Invalid Transcription Method");
      }
    }

    virtual ASSET::ConstraintInterface make_shooter() {
      auto Integ = Integrator<DODE> {this->ode, this->integrator.DefStepSize};
      Integ.Adaptive = this->integrator.Adaptive;
      Integ.FastAdaptiveSTM = this->integrator.FastAdaptiveSTM;
      Integ.AbsTols = this->integrator.AbsTols;
      Integ.MinStepSize = this->integrator.MinStepSize;
      Integ.MaxStepSize = this->integrator.MaxStepSize;
      Integ.EnableVectorization = this->EnableVectorization;
      Integ.VectorizeBatchCalls = this->integrator.VectorizeBatchCalls;

      if (OldShootingDefect) {
        auto shooter = ShootingDefect {this->ode, Integ};
        shooter.EnableHessianSparsity = this->EnableHessianSparsity;
        return ASSET::ConstraintInterface(shooter);
      } else {
        auto shooter = CentralShootingDefect {this->ode, Integ};
        shooter.EnableHessianSparsity = this->EnableHessianSparsity;
        shooter.EnableVectorization = this->EnableVectorization;
        return ASSET::ConstraintInterface(shooter);
      }
    }


    virtual Eigen::VectorXd calc_global_error() const {
      
        auto CalcError = [&](auto& Integ, const auto& Traj) {

            ODEState<double> Xin;
            ODEState<double> Xout;

            double T0 = Traj[0][this->TVar()];
            double TF = Traj.back()[this->TVar()];

            int BlockSize = this->numTranCardStates;
            int numBlocks = (Traj.size() - 1) / (BlockSize - 1);


            Xin =Traj[0];

            for (int i = 0; i < numBlocks; i++) {
                int start = (BlockSize - 1) * i;
                int stop = (BlockSize - 1) * (i + 1);

                double tf = Traj[stop][this->TVar()];
                Xout = Integ.integrate(Xin, tf);
                Xin = Xout;
            }

            Eigen::VectorXd gerror = (Traj.back() - Xout).head(this->XVars()).cwiseAbs();
            return gerror;
        };


      if (this->AutoScaling) {
          Integrator<ScaledODE> Integ = this->make_scaled_reintegrator();
          auto ActiveTrajTmp = this->ActiveTraj;
          for (auto& T : ActiveTrajTmp) {
              T = T.cwiseQuotient(this->XtUPUnits);
          }
          return CalcError(Integ, ActiveTrajTmp);

      }
      else {
          Integrator<DODE> Integ = this->make_reintegrator();
          return CalcError(Integ, this->ActiveTraj);
      }


    }

    virtual void get_meshinfo_deboor(Eigen::VectorXd& tsnd,
                                     Eigen::MatrixXd& mesh_errors,
                                     Eigen::MatrixXd& mesh_dist) const {

      double T0 = this->ActiveTraj[0][this->TVar()];
      double TF = this->ActiveTraj.back()[this->TVar()];

      int BlockSize = this->numTranCardStates;
      int numBlocks = (this->ActiveTraj.size() - 1) / (BlockSize - 1);
      

      mesh_errors.resize(this->XVars(), numBlocks + 1);
      mesh_dist.resize(this->XVars(), numBlocks + 1);
      tsnd.resize(numBlocks + 1);


      Eigen::VectorXd XerrWeights(this->numTranCardStates);
      Eigen::VectorXd DXerrWeights(this->numTranCardStates);
      std::vector<ODEDeriv<double>> Derivs(ActiveTraj.size(), ODEDeriv<double>::Zero(this->XVars()));


      for (int i = 0; i < this->ActiveTraj.size(); i++) {
        this->ode.compute(ActiveTraj[i], Derivs[i]);
      }

      ////////////////////////////////////////////////////
      auto factorial = [](int n) {
        double fact = 1;
        for (int i = 1; i <= n; i++)
          fact = fact * i;
        return fact;
      };

      double PolyFact = factorial(this->Order);
      double ErrorWeight;

      auto FillErrorInfo = [&](auto CardStates) {
        for (int i = 0; i < CardStates.value; i++) {
          XerrWeights[i] = LGLCoeffs<CardStates.value>::Cardinal_XPower_Weights[i][0] * PolyFact;
          DXerrWeights[i] = LGLCoeffs<CardStates.value>::Cardinal_DXPower_Weights[i][0] * PolyFact;
        }
        ErrorWeight = LGLCoeffs<CardStates.value>::ErrorWeight;
      };


      switch (this->TranscriptionMode) {
        case TranscriptionModes::LGL7:
          FillErrorInfo(int_const<4>());
          break;
        case TranscriptionModes::LGL5:
          FillErrorInfo(int_const<3>());
          break;
        case TranscriptionModes::LGL3:
          FillErrorInfo(int_const<2>());
          break;
        case TranscriptionModes::CentralShooting:
          FillErrorInfo(int_const<2>());
          break;
        case TranscriptionModes::Trapezoidal:
          ErrorWeight = 1 / 12.0;
          XerrWeights.setZero();
          DXerrWeights[0] = -1;
          DXerrWeights[1] = 1;
          break;
        default: {
          throw std::invalid_argument("Invalid Transcription Method");
          break;
        }
      }
      ////////////////////////////////////////////////////
      std::vector<ODEDeriv<double>> yvecs(numBlocks);
      Eigen::VectorXd hs(numBlocks);

      for (int i = 0; i < numBlocks; i++) {
        int start = (BlockSize - 1) * i;

        hs[i] =
            this->ActiveTraj[start + (BlockSize - 1)][this->TVar()] - this->ActiveTraj[start][this->TVar()];

        tsnd[i] = (this->ActiveTraj[start][this->TVar()] - T0) / (TF - T0);

        ODEDeriv<double> yvec(this->XVars());
        yvec.setZero();
        double powh = std::pow(hs[i], this->Order);

        ODEDeriv<double> dtemp(this->XVars());


        if (this->UVars() != 0 && this->ControlMode == BlockConstant) {
          ODEState<double> tmp = this->ActiveTraj[start + BlockSize - 1];
          tmp.segment(this->XtVars(), this->UVars()) =
              this->ActiveTraj[start].segment(this->XtVars(), this->UVars());

          dtemp = Derivs[start + BlockSize - 1];
          Derivs[start + BlockSize - 1].setZero();
          this->ode.compute(tmp, Derivs[start + BlockSize - 1]);
        }

        for (int j = 0; j < BlockSize; j++) {
          yvec += this->ActiveTraj[start + j].head(this->XVars()) * XerrWeights[j] / powh;
          yvec += Derivs[start + j].head(this->XVars()) * DXerrWeights[j] * hs[i] / powh;
        }

        if (this->UVars() != 0 && this->ControlMode == BlockConstant) {
          Derivs[start + BlockSize - 1] = dtemp;
        }

        yvecs[i] = yvec;
      }

      /////////////////////////
      if (this->AutoScaling) {
          // All errors assessed in scaled units
          // yvecs has dims of X/t^Order
          for (int i = 0; i < numBlocks;i++) {
              yvecs[i] = (yvecs[i].cwiseQuotient(this->XtUPUnits.head(this->XVars()))).eval();
              yvecs[i] *= std::pow(this->XtUPUnits[this->XVars()], this->Order);
              hs[i] /= this->XtUPUnits[this->XVars()];
          }
      }
      ////////////////////////

      tsnd[numBlocks] = 1.0;


      for (int i = 0; i < numBlocks; i++) {
        ODEDeriv<double> err_tmp(this->XVars());

        if (i > 0 && i < (numBlocks - 1)) {

        err_tmp = ((yvecs[i] - yvecs[i - 1]) / (hs[i] + hs[i - 1])).cwiseAbs()
                    + ((yvecs[i + 1] - yvecs[i]) / (hs[i] + hs[i + 1])).cwiseAbs();

        } else if (i == 0) {
        err_tmp = (2 * (yvecs[i] - yvecs[i + 1]) / (hs[i] + hs[i + 1])).cwiseAbs();
        } else {
        err_tmp = (2 * (yvecs[i] - yvecs[i - 1]) / (hs[i] + hs[i - 1])).cwiseAbs();
        }
        
        mesh_dist.col(i) = err_tmp.array().pow(1 / (this->Order + 1));
        mesh_errors.col(i) = err_tmp * std::pow(std::abs(hs[i]), this->Order + 1) * ErrorWeight;
      }

      mesh_dist.col(numBlocks) = mesh_dist.col(numBlocks - 1);
      mesh_errors.col(numBlocks) = mesh_errors.col(numBlocks - 1);
    }


    virtual void get_meshinfo_integrator(Eigen::VectorXd& tsnd,
                                         Eigen::MatrixXd& mesh_errors,
                                         Eigen::MatrixXd& mesh_dist) const {




      auto CalcError = [&](auto& Integ, const auto& Traj) {

          
          double T0 = Traj[0][this->TVar()];
          double TF = Traj.back()[this->TVar()];

          int BlockSize = this->numTranCardStates;
          int numBlocks = (Traj.size() - 1) / (BlockSize - 1);

          mesh_errors.resize(this->XVars(), numBlocks + 1);
          mesh_dist.resize(this->XVars(), numBlocks + 1);
          tsnd.resize(numBlocks + 1);

          Eigen::MatrixXd tmp_mat(this->XVars(), Traj.size() - 1);


          std::vector<ODEState<double>> Xins(Traj.size() - 1);
          Eigen::VectorXd tfs(Traj.size() - 1);

          for (int i = 0; i < Traj.size() - 1; i++) {
              Xins[i] = Traj[i];
              tfs[i] = Traj[i + 1][this->TVar()];
          }
          auto Xouts = Integ.integrate(Xins, tfs);

          for (int i = 0; i < Traj.size() - 1; i++) {
              tmp_mat.col(i) =
                  (Xouts[i].head(this->XVars()) - Traj[i + 1].head(this->XVars())).cwiseAbs();
          }


          double max_err = tmp_mat.maxCoeff();
          ODEDeriv<double> evec(this->XVars());

          for (int i = 0; i < numBlocks; i++) {
              int start = (BlockSize - 1) * i;
              int stop = (BlockSize - 1) * (i + 1);

              double t0 = Traj[start][this->TVar()];
              double tf = Traj[stop][this->TVar()];

              tsnd[i] = (t0 - T0) / (TF - T0);

              evec.setZero();

              for (int j = 0; j < BlockSize - 1; j++) {
                  evec += tmp_mat.col(start + j) / (BlockSize - 1);
              }

              evec.setZero();

              for (int j = 0; j < BlockSize - 1; j++) {
                  double ti = Traj[start + j][this->TVar()];
                  double tn = Traj[start + j + 1][this->TVar()];

                  evec += tmp_mat.col(start + j) * std::abs((tn - ti) / (tf - t0));

              }

              double h = std::abs(tf - t0);
              mesh_errors.col(i) = evec;
              mesh_dist.col(i) = mesh_errors.col(i) / (std::pow(h, this->Order + 1) * max_err);
              mesh_dist.col(i) = (mesh_dist.col(i).array().pow(1 / (this->Order + 1))).eval();
          }

          mesh_errors.col(numBlocks) = mesh_errors.col(numBlocks - 1);
          mesh_dist.col(numBlocks) = mesh_dist.col(numBlocks - 1);
          tsnd[numBlocks] = 1.0;


      };
     
      if (this->AutoScaling) {
          Integrator<ScaledODE> Integ = this->make_scaled_reintegrator();
          auto ActiveTrajTmp = this->ActiveTraj;
          for (auto& T : ActiveTrajTmp) {
              T = T.cwiseQuotient(this->XtUPUnits);
          }
          return CalcError(Integ, ActiveTrajTmp);

      }
      else {
          Integrator<DODE> Integ = this->make_reintegrator();
          return CalcError(Integ, this->ActiveTraj);
      }
      
    }



    template<class PyClass>
    static void BuildImpl(PyClass& phase) {
      phase.def(py::init<DODE, TranscriptionModes>());
      phase.def(py::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd>&, int>());
      phase.def(py::init<DODE, TranscriptionModes, const std::vector<Eigen::VectorXd>&, int, bool>());

      phase.def(py::init<DODE, std::string>());
      phase.def(py::init<DODE, std::string, const std::vector<Eigen::VectorXd>&, int>());
      phase.def(py::init<DODE, std::string, const std::vector<Eigen::VectorXd>&, int, bool>());
    }

    static void Build(py::module& m) {
      auto phase = py::class_<ODEPhase<DODE>, std::shared_ptr<ODEPhase<DODE>>, ODEPhaseBase>(m, "phase");

      BuildImpl(phase);
      phase.def_readwrite("integrator", &ODEPhase<DODE>::integrator);
      phase.def_readwrite("EnableHessianSparsity", &ODEPhase<DODE>::EnableHessianSparsity);
      phase.def_readwrite("OldShootingDefect", &ODEPhase<DODE>::OldShootingDefect);

      phase.def_readwrite("AutoScaling", &ODEPhase<DODE>::AutoScaling);
      phase.def("setUnits", &ODEPhase<DODE>::setUnits);
      phase.def("get_input_scale", &ODEPhase<DODE>::get_input_scale);



    }
  };

}  // namespace ASSET
