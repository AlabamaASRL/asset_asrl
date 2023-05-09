#pragma once

#include "LGLInterpTable.h"
#include "MeshIterateInfo.h"
#include "ODESizes.h"
#include "OptimalControlFlags.h"
#include "PhaseIndexer.h"
#include "Solvers/NonLinearProgram.h"
#include "Solvers/OptimizationProblemBase.h"
#include "Solvers/PSIOPT.h"
#include "StateFunction.h"
#include "VectorFunctions/ASSET_VectorFunctions.h"
#include "pch.h"

namespace ASSET {

  struct OptimalControlProblem;

  struct ODEPhaseBase : ODESize<-1, -1, -1>, OptimizationProblemBase {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    using VectorFunctionalX = GenericFunction<-1, -1>;
    using ScalarFunctionalX = GenericFunction<-1, 1>;

    using StateConstraint = StateFunction<VectorFunctionalX>;
    using StateObjective = StateFunction<ScalarFunctionalX>;

    template<int V>
    using int_const = std::integral_constant<int, V>;

    friend OptimalControlProblem;

    int calc_threads();

   protected:
    PhaseIndexer indexer;
    bool doTranscription = true;
    bool EnableVectorization = true;

    int numDefects = 0;
    int numStatParams = 0;
    VectorXd DefBinSpacing;
    VectorXi DefsPerBin;

    TranscriptionModes TranscriptionMode = TranscriptionModes::LGL3;
    ControlModes ControlMode = ControlModes::FirstOrderSpline;
    IntegralModes IntegralMode = IntegralModes::BaseIntegral;
    int numTranCardStates = 2;
    double Order = 3;

    LGLInterpTable Table;

    bool TrajectoryLoaded = false;
    std::vector<VectorXd> ActiveTraj;
    VectorXd ActiveODEParams;
    VectorXd ActiveStaticParams;

    bool MultipliersLoaded = false;
    bool PostOptInfoValid = false;

    VectorXd ActiveEqLmults;
    VectorXd ActiveIqLmults;
    VectorXd ActiveEqCons;
    VectorXd ActiveIqCons;

    std::map<int, StateConstraint> userEqualities;
    std::map<int, StateConstraint> userInequalities;
    std::map<int, StateObjective> userStateObjectives;
    std::map<int, StateObjective> userIntegrands;
    std::map<int, StateObjective> userParamIntegrands;

    int DynamicsFuncIndex = 0;
    int ControlFuncsIndex = -1;
    VectorXi NodeSpacingFuncIndices;
    int TranSpacingFuncIndices = 0;
    int ConstraintOrder = 0;


    ///////////////////////
   public:
    bool AdaptiveMesh = false;
    bool PrintMeshInfo = true;
    int MaxMeshIters = 10;
    int MaxSegments = 10000;
    int MinSegments = 4;


    int NumExtraSegs = 4;
    double MeshRedFactor = .5;
    double MeshIncFactor = 5.0;
    double MeshErrFactor = 10.0;
    bool ForceOneMeshIter = false;
    bool SolveOnlyFirst = true;
    bool NewError = false;
    bool DetectControlSwitches = false;

    double RelSwitchTol = .3;
    double AbsSwitchTol = 1.0e-6;

    double MeshTol = 1.0e-6;

    std::string MeshErrorEstimator = "deboor";
    std::string MeshErrorCriteria = "max";     //"max,avg,geometric,endtoend"
    std::string MeshErrorDistributor = "avg";  // "max,avg,geometric,endtoend"
    PSIOPT::ConvergenceFlags MeshAbortFlag = PSIOPT::ConvergenceFlags::DIVERGING;

    bool MeshConverged = false;

    std::vector<MeshIterateInfo> MeshIters;


    void setAdaptiveMesh(bool amesh) {
      this->AdaptiveMesh = amesh;
    }
    void setMeshTol(double t) {
      this->MeshTol = abs(t);
    }
    void setMeshRedFactor(double t) {
      this->MeshRedFactor = abs(t);
    }
    void setMeshIncFactor(double t) {
      this->MeshIncFactor = abs(t);
    }
    void setMeshErrFactor(double t) {
      this->MeshErrFactor = abs(t);
    }
    void setMaxMeshIters(int it) {
      this->MaxMeshIters = abs(it);
    }
    void setMinSegments(int it) {
      this->MinSegments = abs(it);
    }
    void setMaxSegments(int it) {
      this->MaxSegments = abs(it);
    }
    void setMeshErrorCriteria(std::string m) {
      this->MeshErrorCriteria = m;
    }
    void setMeshErrorEstimator(std::string m) {
      this->MeshErrorEstimator = m;
    }


    std::vector<MeshIterateInfo> getMeshIters() const {
      return this->MeshIters;
    }


    ///////////////////

   public:
    /////////////////////////////////////////////////////////////////////////////

    void enable_vectorization(bool b) {
      this->EnableVectorization = b;
    }
    void resetTranscription() {
      this->doTranscription = true;
    };

    void invalidatePostOptInfo() {
      this->PostOptInfoValid = false;
    };

    ODEPhaseBase() {
    }
    ODEPhaseBase(int Xv, int Uv, int Pv) {
      this->setXVars(Xv);
      this->setUVars(Uv);
      this->setPVars(Pv);
    }
    virtual ~ODEPhaseBase() = default;

    //////////////////////////////////////////////////
    virtual void setControlMode(ControlModes m) {
      this->resetTranscription();
      this->invalidatePostOptInfo();
      this->ControlMode = m;
      if (this->ControlMode == BlockConstant) {
        this->Table.BlockedControls = true;
      } else {
        this->Table.BlockedControls = false;
      }
    }

    void setControlMode(std::string m) {
      this->setControlMode(strto_ControlMode(m));
    }

    virtual void setIntegralMode(IntegralModes m) {
      this->resetTranscription();
      this->IntegralMode = m;
    }
    virtual void setTranscriptionMode(TranscriptionModes m) = 0;

    void switchTranscriptionMode(TranscriptionModes m, VectorXd DBS, VectorXi DPB) {
      this->setTranscriptionMode(m);
      this->setTraj(this->ActiveTraj, DBS, DPB);
    }
    void switchTranscriptionMode(TranscriptionModes m) {
      this->switchTranscriptionMode(m, this->DefBinSpacing, this->DefsPerBin);
    }


    void switchTranscriptionMode(std::string m, VectorXd DBS, VectorXi DPB) {
      this->switchTranscriptionMode(strto_TranscriptionMode(m), DBS, DPB);
    }
    void switchTranscriptionMode(std::string m) {
      this->switchTranscriptionMode(strto_TranscriptionMode(m));
    }


    /////////////////////////////////////////////////
    template<class FuncMap>
    void removeFuncImpl(FuncMap& map, int index, const std::string& funcstr) {
      this->resetTranscription();
      this->invalidatePostOptInfo();
      if (index == -1 && map.size() > 0) {
        index = map.rbegin()->first;
      }

      if (map.count(index) == 0) {
        throw std::invalid_argument(fmt::format("No {0:} with index {1:} exists in phase.", funcstr, index));
      }
      map.erase(index);
    }

    template<class FuncType, class FuncMap>
    int addFuncImpl(FuncType func, FuncMap& map, const std::string& funcstr) {
      this->resetTranscription();
      this->invalidatePostOptInfo();
      int index = map.size() == 0 ? 0 : map.rbegin()->first + 1;
      map[index] = func;
      map[index].StorageIndex = index;
      check_function_size(map.at(index), funcstr);
      return index;
    }


    /////////////////////////////////////////////////
    int addEqualCon(StateConstraint con) {
      return addFuncImpl(con, this->userEqualities, "Equality Constraint");
    }
    int addEqualCon(PhaseRegionFlags reg, VectorFunctionalX fun, VectorXi vars) {
      return this->addEqualCon(StateConstraint(fun, reg, vars));
    }
    int addEqualCon(PhaseRegionFlags reg, VectorFunctionalX fun, VectorXi xv, VectorXi opv, VectorXi spv) {
      return this->addEqualCon(StateConstraint(fun, reg, xv, opv, spv));
    }


    int addEqualCon(std::string reg, VectorFunctionalX fun, VectorXi vars) {
      return this->addEqualCon(StateConstraint(fun, strto_PhaseRegionFlag(reg), vars));
    }
    int addEqualCon(std::string reg, VectorFunctionalX fun, VectorXi xv, VectorXi opv, VectorXi spv) {
      return this->addEqualCon(StateConstraint(fun, strto_PhaseRegionFlag(reg), xv, opv, spv));
    }


    int addDeltaVarEqualCon(PhaseRegionFlags reg, int var, double value, double scale);
    int addDeltaVarEqualCon(int var, double value) {
      return this->addDeltaVarEqualCon(PhaseRegionFlags::FrontandBack, var, value, 1.0);
    }
    int addDeltaVarEqualCon(int var, double value, double scale) {
      return this->addDeltaVarEqualCon(PhaseRegionFlags::FrontandBack, var, value, scale);
    }

    int addDeltaTimeEqualCon(double value, double scale) {
      return this->addDeltaVarEqualCon(this->TVar(), value, scale);
    }
    int addDeltaTimeEqualCon(double value) {
      return this->addDeltaVarEqualCon(this->TVar(), value);
    }

    VectorXi addBoundaryValues(PhaseRegionFlags reg, VectorXi args, const VectorXd& value);
    int addValueLock(PhaseRegionFlags reg, VectorXi args);

    int addValueLock(std::string reg, VectorXi args) {
      return this->addValueLock(strto_PhaseRegionFlag(reg), args);
    }


    int addBoundaryValue(PhaseRegionFlags reg, VectorXi args, const VectorXd& value);
    int addBoundaryValue(std::string reg, VectorXi args, const VectorXd& value) {
      return this->addBoundaryValue(strto_PhaseRegionFlag(reg), args, value);
    }

    int addPeriodicityCon(VectorXi args);
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int addInequalCon(StateConstraint con) {
      return addFuncImpl(con, this->userInequalities, "Inequality Constraint");
    }
    int addInequalCon(PhaseRegionFlags reg, VectorFunctionalX fun, VectorXi vars) {
      return this->addInequalCon(StateConstraint(fun, reg, vars));
    }
    int addInequalCon(PhaseRegionFlags reg, VectorFunctionalX fun, VectorXi xv, VectorXi opv, VectorXi spv) {
      return this->addInequalCon(StateConstraint(fun, reg, xv, opv, spv));
    }

    int addInequalCon(std::string reg, VectorFunctionalX fun, VectorXi vars) {
      return this->addInequalCon(StateConstraint(fun, strto_PhaseRegionFlag(reg), vars));
    }
    int addInequalCon(std::string reg, VectorFunctionalX fun, VectorXi xv, VectorXi opv, VectorXi spv) {
      return this->addInequalCon(StateConstraint(fun, strto_PhaseRegionFlag(reg), xv, opv, spv));
    }

    ////////////////////////////////////////////////////
    int addLUVarBound(
        PhaseRegionFlags reg, int var, double lowerbound, double upperbound, double lbscale, double ubscale);


    int addLUVarBound(PhaseRegionFlags reg, int var, double lowerbound, double upperbound, double scale) {
      return this->addLUVarBound(reg, var, lowerbound, upperbound, scale, scale);
    }


    Eigen::VectorXi addLUVarBounds(
        PhaseRegionFlags reg, Eigen::VectorXi vars, double lowerbound, double upperbound, double scale) {
      Eigen::VectorXi cnums(vars.size());
      for (int i = 0; i < cnums.size(); i++) {
        cnums[i] = this->addLUVarBound(reg, vars[i], lowerbound, upperbound, scale, scale);
      }

      return cnums;
    }

    int addLUVarBound(PhaseRegionFlags reg, int var, double lowerbound, double upperbound) {
      return this->addLUVarBound(reg, var, lowerbound, upperbound, 1.0);
    }

    int addLUVarBound(
        std::string reg, int var, double lowerbound, double upperbound, double lbscale, double ubscale) {
      return this->addLUVarBound(strto_PhaseRegionFlag(reg), var, lowerbound, upperbound, lbscale, ubscale);
    }

    int addLUVarBound(std::string reg, int var, double lowerbound, double upperbound, double scale) {
      return this->addLUVarBound(reg, var, lowerbound, upperbound, scale, scale);
    }

    Eigen::VectorXi addLUVarBounds(
        std::string reg, Eigen::VectorXi vars, double lowerbound, double upperbound, double scale) {
      return addLUVarBounds(strto_PhaseRegionFlag(reg), vars, lowerbound, upperbound, scale);
    }

    int addLUVarBound(std::string reg, int var, double lowerbound, double upperbound) {
      return this->addLUVarBound(reg, var, lowerbound, upperbound, 1.0);
    }
    ////////////////////////////////////////////////////


    int addLowerVarBound(PhaseRegionFlags reg, int var, double lowerbound, double lbscale);
    int addLowerVarBound(std::string reg, int var, double lowerbound, double lbscale) {
      return addLowerVarBound(strto_PhaseRegionFlag(reg), var, lowerbound, lbscale);
    }

    int addLowerVarBound(PhaseRegionFlags reg, int var, double lowerbound) {
      return this->addLowerVarBound(reg, var, lowerbound, 1.0);
    }
    int addLowerVarBound(std::string reg, int var, double lowerbound) {
      return this->addLowerVarBound(reg, var, lowerbound, 1.0);
    }

    int addUpperVarBound(PhaseRegionFlags reg, int var, double upperbound, double ubscale);
    int addUpperVarBound(std::string reg, int var, double upperbound, double ubscale) {
      return addUpperVarBound(strto_PhaseRegionFlag(reg), var, upperbound, ubscale);
    }

    int addUpperVarBound(PhaseRegionFlags reg, int var, double upperbound) {
      return this->addUpperVarBound(reg, var, upperbound, 1.0);
    }
    int addUpperVarBound(std::string reg, int var, double upperbound) {
      return this->addUpperVarBound(reg, var, upperbound, 1.0);
    }

    ///////////////////////////////////////////////////
    int addLUNormBound(PhaseRegionFlags reg,
                       VectorXi vars,
                       double lowerbound,
                       double upperbound,
                       double lbscale,
                       double ubscale);
    int addLUNormBound(std::string reg,
                       VectorXi vars,
                       double lowerbound,
                       double upperbound,
                       double lbscale,
                       double ubscale) {
      return addLUNormBound(strto_PhaseRegionFlag(reg), vars, lowerbound, upperbound, lbscale, ubscale);
    }

    int addLUNormBound(
        PhaseRegionFlags reg, VectorXi vars, double lowerbound, double upperbound, double scale) {
      return this->addLUNormBound(reg, vars, lowerbound, upperbound, scale, scale);
    }
    int addLUNormBound(PhaseRegionFlags reg, VectorXi vars, double lowerbound, double upperbound) {
      return this->addLUNormBound(reg, vars, lowerbound, upperbound, 1.0);
    }
    int addLUNormBound(std::string reg, VectorXi vars, double lowerbound, double upperbound, double scale) {
      return this->addLUNormBound(reg, vars, lowerbound, upperbound, scale, scale);
    }
    int addLUNormBound(std::string reg, VectorXi vars, double lowerbound, double upperbound) {
      return this->addLUNormBound(reg, vars, lowerbound, upperbound, 1.0);
    }


    int addLowerNormBound(PhaseRegionFlags reg, VectorXi vars, double lowerbound, double lbscale);
    int addLowerNormBound(std::string reg, VectorXi vars, double lowerbound, double lbscale) {
      return addLowerNormBound(strto_PhaseRegionFlag(reg), vars, lowerbound, lbscale);
    }

    int addLowerNormBound(PhaseRegionFlags reg, VectorXi vars, double lowerbound) {
      return this->addLowerNormBound(reg, vars, lowerbound, 1.0);
    }
    int addLowerNormBound(std::string reg, VectorXi vars, double lowerbound) {
      return this->addLowerNormBound(reg, vars, lowerbound, 1.0);
    }

    int addUpperNormBound(PhaseRegionFlags reg, VectorXi vars, double upperbound, double ubscale);
    int addUpperNormBound(std::string reg, VectorXi vars, double upperbound, double ubscale) {
      return addUpperNormBound(strto_PhaseRegionFlag(reg), vars, upperbound, ubscale);
    }
    int addUpperNormBound(PhaseRegionFlags reg, VectorXi vars, double upperbound) {
      return this->addUpperNormBound(reg, vars, upperbound, 1.0);
    }
    int addUpperNormBound(std::string reg, VectorXi vars, double upperbound) {
      return this->addUpperNormBound(reg, vars, upperbound, 1.0);
    }

    ///////////////////////////////////////////////////
    int addLUSquaredNormBound(PhaseRegionFlags reg,
                              VectorXi vars,
                              double lowerbound,
                              double upperbound,
                              double lbscale,
                              double ubscale);

    int addLUSquaredNormBound(std::string reg,
                              VectorXi vars,
                              double lowerbound,
                              double upperbound,
                              double lbscale,
                              double ubscale) {
      return addLUSquaredNormBound(
          strto_PhaseRegionFlag(reg), vars, lowerbound, upperbound, lbscale, ubscale);
    }

    int addLUSquaredNormBound(
        PhaseRegionFlags reg, VectorXi vars, double lowerbound, double upperbound, double scale) {
      return this->addLUSquaredNormBound(reg, vars, lowerbound, upperbound, scale, scale);
    }
    int addLUSquaredNormBound(PhaseRegionFlags reg, VectorXi vars, double lowerbound, double upperbound) {
      return this->addLUSquaredNormBound(reg, vars, lowerbound, upperbound, 1.0);
    }
    int addLUSquaredNormBound(
        std::string reg, VectorXi vars, double lowerbound, double upperbound, double scale) {
      return this->addLUSquaredNormBound(reg, vars, lowerbound, upperbound, scale, scale);
    }
    int addLUSquaredNormBound(std::string reg, VectorXi vars, double lowerbound, double upperbound) {
      return this->addLUSquaredNormBound(reg, vars, lowerbound, upperbound, 1.0);
    }

    int addLowerSquaredNormBound(PhaseRegionFlags reg, VectorXi vars, double lowerbound, double lbscale);
    int addLowerSquaredNormBound(std::string reg, VectorXi vars, double lowerbound, double lbscale) {
      return addLowerSquaredNormBound(strto_PhaseRegionFlag(reg), vars, lowerbound, lbscale);
    }
    int addLowerSquaredNormBound(PhaseRegionFlags reg, VectorXi vars, double lowerbound) {
      return this->addLowerSquaredNormBound(reg, vars, lowerbound, 1.0);
    }
    int addLowerSquaredNormBound(std::string reg, VectorXi vars, double lowerbound) {
      return this->addLowerSquaredNormBound(reg, vars, lowerbound, 1.0);
    }

    int addUpperSquaredNormBound(PhaseRegionFlags reg, VectorXi vars, double upperbound, double ubscale);
    int addUpperSquaredNormBound(std::string reg, VectorXi vars, double upperbound, double ubscale) {
      return addUpperSquaredNormBound(strto_PhaseRegionFlag(reg), vars, upperbound, ubscale);
    }
    int addUpperSquaredNormBound(PhaseRegionFlags reg, VectorXi vars, double upperbound) {
      return this->addUpperSquaredNormBound(reg, vars, upperbound, 1.0);
    }
    int addUpperSquaredNormBound(std::string reg, VectorXi vars, double upperbound) {
      return this->addUpperSquaredNormBound(reg, vars, upperbound, 1.0);
    }


    ///////////////////////////////////////////////////
    int addLowerFuncBound(
        PhaseRegionFlags reg, ScalarFunctionalX func, VectorXi vars, double lowerbound, double lbscale);
    int addLowerFuncBound(
        std::string reg, ScalarFunctionalX func, VectorXi vars, double lowerbound, double lbscale) {
      return addLowerFuncBound(strto_PhaseRegionFlag(reg), func, vars, lowerbound, lbscale);
    }

    int addUpperFuncBound(
        PhaseRegionFlags reg, ScalarFunctionalX func, VectorXi vars, double upperbound, double ubscale);
    int addUpperFuncBound(
        std::string reg, ScalarFunctionalX func, VectorXi vars, double upperbound, double ubscale) {
      return addUpperFuncBound(strto_PhaseRegionFlag(reg), func, vars, upperbound, ubscale);
    }


    int addLowerFuncBound(PhaseRegionFlags reg, ScalarFunctionalX func, VectorXi vars, double lowerbound) {
      return this->addLowerFuncBound(reg, func, vars, lowerbound, 1.0);
    }
    int addUpperFuncBound(PhaseRegionFlags reg, ScalarFunctionalX func, VectorXi vars, double upperbound) {
      return this->addUpperFuncBound(reg, func, vars, upperbound, 1.0);
    }

    int addLowerFuncBound(std::string reg, ScalarFunctionalX func, VectorXi vars, double lowerbound) {
      return this->addLowerFuncBound(reg, func, vars, lowerbound, 1.0);
    }
    int addUpperFuncBound(std::string reg, ScalarFunctionalX func, VectorXi vars, double upperbound) {
      return this->addUpperFuncBound(reg, func, vars, upperbound, 1.0);
    }


    int addLUFuncBound(PhaseRegionFlags reg,
                       ScalarFunctionalX func,
                       VectorXi vars,
                       double lowerbound,
                       double upperbound,
                       double lbscale,
                       double ubscale);

    int addLUFuncBound(std::string reg,
                       ScalarFunctionalX func,
                       VectorXi vars,
                       double lowerbound,
                       double upperbound,
                       double lbscale,
                       double ubscale) {
      return addLUFuncBound(strto_PhaseRegionFlag(reg), func, vars, lowerbound, upperbound, lbscale, ubscale);
    }

    int addLUFuncBound(PhaseRegionFlags reg,
                       ScalarFunctionalX func,
                       VectorXi vars,
                       double lowerbound,
                       double upperbound,
                       double scale) {
      return this->addLUFuncBound(reg, func, vars, lowerbound, upperbound, scale, scale);
    }
    int addLUFuncBound(
        PhaseRegionFlags reg, ScalarFunctionalX func, VectorXi vars, double lowerbound, double upperbound) {
      return this->addLUFuncBound(reg, func, vars, lowerbound, upperbound, 1.0, 1.0);
    }

    int addLUFuncBound(std::string reg,
                       ScalarFunctionalX func,
                       VectorXi vars,
                       double lowerbound,
                       double upperbound,
                       double scale) {
      return this->addLUFuncBound(reg, func, vars, lowerbound, upperbound, scale, scale);
    }
    int addLUFuncBound(
        std::string reg, ScalarFunctionalX func, VectorXi vars, double lowerbound, double upperbound) {
      return this->addLUFuncBound(reg, func, vars, lowerbound, upperbound, 1.0, 1.0);
    }

    ///////////////////////////////////////////////////////////////
    int addLowerDeltaVarBound(PhaseRegionFlags reg, int var, double lowerbound, double lbscale);
    int addLowerDeltaVarBound(int var, double lowerbound, double lbscale) {
      return this->addLowerDeltaVarBound(PhaseRegionFlags::FrontandBack, var, lowerbound, lbscale);
    }
    int addLowerDeltaVarBound(int var, double lowerbound) {
      return this->addLowerDeltaVarBound(PhaseRegionFlags::FrontandBack, var, lowerbound, 1.0);
    }

    int addLowerDeltaTimeBound(double lowerbound, double lbscale) {
      return this->addLowerDeltaVarBound(this->TVar(), lowerbound, lbscale);
    }
    int addLowerDeltaTimeBound(double lowerbound) {
      return this->addLowerDeltaVarBound(this->TVar(), lowerbound, 1.0);
    }

    int addUpperDeltaVarBound(PhaseRegionFlags reg, int var, double upperbound, double ubscale);
    int addUpperDeltaVarBound(int var, double upperbound, double ubscale) {
      return this->addUpperDeltaVarBound(PhaseRegionFlags::FrontandBack, var, upperbound, ubscale);
    }
    int addUpperDeltaVarBound(int var, double upperbound) {
      return this->addUpperDeltaVarBound(PhaseRegionFlags::FrontandBack, var, upperbound, 1.0);
    }

    int addUpperDeltaTimeBound(double upperbound, double ubscale) {
      return this->addUpperDeltaVarBound(this->TVar(), upperbound, ubscale);
    }
    int addUpperDeltaTimeBound(double upperbound) {
      return this->addUpperDeltaVarBound(this->TVar(), upperbound, 1.0);
    }

    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int addStateObjective(StateObjective obj) {
      return addFuncImpl(obj, this->userStateObjectives, "State Objective");
    }
    int addStateObjective(PhaseRegionFlags reg, ScalarFunctionalX fun, VectorXi vars) {
      return this->addStateObjective(StateObjective(fun, reg, vars));
    }
    int addStateObjective(
        PhaseRegionFlags reg, ScalarFunctionalX fun, VectorXi xv, VectorXi opv, VectorXi spv) {
      return this->addStateObjective(StateObjective(fun, reg, xv, opv, spv));
    }

    int addStateObjective(std::string reg, ScalarFunctionalX fun, VectorXi vars) {
      return this->addStateObjective(StateObjective(fun, strto_PhaseRegionFlag(reg), vars));
    }
    int addStateObjective(std::string reg, ScalarFunctionalX fun, VectorXi xv, VectorXi opv, VectorXi spv) {
      return this->addStateObjective(StateObjective(fun, strto_PhaseRegionFlag(reg), xv, opv, spv));
    }


    int addValueObjective(PhaseRegionFlags reg, int var, double scale);
    int addValueObjective(std::string reg, int var, double scale) {
      return addValueObjective(strto_PhaseRegionFlag(reg), var, scale);
    }


    int addDeltaVarObjective(int var, double lbscale);
    int addDeltaTimeObjective(double scale) {
      return this->addDeltaVarObjective(this->TVar(), scale);
    }
    ///////////////////////////////////////////////////
    int addIntegralObjective(StateObjective obj) {
      return addFuncImpl(obj, this->userIntegrands, "Integral Objective");
    }
    int addIntegralObjective(ScalarFunctionalX fun, VectorXi vars) {
      return this->addIntegralObjective(StateObjective(fun, Path, vars));
    }
    int addIntegralObjective(ScalarFunctionalX fun, VectorXi XtUvars, VectorXi OPvars, VectorXi SPvars) {
      return this->addIntegralObjective(StateObjective(fun, Path, XtUvars, OPvars, SPvars));
    }
    ///////////////////////////////////////////////////
    int addIntegralParamFunction(StateObjective con, int pv) {
      VectorXi epv(1);
      epv[0] = pv;
      int index = addFuncImpl(con, this->userParamIntegrands, "Integral Parameter Function");
      this->userParamIntegrands[index].EXTVars = epv;
      return index;
    }
    int addIntegralParamFunction(ScalarFunctionalX fun, VectorXi vars, int accum_parm) {
      return this->addIntegralParamFunction(StateObjective(fun, Path, vars), accum_parm);
    }

    int addIntegralParamFunction(
        ScalarFunctionalX fun, VectorXi XtUvars, VectorXi OPvars, VectorXi SPvars, int accum_parm) {
      return this->addIntegralParamFunction(StateObjective(fun, Path, XtUvars, OPvars, SPvars), accum_parm);
    }


    /////////////////////////////////////////////////


    void removeEqualCon(int index) {
      this->removeFuncImpl(this->userEqualities, index, "Equality Constraint");
    }
    void removeInequalCon(int index) {
      this->removeFuncImpl(this->userInequalities, index, "Inequality Constraint");
    }
    void removeStateObjective(int index) {
      this->removeFuncImpl(this->userStateObjectives, index, "State Objective");
    }
    void removeIntegralObjective(int index) {
      this->removeFuncImpl(this->userIntegrands, index, "Integral Objective");
    }
    void removeIntegralParamFunction(int index) {
      this->removeFuncImpl(this->userParamIntegrands, index, "Integral Parameter Function");
    }


    /////////////////////////////////////////////////

    virtual void setTraj(const std::vector<Eigen::VectorXd>& mesh,
                         Eigen::VectorXd DBS,
                         Eigen::VectorXi DPB,
                         bool LerpTraj);

    void setTraj(const std::vector<Eigen::VectorXd>& mesh, Eigen::VectorXd DBS, Eigen::VectorXi DPB) {
      this->setTraj(mesh, DBS, DPB, false);
    }

    void setTraj(const std::vector<Eigen::VectorXd>& mesh, int ndef, bool LerpTraj) {
      VectorXd dbs(2);
      dbs[0] = 0.0;
      dbs[1] = 1.0;
      VectorXi dpb(1);
      dpb[0] = ndef;
      this->setTraj(mesh, dbs, dpb, LerpTraj);
    }
    void setTraj(const std::vector<Eigen::VectorXd>& mesh, int ndef) {
      this->setTraj(mesh, ndef, false);
    }


    void refineTrajManual(VectorXd DBS, VectorXi DPB);

    void refineTrajManual(int ndef) {
      VectorXd dbs(2);
      dbs[0] = 0.0;
      dbs[1] = 1.0;
      VectorXi dpb(1);
      dpb[0] = ndef;
      this->refineTrajManual(dbs, dpb);
    }
    std::vector<Eigen::VectorXd> refineTrajEqual(int n);

    void refineTrajAuto();


    void setStaticParams(VectorXd parm) {
      this->ActiveStaticParams = parm;
      this->numStatParams = parm.size();
      this->resetTranscription();
    }
    void subStaticParams(VectorXd parm) {
      this->ActiveStaticParams = parm;
      if (this->numStatParams == parm.size()) {
        // expected behavior
      } else {
        this->numStatParams = parm.size();
        this->resetTranscription();
      }
    }

    void subVariables(PhaseRegionFlags reg, VectorXi indices, VectorXd vals);
    void subVariable(PhaseRegionFlags reg, int var, double val) {
      VectorXi indices(1);
      indices[0] = var;
      VectorXd vals(1);
      vals[0] = val;
      this->subVariables(reg, indices, vals);
    }

    void subVariables(std::string reg, VectorXi indices, VectorXd vals) {
      this->subVariables(strto_PhaseRegionFlag(reg), indices, vals);
    }
    void subVariable(std::string reg, int var, double val) {
      this->subVariable(strto_PhaseRegionFlag(reg), var, val);
    }

    std::vector<Eigen::VectorXd> returnTraj() const {
      return this->ActiveTraj;
    }

    std::vector<Eigen::VectorXd> returnTrajRange(int num, double tl, double th) {
      this->Table.loadRegularData(this->DefsPerBin.sum(), this->ActiveTraj);
      return this->Table.InterpRange(num, tl, th);
    }
    std::vector<Eigen::VectorXd> returnTrajRangeND(int num, double tl, double th) {
      this->Table.loadRegularData(this->DefsPerBin.sum(), this->ActiveTraj);
      return this->Table.NDequidist(num, tl, th);
    }
    LGLInterpTable returnTrajTable() {
      LGLInterpTable tabt = this->Table;
      tabt.loadExactData(this->ActiveTraj);
      return tabt;
    }

    Eigen::VectorXd returnStaticParams() const {
      return this->ActiveStaticParams;
    }


    std::vector<Eigen::VectorXd> returnUSplineConLmults() const {

      if (!this->PostOptInfoValid) {
        throw std::invalid_argument("No multipliers to return, a solve or optimize call must be made "
                                    "before returning constraint multipliers ");
      }

      if (this->ControlFuncsIndex < 0) {
        return std::vector<Eigen::VectorXd>();
      } else {
        return this->indexer.getFuncEqMultipliers(this->ControlFuncsIndex, this->ActiveEqLmults);
      }
    }

    std::vector<Eigen::VectorXd> returnUSplineConVals() const {

      if (!this->PostOptInfoValid) {
        throw std::invalid_argument("No constraints to return, a solve or optimize call must be made "
                                    "before returning constraint values ");
      }
      if (this->ControlFuncsIndex < 0) {
        return std::vector<Eigen::VectorXd>();
      } else {
        return this->indexer.getFuncEqMultipliers(this->ControlFuncsIndex, this->ActiveEqCons);
      }
    }

    std::vector<Eigen::VectorXd> returnEqualConLmults(int index) const {
      if (!this->PostOptInfoValid) {
        throw std::invalid_argument("No multipliers to return, a solve or optimize call must be made "
                                    "before returning constraint multipliers ");
      }
      int Gindex = this->userEqualities.at(index).GlobalIndex;
      return this->indexer.getFuncEqMultipliers(Gindex, this->ActiveEqLmults);
    }
    std::vector<Eigen::VectorXd> returnEqualConVals(int index) const {
      if (!this->PostOptInfoValid) {
        throw std::invalid_argument("No constraints to return, a solve or optimize call must be made "
                                    "before returning constraint values ");
      }
      int Gindex = this->userEqualities.at(index).GlobalIndex;
      return this->indexer.getFuncEqMultipliers(Gindex, this->ActiveEqCons);
    }

    std::vector<Eigen::VectorXd> returnInequalConLmults(int index) const {
      if (!this->PostOptInfoValid) {
        throw std::invalid_argument("No multipliers to return, a solve or optimize call must be made "
                                    "before returning constraint multipliers ");
      }
      int Gindex = this->userInequalities.at(index).GlobalIndex;
      return this->indexer.getFuncIqMultipliers(Gindex, this->ActiveIqLmults);
    }

    std::vector<Eigen::VectorXd> returnInequalConVals(int index) const {
      if (!this->PostOptInfoValid) {
        throw std::invalid_argument("No constraints to return, a solve or optimize call must be made "
                                    "before returning constraint values ");
      }
      int Gindex = this->userInequalities.at(index).GlobalIndex;
      return this->indexer.getFuncIqMultipliers(Gindex, this->ActiveIqCons);
    }

    std::vector<Eigen::VectorXd> returnCostateTraj() const;
    std::vector<Eigen::VectorXd> returnTrajError() const;

    /////////////////////////////////////////////////
   protected:
    virtual void transcribe_dynamics() = 0;
    virtual void transcribe_axis_funcs();
    virtual void transcribe_control_funcs();
    virtual void transcribe_integrals();
    virtual void transcribe_basic_funcs();

    void initIndexing() {
      this->indexer = PhaseIndexer(this->XVars(), this->UVars(), this->PVars(), this->numStatParams);
      bool blockcon = false;
      if (this->ControlMode == ControlModes::BlockConstant)
        blockcon = true;
      this->indexer.set_dimensions(this->numTranCardStates, this->numDefects, blockcon);
    }
    void check_functions(int pnum);

    template<class T>
    static void check_function_size(const T& func, std::string ftype) {
      int irows = func.Func.IRows();
      switch (func.RegionFlag) {
        case Front:
        case Back:
        case Path:
        case Params:
        case ODEParams:
        case StaticParams:
        case NodalPath: {
          int isize = func.XtUVars.size() + func.OPVars.size() + func.SPVars.size();
          if (irows != isize) {

            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "Input size of {0:} (IRows = {1:}) does not match that implied by indexing parameters "
                       "(IRows = {2:}).\n",
                       ftype,
                       irows,
                       isize);

            throw std::invalid_argument("");
          }
          break;
        }
        case FrontandBack:
        case BackandFront:
        case PairWisePath: {
          int isize = func.XtUVars.size() * 2 + func.OPVars.size() + func.SPVars.size();
          if (irows != isize) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "Input size of {0:} (IRows = {1:}) does not match that implied by indexing parameters "
                       "(IRows = {2:}).\n",
                       ftype,
                       irows,
                       isize);
            throw std::invalid_argument("");
          }
          break;
        }
        default: {
          break;
        }
      }
    }


    static void check_lbscale(double lbscale) {
      if (lbscale <= 0.0) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Lower-bound scale ({0:.3e}) must be a strictly positive number.\n",
                   lbscale);
        throw std::invalid_argument("");
      }
    }
    static void check_ubscale(double ubscale) {
      if (ubscale <= 0.0) {
        fmt::print(fmt::fg(fmt::color::red),
                   "Transcription Error!!!\n"
                   "Upper-bound scale ({0:.3e}) must be a strictly positive number.\n",
                   ubscale);
        throw std::invalid_argument("");
      }
    }


    void transcribe_phase(int vo, int eqo, int iqo, std::shared_ptr<NonLinearProgram> np, int pnum);


    Eigen::VectorXd makeSolverInput() const {
      return this->indexer.makeSolverInput(this->ActiveTraj, this->ActiveStaticParams);
    }
    void collectSolverOutput(const VectorXd& Vars) {
      this->indexer.collectSolverOutput(Vars, this->ActiveTraj, this->ActiveStaticParams);
    }
    void collectSolverMultipliers(const VectorXd& EM, const VectorXd& IM) {
      this->MultipliersLoaded = true;
      this->ActiveEqLmults = EM;
      this->ActiveIqLmults = IM;
    }

    void collectPostOptInfo(const VectorXd& EC, const VectorXd& EM, const VectorXd& IC, const VectorXd& IM) {

    

      this->PostOptInfoValid = true;
      this->MultipliersLoaded = true;

      this->ActiveEqCons = EC;
      this->ActiveIqCons = IC;
      this->ActiveEqLmults = EM;
      this->ActiveIqLmults = IM;
    }


    PSIOPT::ConvergenceFlags psipot_call_impl(std::string mode);

    PSIOPT::ConvergenceFlags phase_call_impl(std::string mode);


   public:
    void transcribe(bool showstats, bool showfuns);

    void transcribe() {
      this->transcribe(false, false);
    }

    void test_threads(int i, int j, int n);

    void jet_initialize() {
      this->setThreads(1, 1);
      this->optimizer->PrintLevel = 10;
      this->PrintMeshInfo = false;

      this->transcribe();
    }
    void jet_release() {
      this->indexer = PhaseIndexer();
      this->optimizer->release();
      this->initThreads();
      this->optimizer->PrintLevel = 0;
      this->PrintMeshInfo = true;
      this->nlp = std::shared_ptr<NonLinearProgram>();
      this->resetTranscription();
      this->invalidatePostOptInfo();
    }


    PSIOPT::ConvergenceFlags solve() {
      return phase_call_impl("solve");
    }
    PSIOPT::ConvergenceFlags optimize() {
      return phase_call_impl("optimize");
    }
    PSIOPT::ConvergenceFlags solve_optimize() {
      return phase_call_impl("solve_optimize");
    }
    PSIOPT::ConvergenceFlags solve_optimize_solve() {
      return phase_call_impl("solve_optimize_solve");
    }
    PSIOPT::ConvergenceFlags optimize_solve() {
      return phase_call_impl("optimize_solve");
    }


    /////////////////////////////////////////////////////////////////

    virtual void get_meshinfo_integrator(Eigen::VectorXd& tsnd,
                                         Eigen::MatrixXd& mesh_errors,
                                         Eigen::MatrixXd& mesh_dist) const = 0;
    virtual void get_meshinfo_deboor(Eigen::VectorXd& tsnd,
                                     Eigen::MatrixXd& mesh_errors,
                                     Eigen::MatrixXd& mesh_dist) const = 0;
    virtual Eigen::VectorXd calc_global_error() const = 0;


    virtual void initMeshRefinement() {
      this->MeshConverged = false;
      this->MeshIters.resize(0);
    }

    virtual bool checkMesh();
    virtual void updateMesh();


    virtual Eigen::VectorXd calcSwitches();


    auto getMeshInfo(bool integ, int n) {

      Eigen::VectorXd tsnd;
      Eigen::MatrixXd mesh_errors;
      Eigen::MatrixXd mesh_dist;

      this->Table.loadExactData(this->ActiveTraj);


      if (integ) {
        this->get_meshinfo_integrator(tsnd, mesh_errors, mesh_dist);
      } else {
        this->get_meshinfo_deboor(tsnd, mesh_errors, mesh_dist);
      }

      Eigen::VectorXd error = mesh_errors.colwise().lpNorm<Eigen::Infinity>();
      Eigen::VectorXd dist = mesh_dist.colwise().lpNorm<Eigen::Infinity>();

      Eigen::VectorXd distint(dist.size());
      distint[0] = 0;

      for (int i = 0; i < dist.size() - 1; i++) {
        distint[i + 1] = distint[i] + (dist[i]) * (tsnd[i + 1] - tsnd[i]);
      }

      distint = distint / distint[distint.size() - 1];

      Eigen::VectorXd bins;
      bins.setLinSpaced(n + 1, 0.0, 1.0);
      int elem = 0;
      for (int i = 1; i < n; i++) {
        double di = double(i) / double(n);
        auto it = std::upper_bound(distint.cbegin() + elem, distint.cend(), di);
        elem = int(it - distint.cbegin()) - 1;

        double t0 = tsnd[elem];
        double t1 = tsnd[elem + 1];
        double d0 = distint[elem];
        double d1 = distint[elem + 1];
        double slope = (d1 - d0) / (t1 - t0);
        bins[i] = (di - d0) / slope + t0;
      }

      return std::tuple {tsnd, bins, error};
    }


    static void Build(py::module& m);
  };

}  // namespace ASSET
