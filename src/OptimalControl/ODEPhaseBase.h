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
#include "CommonFunctions/IOScaled.h"
#include "InterfaceTypes.h"

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

    //////////////////////////

    Eigen::VectorXd XtUPUnits;
    Eigen::VectorXd SPUnits;

    std::map<std::string, Eigen::VectorXi> SPidxs;


    ///////////////////////
   public:
    bool AutoScaling = false;
    bool SyncObjectiveScales = true;

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


    void setAutoScaling(bool autoscale) {
        this->AutoScaling = autoscale;
        this->resetTranscription();
        this->invalidatePostOptInfo();
    }


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

      this->XtUPUnits = Eigen::VectorXd::Ones(this->XtUPVars());

    }
    virtual ~ODEPhaseBase() = default;

    //////////////////////////////////////////////////

    virtual void setUnits(const Eigen::VectorXd& XtUPUnits_) = 0;
    virtual void setUnits(const py::kwargs&);


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


    ///////////////////////////////////////////////////

    void setStaticParamVgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
        this->SPidxs = spidxs;
    }
    void addStaticParamVgroups(std::map<std::string, Eigen::VectorXi> spidxs) {
      for(auto& [key, value] : spidxs) {
		this->SPidxs[key] = value;
	  }
    }
    void addStaticParamVgroup(Eigen::VectorXi idx, std::string key) {
        this->SPidxs[key] = idx;
    }
    void addStaticParamVgroup(int idx, std::string key) {
        VectorXi tmp(1);
        tmp << idx;
        this->SPidxs[key] = tmp;
    }

    VectorXi getSPidx(std::string key) const {
        if (SPidxs.count(key) == 0) {
            throw std::invalid_argument(
                fmt::format("No StaticParam variable index group with name: {0:} exists.", key));
        }
		return this->SPidxs.at(key);
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

   

    
    PhaseRegionFlags getRegion(RegionType reg_t) const{
        PhaseRegionFlags reg;

        if (std::holds_alternative<PhaseRegionFlags>(reg_t)) {
            reg = std::get<PhaseRegionFlags>(reg_t);
        }
        else if (std::holds_alternative<std::string>(reg_t)) {
            reg = strto_PhaseRegionFlag(std::get<std::string>(reg_t));
        }
        return reg;
    }


    VectorXi getXtUPVars(PhaseRegionFlags reg,VarIndexType XtUPvars_t) const {

        VectorXi XtUPvars;
        
        /////////////////////////////////////////////////
        if (std::holds_alternative<int>(XtUPvars_t)) {
            XtUPvars.resize(1);
            XtUPvars[0] = std::get<int>(XtUPvars_t);
        }
        else if (std::holds_alternative<VectorXi>(XtUPvars_t)) {
            XtUPvars = std::get<VectorXi>(XtUPvars_t);
        }
        else if (std::holds_alternative<std::string>(XtUPvars_t)) {
            if (reg != StaticParams) {
                XtUPvars = this->idx(std::get<std::string>(XtUPvars_t));
            }
            else {
                XtUPvars = this->getSPidx(std::get<std::string>(XtUPvars_t));
            }
            if (reg == ODEParams) {
                // Convert to 0 based index
                for (int i = 0; i < XtUPvars.size(); i++) {
                    XtUPvars[i] -= this->XtUVars();
                }
            }
        }
        else if (std::holds_alternative<std::vector<std::string>>(XtUPvars_t)) {

            std::vector<VectorXi> varvec;
            int size = 0;

            auto tmpvars = std::get<std::vector<std::string>>(XtUPvars_t);

            for (auto tmpv : tmpvars) {
                if (reg != StaticParams) {
                    varvec.push_back(this->idx(tmpv));
                }
                else {
                    varvec.push_back(this->getSPidx(tmpv));
                }

                size += varvec.back().size();
            }
            XtUPvars.resize(size);

            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    XtUPvars[next] = varv[i];
                    next++;
                }
            }

            if (reg == ODEParams) {
                // Convert to 0 based index
                for (int i = 0; i < XtUPvars.size(); i++) {
                    XtUPvars[i] -= this->XtUVars();
                }
            }

            
        }
        return XtUPvars;
    }

    VectorXi getOPVars(PhaseRegionFlags reg, VarIndexType OPvars_t) const {

        VectorXi OPvars;

        if (std::holds_alternative<int>(OPvars_t)) {
            OPvars.resize(1);
            OPvars[0] = std::get<int>(OPvars_t);
        }
        else if (std::holds_alternative<VectorXi>(OPvars_t)) {
            OPvars = std::get<VectorXi>(OPvars_t);
        }
        else if (std::holds_alternative<std::string>(OPvars_t)) {
            OPvars = this->idx(std::get<std::string>(OPvars_t));

            for (int i = 0; i < OPvars.size(); i++) {
                // Convert to 0 based index
                OPvars[i] -= this->XtUVars();
            }
        }
        else if (std::holds_alternative<std::vector<std::string>>(OPvars_t)) {
            std::vector<VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(OPvars_t);
            for (auto tmpv : tmpvars) {
                varvec.push_back(this->idx(tmpv));
                size += varvec.back().size();
            }
            OPvars.resize(size);

            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    // Convert to 0 based index
                    OPvars[next] = varv[i] - this->XtUVars();
                    next++;
                }
            }
        }

        return OPvars;
    }

    VectorXi getSPVars(PhaseRegionFlags reg, VarIndexType SPvars_t) const {

        VectorXi SPvars;

        if (std::holds_alternative<int>(SPvars_t)) {
            SPvars.resize(1);
            SPvars[0] = std::get<int>(SPvars_t);
        }
        else if (std::holds_alternative<VectorXi>(SPvars_t)) {
            SPvars = std::get<VectorXi>(SPvars_t);
        }
        else if (std::holds_alternative<std::string>(SPvars_t)) {
            SPvars = this->getSPidx(std::get<std::string>(SPvars_t));
        }
        else if (std::holds_alternative<std::vector<std::string>>(SPvars_t)) {
            std::vector<VectorXi> varvec;
            int size = 0;
            auto tmpvars = std::get<std::vector<std::string>>(SPvars_t);
            for (auto tmpv : tmpvars) {
                varvec.push_back(this->getSPidx(tmpv));
                size += varvec.back().size();
            }
            SPvars.resize(size);
            int next = 0;
            for (auto varv : varvec) {
                for (int i = 0; i < varv.size(); i++) {
                    SPvars[next] = varv[i];
                    next++;
                }
            }
        }
        

        return SPvars;
    }



    template<class FuncHolder, class FuncType>
    FuncHolder makeFuncImpl(RegionType reg_t,
                     FuncType fun,
                     VarIndexType XtUPvars_t,
                     VarIndexType OPvars_t,
                     VarIndexType SPvars_t,
                     ScaleType scale_t) {


        PhaseRegionFlags reg = getRegion(reg_t);
        FuncHolder func;

        if (std::holds_alternative<std::string>(XtUPvars_t)) {
            std::string vars = std::get<std::string>(XtUPvars_t);
            if (vars == "All" || vars == "all" ||vars=="XtUP") {
                // Default case where the function has same inputs and order as ODE
                VectorXi XtUPvars;
                VectorXi OPvars;
                VectorXi SPvars;
                XtUPvars.setLinSpaced(this->XtUVars(), 0, this->XtUVars() - 1);
                if (this->PVars() > 0) {
                    OPvars.setLinSpaced(this->PVars(), 0, this->PVars() - 1);
                }
                func = FuncHolder(fun, reg, XtUPvars, OPvars, SPvars, scale_t);
                return func; // return early
            }
        }

        VectorXi XtUPvars = this->getXtUPVars(reg,XtUPvars_t);
        VectorXi OPvars;
        VectorXi SPvars;
        /////////////////////////////////////////////////
        if (reg != ODEParams && reg != StaticParams) {
            // If region is Params then the indices are held in XtUPvars_t and the others are emtpy
            OPvars = this->getOPVars(reg, OPvars_t);
            SPvars = this->getSPVars(reg, SPvars_t);
            func = FuncHolder(fun, reg, XtUPvars, OPvars, SPvars, scale_t);

        }
        else {
            func = FuncHolder(fun, reg, XtUPvars, scale_t);
        }
        
        return func;
    }
   


    
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int addEqualCon(StateConstraint con) {
        return addFuncImpl(con, this->userEqualities, "Equality Constraint");
    }

    int addEqualCon(RegionType reg_t,
        VectorFunctionalX fun,
        VarIndexType XtUPvars_t,
        VarIndexType OPvars_t,
        VarIndexType SPvars_t,
        ScaleType scale_t) {

        auto con = makeFuncImpl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        return addFuncImpl(con, this->userEqualities, "Equality Constraint");
    }

    int addEqualCon(RegionType reg_t,
        VectorFunctionalX fun,
        VarIndexType XtUPvars_t,
        ScaleType scale_t) {

        VectorXi empty;
     
        auto con = makeFuncImpl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t, empty, empty, scale_t);
        return addFuncImpl(con, this->userEqualities, "Equality Constraint");
    }

    int addBoundaryValue(RegionType reg, VarIndexType args, const std::variant<double,VectorXd> & value, ScaleType scale_t);
    int addDeltaVarEqualCon(VarIndexType var, double value, double scale, ScaleType scale_t);
    int addDeltaTimeEqualCon(double value, double scale, ScaleType scale_t) {
        return this->addDeltaVarEqualCon(this->TVar(), value, scale, scale_t);

    }
    int addValueLock(RegionType reg, VarIndexType args, ScaleType scale_t);
    int addPeriodicityCon(VarIndexType args, ScaleType scale_t);

    /////////////////////////////////////////////////
    
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int addInequalCon(StateConstraint con) {
        return addFuncImpl(con, this->userInequalities, "Inequality Constraint");
    }
    int addInequalCon(RegionType reg_t,
        VectorFunctionalX fun,
        VarIndexType XtUPvars_t,
        VarIndexType OPvars_t,
        VarIndexType SPvars_t,
        ScaleType scale_t) {

        auto con = makeFuncImpl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        return addFuncImpl(con, this->userInequalities, "Inequality Constraint");
    }

    int addInequalCon(RegionType reg_t,
        VectorFunctionalX fun,
        VarIndexType XtUPvars_t,
        ScaleType scale_t) {

        VectorXi empty;

        auto con = makeFuncImpl<StateConstraint, VectorFunctionalX>(reg_t, fun, XtUPvars_t, empty, empty, scale_t);
        return addFuncImpl(con, this->userInequalities, "Inequality Constraint");
    }

    ////////////////////////
    int addLUVarBound(
        RegionType reg, VarIndexType var, double lowerbound, double upperbound, double lbscale, double ubscale,
        ScaleType scale_t);

    int addLUVarBound(
        RegionType reg, VarIndexType var, double lowerbound, double upperbound, double scale,
        ScaleType scale_t) {
        return this->addLUVarBound(reg, var, lowerbound, upperbound, scale, scale, scale_t);
    }
    int addLUVarBound(
        RegionType reg, VarIndexType var, double lowerbound, double upperbound,
        ScaleType scale_t) {
        return this->addLUVarBound(reg, var, lowerbound, upperbound, 1.0, 1.0, scale_t);
    }

    int addLowerVarBound(
        RegionType reg, VarIndexType var, double lowerbound, double lbscale,ScaleType scale_t);

    int addUpperVarBound(
        RegionType reg, VarIndexType var, double upperbound, double ubscale, ScaleType scale_t);

    int addLUFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        VarIndexType OPvars,
        VarIndexType SPvars,
        double lowerbound,
        double upperbound,
        double lbscale,
        double ubscale, ScaleType scale_t);

    

    int addLUFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        double lowerbound,
        double upperbound,
        double lbscale,
        double ubscale, ScaleType scale_t) {

        VectorXi empty;

        return addLUFuncBound(reg, func, XtUPvars, empty, empty, lowerbound,upperbound, lbscale, ubscale, scale_t);
    }

    int addLUFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        VarIndexType OPvars,
        VarIndexType SPvars,
        double lowerbound,
        double upperbound,
        double scale,
        ScaleType scale_t) {
        return addLUFuncBound(reg, func, XtUPvars, OPvars, SPvars, lowerbound, upperbound, scale, scale, scale_t);
    }

    int addLUFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        double lowerbound,
        double upperbound,
        double scale,
        ScaleType scale_t) {
        VectorXi empty;
        return addLUFuncBound(reg, func, XtUPvars, empty, empty, lowerbound, upperbound, scale, scale, scale_t);
    }

    int addLowerFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        VarIndexType OPvars,
        VarIndexType SPvars,
        double lowerbound,
        double lbscale,
        ScaleType scale_t);

    int addLowerFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        double lowerbound,
        double lbscale,
        ScaleType scale_t) {
        VectorXi empty;

        return this->addLowerFuncBound(reg, func, XtUPvars, empty, empty, lowerbound, lbscale, scale_t);
    }


    int addUpperFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        VarIndexType OPvars,
        VarIndexType SPvars,
        double upperbound,
        double ubscale,
        ScaleType scale_t);

    int addUpperFuncBound(RegionType reg,
        ScalarFunctionalX func,
        VarIndexType XtUPvars,
        double upperbound,
        double ubscale,
        ScaleType scale_t) {
        VectorXi empty;

        return this->addUpperFuncBound(reg, func, XtUPvars, empty, empty, upperbound, ubscale, scale_t);
    }

    int addLUNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double lowerbound,
        double upperbound,
        double lbscale,
        double ubscale,
        ScaleType scale_t);

    int addLUNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double lowerbound,
        double upperbound,
        double scale,
        ScaleType scale_t) {
        return this->addLUNormBound(reg, XtUPvars, lowerbound, upperbound, scale, scale, scale_t);
    }

    int addLUSquaredNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double lowerbound,
        double upperbound,
        double lbscale,
        double ubscale,
        ScaleType scale_t);

    int addLUSquaredNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double lowerbound,
        double upperbound,
        double scale,
        ScaleType scale_t) {
        return this->addLUSquaredNormBound(reg, XtUPvars, lowerbound, upperbound, scale, scale, scale_t);
    }


    int addLowerNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double lowerbound,
        double lbscale,
        ScaleType scale_t);

    int addLowerSquaredNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double lowerbound,
        double lbscale,
        ScaleType scale_t);

    int addUpperNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double upperbound,
        double ubscale,
        ScaleType scale_t);

    int addUpperSquaredNormBound(RegionType reg,
        VarIndexType XtUPvars,
        double upperbound,
        double ubscale,
        ScaleType scale_t);
    //
    int addLowerDeltaVarBound(RegionType reg, VarIndexType var, double lowerbound, double lbscale,
        ScaleType scale_t);
    int addLowerDeltaVarBound(VarIndexType var, double lowerbound, double lbscale,
        ScaleType scale_t) {
        return this->addLowerDeltaVarBound(PhaseRegionFlags::FrontandBack, var, lowerbound, lbscale, scale_t);
    }
    int addLowerDeltaTimeBound(double lowerbound, double lbscale,
        ScaleType scale_t) {
        return this->addLowerDeltaVarBound(this->TVar(), lowerbound, lbscale, scale_t);
    }
    ///
    int addUpperDeltaVarBound(RegionType reg, VarIndexType var, double upperbound, double ubscale,
        ScaleType scale_t);
    int addUpperDeltaVarBound(VarIndexType var, double upperbound, double ubscale,
        ScaleType scale_t) {
        return this->addUpperDeltaVarBound(PhaseRegionFlags::FrontandBack, var, upperbound, ubscale, scale_t);
    }
    int addUpperDeltaTimeBound(double upperbound, double ubscale,
        ScaleType scale_t) {
        return this->addUpperDeltaVarBound(this->TVar(), upperbound, ubscale, scale_t);
    }

    ///////////////////////////////////////////////////////////////
    
    

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

    Eigen::VectorXi addLUVarBounds(
        std::string reg, Eigen::VectorXi vars, double lowerbound, double upperbound, double scale) {
      return addLUVarBounds(strto_PhaseRegionFlag(reg), vars, lowerbound, upperbound, scale);
    }

    
    ////////////////////////////////////////////////////


   
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    //////////////////////////////////////////////////
    int addStateObjective(StateObjective obj) {
        return addFuncImpl(obj, this->userStateObjectives, "State Objective");
    }
    int addStateObjective(RegionType reg_t,
        ScalarFunctionalX fun,
        VarIndexType XtUPvars_t,
        VarIndexType OPvars_t,
        VarIndexType SPvars_t,
        ScaleType scale_t) {

        auto con = makeFuncImpl<StateObjective, ScalarFunctionalX>(reg_t, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        return addFuncImpl(con, this->userStateObjectives, "State Objective");
    }

    int addStateObjective(RegionType reg_t,
        ScalarFunctionalX fun,
        VarIndexType XtUPvars_t,
        ScaleType scale_t) {

        VectorXi empty;

        auto con = makeFuncImpl<StateObjective, ScalarFunctionalX>(reg_t, fun, XtUPvars_t, empty, empty, scale_t);
        return addFuncImpl(con, this->userStateObjectives, "State Objective");
    }
    int addValueObjective(RegionType reg, VarIndexType var, double scale,ScaleType scale_t);
    int addDeltaVarObjective(VarIndexType var, double scale, ScaleType scale_t);
    int addDeltaTimeObjective(double scale, ScaleType scale_t) {
        return this->addDeltaVarObjective(this->TVar(), scale, scale_t);
    }

    ///////////////////////////////////////////////
    
    


    
    ///////////////////////////////////////////////////
    int addIntegralObjective(StateObjective obj) {
        return addFuncImpl(obj, this->userIntegrands, "Integral Objective");
    }

    int addIntegralObjective(
        ScalarFunctionalX fun,
        VarIndexType XtUPvars_t,
        VarIndexType OPvars_t,
        VarIndexType SPvars_t,
        ScaleType scale_t) {

        auto con = makeFuncImpl<StateObjective, ScalarFunctionalX>(Path, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        return addFuncImpl(con, this->userIntegrands, "Integral Objective");
    }

    int addIntegralObjective(
        ScalarFunctionalX fun,
        VarIndexType XtUPvars_t,
        ScaleType scale_t) {

        VectorXi empty;

        auto con = makeFuncImpl<StateObjective, ScalarFunctionalX>(Path, fun, XtUPvars_t, empty, empty, scale_t);
        return addFuncImpl(con, this->userIntegrands, "Integral Objective");
    }


    ///////////////////////////////////////////////////
    int addIntegralParamFunction(StateObjective con, int pv) {
        VectorXi epv(1);
        epv[0] = pv;
        int index = addFuncImpl(con, this->userParamIntegrands, "Integral Parameter Function");
        this->userParamIntegrands[index].EXTVars = epv;
        return index;
    }

    int addIntegralParamFunction(
        ScalarFunctionalX fun,
        VarIndexType XtUPvars_t,
        VarIndexType OPvars_t,
        VarIndexType SPvars_t,
        int accum_parm,
        ScaleType scale_t) {

        VectorXi epv(1);
        epv[0] = accum_parm;

        auto con = makeFuncImpl<StateObjective, ScalarFunctionalX>(Path, fun, XtUPvars_t, OPvars_t, SPvars_t, scale_t);
        int index = addFuncImpl(con, this->userParamIntegrands, "Integral Parameter Function");
        this->userParamIntegrands[index].EXTVars = epv;
        return index;
    }

    int addIntegralParamFunction(
        ScalarFunctionalX fun,
        VarIndexType XtUPvars_t,
        int accum_parm,
        ScaleType scale_t) {
        VectorXi empty;
        return addIntegralParamFunction(fun, XtUPvars_t, empty, empty, accum_parm, scale_t);
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

    void setTraj(const std::vector<Eigen::VectorXd>& mesh);

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


    void setStaticParams(VectorXd parm,VectorXd units) {
      if (units.size() != parm.size()) {
          throw std::invalid_argument("Size of static parameter vector and scaling units vector must match");
      }

      this->ActiveStaticParams = parm;
      this->numStatParams = parm.size();
      this->resetTranscription();
      this->SPUnits = units;
    }
    void setStaticParams(VectorXd parm) {
        VectorXd units(parm.size());
        units.setOnes();
        return this->setStaticParams(parm, units);
    }

    void addStaticParams(VectorXd parm, VectorXd units) {
	  if (this->numStatParams == 0) {
		this->setStaticParams(parm,units);
	  } else {
          VectorXd parmstmp(this->ActiveStaticParams.size()+ parm.size());
          parmstmp << this->ActiveStaticParams, parm;
          VectorXd unitstmp(this->SPUnits.size() + units.size());
          unitstmp << this->SPUnits, units;
          this->setStaticParams(parmstmp,unitstmp);
	  }
	}
	void addStaticParams(VectorXd parm) {
		VectorXd units(parm.size());
		units.setOnes();
		return this->addStaticParams(parm, units);
	}


    void subStaticParams(VectorXd parm) {
      if (this->numStatParams == parm.size()) {
          this->ActiveStaticParams = parm;
      } else {
          this->setStaticParams(parm);
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
    Eigen::VectorXd returnEqualConScales(int index) const {
       return this->userEqualities.at(index).OutputScales;
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
    Eigen::VectorXd returnInequalConScales(int index) const {
        return this->userInequalities.at(index).OutputScales;
    }

    std::vector<Eigen::VectorXd> returnCostateTraj() const;
    std::vector<Eigen::VectorXd> returnTrajError() const;

    Eigen::VectorXd returnIntegralObjectiveScales(int index) const {
        return this->userIntegrands.at(index).OutputScales;
    }
    Eigen::VectorXd returnIntegralParamFunctionScales(int index) const {
        return this->userParamIntegrands.at(index).OutputScales;
    }
    Eigen::VectorXd returnStateObjectiveScales(int index) const {
        return this->userStateObjectives.at(index).OutputScales;
    }
    Eigen::VectorXd returnODEOutputScales() const {
        VectorXd output_scales = XtUPUnits.head(this->XVars()).cwiseInverse() * this->XtUPUnits[this->XVars()];
        return output_scales;
    }

    /////////////////////////////////////////////////
   protected:
    virtual void transcribe_dynamics() = 0;
    virtual void transcribe_axis_funcs();
    virtual void transcribe_control_funcs();
    virtual void transcribe_integrals();
    virtual void transcribe_basic_funcs();

    void initIndexing() {
      this->indexer = PhaseIndexer(this->XVars(), this->UVars(), this->PVars(), this->numStatParams);
      this->indexer.set_dimensions(this->numTranCardStates, this->numDefects, 
          this->ControlMode == ControlModes::BlockConstant);
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
        case InnerPath:
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

    Eigen::VectorXd get_input_scale(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV, VectorXi SPV) const;

    std::vector<Eigen::VectorXd> get_test_inputs(PhaseRegionFlags flag, VectorXi XtUV, VectorXi OPV, VectorXi SPV) const;

    void calc_auto_scales();

    std::vector<double> get_objective_scales();
    void update_objective_scales(double scale);
    


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

        if (this->AutoScaling) {
            auto ActiveTrajTmp = this->ActiveTraj;

            for (auto& T : ActiveTrajTmp) {
                T = T.cwiseQuotient(this->XtUPUnits);
            }

            VectorXd StaticParamsTmp;
            if (this->ActiveStaticParams.size() > 0 && this->SPUnits.size()>0) {
                StaticParamsTmp = this->ActiveStaticParams.cwiseQuotient(this->SPUnits);
            }
            return this->indexer.makeSolverInput(ActiveTrajTmp, StaticParamsTmp);

      
        } else {
        return this->indexer.makeSolverInput(this->ActiveTraj, this->ActiveStaticParams);
        }
    }
    void collectSolverOutput(const VectorXd& Vars) {
      this->indexer.collectSolverOutput(Vars, this->ActiveTraj, this->ActiveStaticParams);

      if (this->AutoScaling) {
          for (auto& T : this->ActiveTraj) {
              T = T.cwiseProduct(this->XtUPUnits);
          }
          if (this->ActiveStaticParams.size() > 0 && this->SPUnits.size() > 0) {
              this->ActiveStaticParams = this->ActiveStaticParams.cwiseProduct(this->SPUnits);
          }

      }

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
