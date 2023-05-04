
#pragma once

#include "ODESizes.h"
#include "OptimalControlFlags.h"
#include "Solvers/NonLinearProgram.h"
#include "pch.h"

namespace ASSET {

  struct PhaseIndexer : ODESize<-1, -1, -1> {
    using VectorXi = Eigen::VectorXi;
    using MatrixXi = Eigen::MatrixXi;

    int StaticPVars;
    int StatPVars() const {
      return this->StaticPVars;
    }

    std::shared_ptr<NonLinearProgram> nlp;

    VectorXi ODEFirstStateLocs;
    VectorXi ODELastStateLocs;
    VectorXi ODEParamLocs;
    VectorXi StaticParamLocs;

    int numDefects;
    int numStates;
    int numControls;

    bool BlockedControls = false;
    int BlockedControlStart;

    int DefectCardinalStates;
    int numNodalStates;

    int numPhaseVars;
    int numPhaseEqCons = 0;
    int numPhaseIqCons = 0;
    int nextPhaseEqCon = 0;
    int nextPhaseIqCon = 0;

    int StartObj = 0;
    int StartEq = 0;
    int StartIq = 0;

    int StartEqCons = 0;
    int StartIqCons = 0;

    int numObjFuns = 0;
    int numEqFuns = 0;
    int numIqFuns = 0;

    PhaseIndexer() {
    }
    PhaseIndexer(int Xv, int Uv, int OPv, int SPv) {
      this->setXVars(Xv);
      this->setUVars(Uv);
      this->setPVars(OPv);
      this->StaticPVars = SPv;
    }

    void set_dimensions(int DCS, int Dnum, bool BlockCon) {
      this->numDefects = Dnum;
      this->DefectCardinalStates = DCS;
      this->numStates = (this->DefectCardinalStates - 1) * this->numDefects + 1;
      this->numNodalStates = this->numDefects + 1;

      this->BlockedControls = BlockCon;

      if (this->BlockedControls) {
        this->numPhaseVars = this->numStates * this->XtVars() + this->numDefects * this->UVars()
                             + this->PVars() + this->StatPVars();

        ODEFirstStateLocs.setLinSpaced(this->XtUVars(), 0, this->XtUVars() - 1);
        ODEFirstStateLocs.tail(this->UVars()) +=
            VectorXi::Constant(this->UVars(), this->XtVars() * (this->numStates) - this->XtVars());

        ODELastStateLocs =
            ODEFirstStateLocs + VectorXi::Constant(this->XtUVars(), this->XtVars() * (this->numStates - 1));
        ODELastStateLocs.tail(this->UVars()) =
            ODEFirstStateLocs.tail(this->UVars())
            + VectorXi::Constant(this->UVars(), this->UVars() * (this->numDefects - 1));

      } else {
        this->numPhaseVars = this->numStates * this->XtUVars() + this->PVars() + this->StatPVars();

        ODEFirstStateLocs.setLinSpaced(this->XtUVars(), 0, this->XtUVars() - 1);
        ODELastStateLocs =
            ODEFirstStateLocs + VectorXi::Constant(this->XtUVars(), this->XtUVars() * (this->numStates - 1));
      }

      ODEParamLocs.setLinSpaced(this->PVars(), 0, this->PVars() - 1);
      ODEParamLocs +=
          Eigen::VectorXi::Constant(this->PVars(), this->numPhaseVars - this->PVars() - this->StatPVars());

      StaticParamLocs.setLinSpaced(this->StatPVars(), 0, this->StatPVars() - 1);
      StaticParamLocs += Eigen::VectorXi::Constant(this->StatPVars(), this->numPhaseVars - this->StatPVars());
    }
    void begin_indexing(std::shared_ptr<NonLinearProgram> np, int n, int ep, int ip) {
      this->nlp = np;

      this->numPhaseEqCons = 0;
      this->numPhaseIqCons = 0;

      this->ODEFirstStateLocs += Eigen::VectorXi::Constant(this->ODEFirstStateLocs.size(), n);
      this->ODELastStateLocs += Eigen::VectorXi::Constant(this->ODELastStateLocs.size(), n);
      this->ODEParamLocs += Eigen::VectorXi::Constant(this->ODEParamLocs.size(), n);
      this->StaticParamLocs += Eigen::VectorXi::Constant(this->StaticParamLocs.size(), n);
      this->nextPhaseEqCon = ep;
      this->nextPhaseIqCon = ip;

      this->StartEqCons = ep;
      this->StartIqCons = ip;

      this->StartObj = this->nlp->Objectives.size();
      this->StartEq = this->nlp->EqualityConstraints.size();
      this->StartIq = this->nlp->InequalityConstraints.size();

      this->numObjFuns = 0;
      this->numEqFuns = 0;
      this->numIqFuns = 0;
    }

    int addEquality(ConstraintInterface eqfun,
                    PhaseRegionFlags sreg,
                    const Eigen::VectorXi& rxtuv,
                    const Eigen::VectorXi& rodepv,
                    const Eigen::VectorXi& rstatpv,
                    ThreadingFlags Tmode);
    void addPartitionedEquality(const std::vector<ConstraintInterface>& eqfuns,
                                PhaseRegionFlags sreg,
                                const Eigen::VectorXi& rxtuv,
                                const Eigen::VectorXi& rodepv,
                                const Eigen::VectorXi& rstatpv,
                                const std::vector<int>& Tmodes);

    int addAccumulation(ConstraintInterface eqfun,
                        PhaseRegionFlags sreg,
                        const Eigen::VectorXi& rxtuv,
                        const Eigen::VectorXi& rodepv,
                        const Eigen::VectorXi& rstatpv,
                        ConstraintInterface accfun,
                        const Eigen::VectorXi& accpv,
                        ThreadingFlags Tmode);

    int addInequality(ConstraintInterface iqfun,
                      PhaseRegionFlags sreg,
                      const Eigen::VectorXi& rxtuv,
                      const Eigen::VectorXi& rodepv,
                      const Eigen::VectorXi& rstatpv,
                      ThreadingFlags Tmode);

    void addPartitionedInequality(const std::vector<ConstraintInterface>& iqfuns,
                                  PhaseRegionFlags sreg,
                                  const Eigen::VectorXi& rxtuv,
                                  const Eigen::VectorXi& rodepv,
                                  const Eigen::VectorXi& rstatpv,
                                  const std::vector<int>& Tmodes);

    int addObjective(ObjectiveInterface objfun,
                     PhaseRegionFlags sreg,
                     const Eigen::VectorXi& rxtuv,
                     const Eigen::VectorXi& rodepv,
                     const Eigen::VectorXi& rstatpv,
                     ThreadingFlags Tmode);

    int getXTUVarLoc(int vloc, int State) const {
      int v = 0;
      if (this->BlockedControls) {
        if (vloc < XtVars()) {
          v = this->ODEFirstStateLocs[vloc] + State * this->XtVars();
        } else {
          int unum = State / (this->DefectCardinalStates - 1);
          if (unum > (this->numDefects - 1))
            unum = this->numDefects - 1;
          v = this->ODEFirstStateLocs[vloc] + unum * this->UVars();
        }
      } else {
        v = this->ODEFirstStateLocs[vloc] + State * this->XtUVars();
      }
      return v;
    }

    int getXTUVarLoc(int vloc, int State, int Defect) const {
      int v = 0;
      if (this->BlockedControls) {
        if (vloc < XtVars()) {
          v = this->ODEFirstStateLocs[vloc] + State * this->XtVars();
        } else {
          v = this->ODEFirstStateLocs[vloc] + Defect * this->UVars();
        }
      } else {
        v = this->ODEFirstStateLocs[vloc] + State * this->XtUVars();
      }
      return v;
    }

    std::array<Eigen::MatrixXi, 2> make_Vindex_Cindex(PhaseRegionFlags sreg,
                                                      const VectorXi& rxtuv,
                                                      const VectorXi& rodepv,
                                                      const VectorXi& rstatpv,
                                                      int orows,
                                                      int& NextCLoc) const;

    std::array<Eigen::MatrixXi, 2> make_Vindex_Cindex(PhaseRegionFlags sreg,
                                                      const VectorXi& rxtuv,
                                                      const VectorXi& rodepv,
                                                      const VectorXi& rstatpv,
                                                      int orows) const {
      int dummy = 0;
      return this->make_Vindex_Cindex(sreg, rxtuv, rodepv, rstatpv, orows, dummy);
    }

    Eigen::VectorXd makeSolverInput(const std::vector<Eigen::VectorXd>& ActiveTraj,
                                    const Eigen::VectorXd& ActiveStaticParams) const;
    void collectSolverOutput(const Eigen::VectorXd& Vars,
                             std::vector<Eigen::VectorXd>& ActiveTraj,
                             Eigen::VectorXd& ActiveStaticParams) const;

    std::vector<Eigen::VectorXd> getFuncEqMultipliers(int Gindex, const Eigen::VectorXd& EMultphase) const;

    std::vector<Eigen::VectorXd> getFuncIqMultipliers(int Gindex, const Eigen::VectorXd& IMultphase) const;

    void print_stats(bool showfuns) const;

    static void Test();
  };

}  // namespace ASSET
