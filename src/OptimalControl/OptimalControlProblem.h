#pragma once

#include "LinkFunction.h"
#include "ODEPhaseBase.h"
#include "pch.h"

namespace ASSET {

struct OptimalControlProblem:OptimizationProblemBase {
  using VectorXi = Eigen::VectorXi;
  using MatrixXi = Eigen::MatrixXi;

  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;

  using VectorFunctionalX = GenericFunction<-1, -1>;
  using ScalarFunctionalX = GenericFunction<-1, 1>;

  using RegVec = Eigen::Matrix<PhaseRegionFlags, -1, 1>;

  using LinkConstraint = LinkFunction<VectorFunctionalX>;
  using LinkObjective = LinkFunction<ScalarFunctionalX>;
  using StateConstraint = StateFunction<VectorFunctionalX>;
  using StateObjective = StateFunction<ScalarFunctionalX>;
  using StateIntegral = StateFunction<ScalarFunctionalX>;
  using PhasePtr = std::shared_ptr<ODEPhaseBase>;

  using PhaseIndexPack = std::tuple<int,std::string, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>;
  using PhaseIndexPackPtr = std::tuple<PhasePtr, std::string, Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi>;


  std::vector<PhasePtr> phases;
  std::vector<std::string> phase_names;

  

  bool doTranscription = true;
  void resetTranscription() { this->doTranscription = true; };

  std::vector<LinkConstraint> LinkEqualities;
  std::vector<LinkConstraint> LinkInequalities;
  std::vector<LinkObjective>  LinkObjectives;

  VectorXd ActiveLinkParams;
  void setLinkParams(VectorXd lp) {
    this->ActiveLinkParams = lp;
    this->numLinkParams = lp.size();
  }
  VectorXd returnLinkParams() { return this->ActiveLinkParams; }

  VectorXd ActiveEqLmults;
  VectorXd ActiveIqLmults;
  bool MultipliersLoaded = false;

  VectorXi LinkParamLocs;

  VectorXi numPhaseVars;
  VectorXi numPhaseEqCons;
  VectorXi numPhaseIqCons;

  int numLinkParams = 0;
  int numLinkEqCons = 0;
  int numLinkIqCons = 0;

  int StartObj = 0;
  int StartEq = 0;
  int StartIq = 0;

  int numObjFuns = 0;
  int numEqFuns = 0;
  int numIqFuns = 0;

  int numProbVars = 0;
  int numProbEqCons = 0;
  int numProbIqCons = 0;

  ///////////////////////////////
  bool AdaptiveMesh = false;
  bool PrintMeshInfo = true;
  bool SolveOnlyFirst = true;

  int MaxMeshIters = 10;
  PSIOPT::ConvergenceFlags MeshAbortFlag = PSIOPT::ConvergenceFlags::DIVERGING;

  bool MeshConverged = false;

  void setAdaptiveMesh(bool amesh, bool applytophases) {
      this->AdaptiveMesh = amesh;
      if (applytophases) {
          for (auto phase : this->phases) {
              phase->setAdaptiveMesh(amesh);
          }
      }
  }
  

  ///////////////////////////////
  OptimalControlProblem() {

  }
  OptimalControlProblem(std::vector<PhasePtr> ps) {
      this->addPhases(ps);
  }

  int addPhase(PhasePtr p) {
    this->resetTranscription();
    this->phases.push_back(p);
    int index = int(this->phases.size()) - 1;
    this->phase_names.push_back(std::to_string(index));
    check_dupilcate_phases();
    return index;
  }

  std::vector<int> addPhases(std::vector<PhasePtr> ps) {
      std::vector<int> idxs;
      for (auto p : ps) {
          idxs.push_back(this->addPhase(p));
      }
      return idxs;
  }

  int addPhase(PhasePtr p,const std::string & name) {
      this->resetTranscription();
      this->phases.push_back(p);
      int index = int(this->phases.size()) - 1;
      this->phase_names.push_back(name);
      check_dupilcate_phases();
      return index;
  }

  int getPhaseNum(const std::string& name) {
      auto nameit = std::find(phase_names.begin(), phase_names.end(), name);
      if (nameit == phase_names.end()) {
          fmt::print(fmt::fg(fmt::color::red),
              "Transcription Error!!!\n"
              "No phase with name '{0}' exists in OptimalControlProblem.\n", name);
          throw std::invalid_argument("");
      }
      return int(nameit - phase_names.begin());
  }

  int getPhaseNum(PhasePtr p) {
      auto ptrit = std::find_if(phases.begin(), phases.end(),
          [&](PhasePtr pt) {
              return pt.get() == p.get();
          });
      if (ptrit == phases.end()) {
          fmt::print(fmt::fg(fmt::color::red),
              "Transcription Error!!!\n"
              "The requested phase does not exist in OptimalControlProblem\n");
          throw std::invalid_argument("");
      }
      return int(ptrit - phases.begin());
  }

  std::vector<VectorXi> ptl_from_phase_names(std::vector<std::vector<std::string>> ptlnamevec) {
      std::vector<VectorXi> ptl;
      for (auto& appl : ptlnamevec) {
          VectorXi ptlv(appl.size());
          for (int i = 0; i < appl.size(); i++) {
              ptlv[i] = this->getPhaseNum(appl[i]);
          }
          ptl.push_back(ptlv);
      }
      return ptl;
  }

  std::vector<VectorXi> ptl_from_phases(std::vector<std::vector<PhasePtr>> ptlnamevec) {
      std::vector<VectorXi> ptl;
      for (auto& appl : ptlnamevec) {
          VectorXi ptlv(appl.size());
          for (int i = 0; i < appl.size(); i++) {
              ptlv[i] = this->getPhaseNum(appl[i]);
          }
          ptl.push_back(ptlv);
      }
      return ptl;
  }


  void removePhase(int ith) {
    this->resetTranscription();
    if (ith < 0) ith = (this->phases.size() + ith);
    this->phases.erase(this->phases.begin() + ith);
    this->phase_names.erase(this->phase_names.begin() + ith);

  }
  PhasePtr Phase(int ith) {
    if (ith < 0) ith = (this->phases.size() + ith);
    return this->phases[ith];
  }

  /////////////////////////////////////////////////

  template<class FuncType,class PackType, class OutType>
  OutType makeLinkFunc(FuncType f,std::vector<PackType> packs, VectorXi lv) {

      int npacks = packs.size();
      std::vector<Eigen::VectorXi> PTL;
      VectorXi phasenums(npacks);
      Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags(npacks);
      std::vector<Eigen::VectorXi> xtvs(npacks);
      std::vector<Eigen::VectorXi> opvs(npacks);
      std::vector<Eigen::VectorXi> spvs(npacks);

      for (int i = 0; i < npacks; i++) {
          if constexpr (std::is_same<PackType, PhaseIndexPack>::value) {
              phasenums[i] = std::get<0>(packs[i]);
          }
          else {
              phasenums[i] = this->getPhaseNum(std::get<0>(packs[i]));
          }
          
          RegFlags[i] = strto_PhaseRegionFlag(std::get<1>(packs[i]));
          xtvs[i] = std::get<2>(packs[i]);
          opvs[i] = std::get<3>(packs[i]);
          spvs[i] = std::get<4>(packs[i]);
      }

      PTL.push_back(phasenums);
      std::vector<Eigen::VectorXi> lvs;
      lvs.push_back(lv);
      return OutType(f, RegFlags, PTL, xtvs, opvs, spvs, lvs);
  }
  template<class FuncType, class PhaseType, class OutType>
  OutType makeLinkFunc(FuncType f, 
      PhaseType phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhaseType phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {

      auto pack0 = std::tuple{ phase0,reg0,xtv0,opv0,spv0 };
      auto pack1 = std::tuple{ phase1,reg1,xtv1,opv1,spv1 };
      auto packs = std::vector{ pack0,pack1 };
      return this->makeLinkFunc<FuncType,decltype(pack0),OutType>(f, packs, lv);
  }

  template<class FuncType, class PhaseType, class OutType>
  OutType makeLinkFunc(FuncType f,
      PhaseType phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhaseType phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {

      auto pack0 = std::tuple{ phase0,reg0,xtv0,opv0,spv0 };
      auto pack1 = std::tuple{ phase1,reg1,xtv1,opv1,spv1 };
      auto packs = std::vector{ pack0,pack1 };
      Eigen::VectorXi lv;
      lv.resize(0);
      return this->makeLinkFunc<FuncType, decltype(pack0), OutType>(f, packs, lv);
  }

  template<class FuncType, class PhaseType, class OutType>
  OutType makeLinkFunc(FuncType f,
      PhaseType phase0, std::string reg0, Eigen::VectorXi v0,
      PhaseType phase1, std::string reg1, Eigen::VectorXi v1,
      Eigen::VectorXi lv) {

      Eigen::VectorXi xtv0, opv0, spv0, xtv1, opv1, spv1;

      strto_PhaseRegionFlag(reg0);
      strto_PhaseRegionFlag(reg1);

      if (reg0 == "ODEParams")  opv0 = v0;
      else if (reg0 == "StaticParams")  spv0 = v0;
      else  xtv0 = v0;

      if (reg1 == "ODEParams")  opv1 = v1;
      else if (reg1 == "StaticParams")  spv1 = v1;
      else  xtv1 = v1;


      auto pack0 = std::tuple{ phase0,reg0,xtv0,opv0,spv0 };
      auto pack1 = std::tuple{ phase1,reg1,xtv1,opv1,spv1 };
      auto packs = std::vector{ pack0,pack1 };
      return this->makeLinkFunc<FuncType, decltype(pack0), OutType>(f, packs, lv);
  }
  template<class FuncType, class PhaseType, class OutType>
  OutType makeLinkFunc(FuncType f,
      PhaseType phase0, std::string reg0, Eigen::VectorXi v0,
      PhaseType phase1, std::string reg1, Eigen::VectorXi v1) {

      Eigen::VectorXi xtv0, opv0, spv0, xtv1, opv1, spv1;

      strto_PhaseRegionFlag(reg0);
      strto_PhaseRegionFlag(reg1);

      if (reg0 == "ODEParams")  opv0 = v0;
      else if (reg0 == "StaticParams")  spv0 = v0;
      else  xtv0 = v0;

      if (reg1 == "ODEParams")  opv1 = v1;
      else if (reg1 == "StaticParams")  spv1 = v1;
      else  xtv1 = v1;


      auto pack0 = std::tuple{ phase0,reg0,xtv0,opv0,spv0 };
      auto pack1 = std::tuple{ phase1,reg1,xtv1,opv1,spv1 };
      auto packs = std::vector{ pack0,pack1 };
      Eigen::VectorXi lv;
      lv.resize(0);
      return this->makeLinkFunc<FuncType, decltype(pack0), OutType>(f, packs, lv);

  }

  ////////////////////////////////////////////////
  int addLinkEqualCon(LinkConstraint lc) {
    this->resetTranscription();
    this->LinkEqualities.emplace_back(lc);
    int index = int(this->LinkEqualities.size()) - 1;
    this->LinkEqualities.back().StorageIndex = index;
    check_function_size(this->LinkEqualities.back(), "Link Equality Constraint");

    return index;
  }
  ////////////////////////////////////////////


  /////////////// THE NEW EQUALCON INTERFACE//////////////////////////////

  int addLinkEqualCon(VectorFunctionalX lc, std::vector<PhaseIndexPack> packs, VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhaseIndexPack, LinkConstraint>(lc, packs, lv);
      return this->addLinkEqualCon(Func);
  }
  int addLinkEqualCon(VectorFunctionalX lc, std::vector<PhaseIndexPackPtr> packs, VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhaseIndexPackPtr, LinkConstraint>(lc, packs, lv);
      return this->addLinkEqualCon(Func);
  }

  
  int addLinkEqualCon(VectorFunctionalX lc, 
      int phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc, 
               phase0,  reg0,  xtv0,  opv0,  spv0,
               phase1,  reg1,  xtv1,  opv1,  spv1,
               lv);
      return this->addLinkEqualCon(Func);
  }
  int addLinkEqualCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1,
          lv);
      return this->addLinkEqualCon(Func);
  }


  int addLinkEqualCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1);
      return this->addLinkEqualCon(Func);
  }
  int addLinkEqualCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1);
      return this->addLinkEqualCon(Func);
  }



  int addLinkEqualCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1,  Eigen::VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1,lv);
      return this->addLinkEqualCon(Func);
  }
  int addLinkEqualCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1,lv);
      return this->addLinkEqualCon(Func);
  }


  int addLinkEqualCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0, 
      int phase1, std::string reg1, Eigen::VectorXi xtv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0, 
          phase1, reg1, xtv1 );
      return this->addLinkEqualCon(Func);
  }
  int addLinkEqualCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1);
      return this->addLinkEqualCon(Func);
  }










  int addLinkEqualCon(VectorFunctionalX lc, 
      RegVec regs,
      std::vector<VectorXi> ptl, 
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc,
      std::vector<std::string> regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc, 
      LinkFlags regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc,
      std::string regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc,
      RegVec regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc,
      std::vector<std::string> regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc,
      LinkFlags regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }

  int addLinkEqualCon(VectorFunctionalX lc,
      std::string regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkEqualCon(LC);
  }


    ////////////////////////////////////////////////


  int addLinkEqualCon(VectorFunctionalX lc, RegVec regs,
                      std::vector<VectorXi> ptl, std::vector<VectorXi> xtuvs,
                      std::vector<VectorXi> lpvs) {
    std::vector<Eigen::VectorXi> empty;
    empty.resize(regs.size());
    auto LC = LinkConstraint(lc, regs, ptl, xtuvs, empty, empty, lpvs);
    return this->addLinkEqualCon(LC);
  }

 
  int addLinkEqualCon(VectorFunctionalX lc, RegVec regs,
      std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> lpvs) {
      return this->addLinkEqualCon(lc, regs, ptl_from_phases(ptl), xtuvs, lpvs);
  }


  int addLinkEqualCon(VectorFunctionalX lc, LinkFlags regs,
      std::vector<VectorXi> ptl, VectorXi xtuvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
      return this->addLinkEqualCon(LC);
  }
  int addLinkEqualCon(VectorFunctionalX lc, std::string regs,
      std::vector<VectorXi> ptl, VectorXi xtuvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
      return this->addLinkEqualCon(LC);
  }
 
  int addLinkEqualCon(VectorFunctionalX lc, LinkFlags regs,
      std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
      return this->addLinkEqualCon(lc, regs, ptl_from_phases(ptl), xtuvs);
  }
  int addLinkEqualCon(VectorFunctionalX lc, std::string regs,
      std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
      return this->addLinkEqualCon(lc, regs, ptl_from_phases(ptl), xtuvs);
  }

  /////////////////////////////////////////////////////////////

  


  /////////////////////////////////////////////////////////////

  int addLinkParamEqualCon(VectorFunctionalX lc, std::vector<VectorXi> lpvs) {
    std::vector<Eigen::VectorXi> empty;
    return this->addLinkEqualCon(LinkConstraint(
        lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs));
  }
  int addLinkParamEqualCon(VectorFunctionalX lc, VectorXi lpvs) {
    std::vector<Eigen::VectorXi> lpvss;
    lpvss.push_back(lpvs);
    return this->addLinkParamEqualCon(lc, lpvss);
  }

  int addForwardLinkEqualCon(int iphase, int fphase, VectorXi vars) {
    if (iphase < 0) iphase = (this->phases.size() + iphase);
    if (fphase < 0) fphase = (this->phases.size() + fphase);

    std::vector<Eigen::VectorXi> PTL;
    for (int i = iphase; i < fphase; i++) {
      VectorXi pl(2);
      pl[0] = i;
      pl[1] = i + 1;
      PTL.push_back(pl);
    }
    std::vector<Eigen::VectorXi> xtv(2);
    xtv[0] = vars;
    xtv[1] = vars;

    auto args = Arguments<-1>(2 * vars.size());
    auto func = args.head<-1>(vars.size()) - args.tail<-1>(vars.size());

    return this->addLinkEqualCon(
        LinkConstraint(func, LinkFlags::BackToFront, PTL, xtv));
  }

  
  int addForwardLinkEqualCon(PhasePtr iphase, PhasePtr fphase, VectorXi vars) {
      return this->addForwardLinkEqualCon(getPhaseNum(iphase), getPhaseNum(fphase), vars);
  }
  /////////////////////////////////////////////////////////////////////////


  int addDirectLinkEqualCon(LinkFlags LinkFlag, int iphase, VectorXi v1,
                            int fphase, VectorXi v2) {
    if (iphase < 0) iphase = (this->phases.size() + iphase);
    if (fphase < 0) fphase = (this->phases.size() + fphase);

    std::vector<Eigen::VectorXi> PTL;
    VectorXi pl(2);
    pl[0] = iphase;
    pl[1] = fphase;
    PTL.push_back(pl);

    std::vector<Eigen::VectorXi> xtv(2);
    xtv[0] = v1;
    xtv[1] = v2;

    auto args = Arguments<-1>(2 * v1.size());
    auto func = args.head<-1>(v1.size()) - args.tail<-1>(v2.size());
    return this->addLinkEqualCon(LinkConstraint(func, LinkFlag, PTL, xtv));
  }

 

  int addDirectLinkEqualCon(VectorFunctionalX lc, 
      int iphase, PhaseRegionFlags f1, VectorXi v1,
      int fphase, PhaseRegionFlags f2, VectorXi v2) {
      if (iphase < 0) iphase = (this->phases.size() + iphase);
      if (fphase < 0) fphase = (this->phases.size() + fphase);

      std::vector<Eigen::VectorXi> PTL;
      VectorXi pl(2);
      pl[0] = iphase;
      pl[1] = fphase;
      PTL.push_back(pl);

      Eigen::Matrix<PhaseRegionFlags, -1, 1> RegFlags(2);
      RegFlags[0] = f1;
      RegFlags[1] = f2;


      std::vector<Eigen::VectorXi> xtv(2);
      std::vector<Eigen::VectorXi> opv(2);
      std::vector<Eigen::VectorXi> spv(2);

      if (f1 == ODEParams)  opv[0] = v1;
      else if (f1 == StaticParams)  spv[0] = v1;
      else  xtv[0] = v1;

      if (f2 == ODEParams)  opv[1] = v2;
      else if (f2 == StaticParams)  spv[1] = v2;
      else  xtv[1] = v2;

      std::vector<Eigen::VectorXi> lv(1);

      return this->addLinkEqualCon(LinkConstraint(lc, RegFlags, PTL, xtv,opv,spv,lv));
  }

  int addDirectLinkEqualCon(VectorFunctionalX lc, int iphase, std::string f1, VectorXi v1,
      int fphase, std::string f2, VectorXi v2) {
      return this->addDirectLinkEqualCon(lc, iphase, strto_PhaseRegionFlag(f1), v1,
          fphase, strto_PhaseRegionFlag(f2), v2);
  }

  int addDirectLinkEqualCon(int iphase, PhaseRegionFlags f1, VectorXi v1,
      int fphase, PhaseRegionFlags f2, VectorXi v2) {
      auto args = Arguments<-1>(2 * v1.size());
      auto func = args.head<-1>(v1.size()) - args.tail<-1>(v2.size());
      return this->addDirectLinkEqualCon(func, iphase, f1, v1, fphase, f2, v2);
  }

  int addDirectLinkEqualCon(int iphase, std::string f1, VectorXi v1,
      int fphase, std::string f2, VectorXi v2) {
      return this->addDirectLinkEqualCon(iphase, strto_PhaseRegionFlag(f1), v1,
          fphase, strto_PhaseRegionFlag(f2), v2);
  }

  int addDirectLinkEqualCon(PhasePtr iphase, PhaseRegionFlags f1, VectorXi v1,
      PhasePtr fphase, PhaseRegionFlags f2, VectorXi v2) {
      return this->addDirectLinkEqualCon(getPhaseNum(iphase), f1, v1, getPhaseNum(fphase), f2, v2);
  }

  int addDirectLinkEqualCon(PhasePtr iphase, std::string f1, VectorXi v1,
      PhasePtr fphase, std::string f2, VectorXi v2) {
      return this->addDirectLinkEqualCon(getPhaseNum(iphase), f1, v1, getPhaseNum(fphase), f2, v2);
  }

  int addDirectLinkEqualCon(VectorFunctionalX lc, PhasePtr iphase, PhaseRegionFlags f1, VectorXi v1,
      PhasePtr fphase, PhaseRegionFlags f2, VectorXi v2) {
      return this->addDirectLinkEqualCon(lc,getPhaseNum(iphase), f1, v1, getPhaseNum(fphase), f2, v2);
  }
  int addDirectLinkEqualCon(VectorFunctionalX lc, PhasePtr iphase, std::string  f1, VectorXi v1,
      PhasePtr fphase, std::string f2, VectorXi v2) {
      return this->addDirectLinkEqualCon(lc, getPhaseNum(iphase), f1, v1, getPhaseNum(fphase), f2, v2);
  }

  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////

  int addLinkInequalCon(LinkConstraint lc) {
    this->resetTranscription();
    this->LinkInequalities.emplace_back(lc);
    int index = int(this->LinkInequalities.size()) - 1;
    this->LinkInequalities.back().StorageIndex = index;
    check_function_size(this->LinkInequalities.back(), "Link Inequality Constraint");

    return index;
  }


  ////////////////////////////////////////////

  /////////////// THE NEW INEQUALCON INTERFACE//////////////////////////////

  int addLinkInequalCon(VectorFunctionalX lc, std::vector<PhaseIndexPack> packs, VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhaseIndexPack, LinkConstraint>(lc, packs, lv);
      return this->addLinkInequalCon(Func);
  }
  int addLinkInequalCon(VectorFunctionalX lc, std::vector<PhaseIndexPackPtr> packs, VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhaseIndexPackPtr, LinkConstraint>(lc, packs, lv);
      return this->addLinkInequalCon(Func);
  }


  int addLinkInequalCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1,
          lv);
      return this->addLinkInequalCon(Func);
  }
  int addLinkInequalCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1,
          lv);
      return this->addLinkInequalCon(Func);
  }


  int addLinkInequalCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1);
      return this->addLinkInequalCon(Func);
  }
  int addLinkInequalCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1);
      return this->addLinkInequalCon(Func);
  }



  int addLinkInequalCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1, lv);
      return this->addLinkInequalCon(Func);
  }
  int addLinkInequalCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi lv) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1, lv);
      return this->addLinkInequalCon(Func);
  }


  int addLinkInequalCon(VectorFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, int, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1);
      return this->addLinkInequalCon(Func);
  }
  int addLinkInequalCon(VectorFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1) {
      auto Func = this->makeLinkFunc< VectorFunctionalX, PhasePtr, LinkConstraint>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1);
      return this->addLinkInequalCon(Func);
  }

  
  //////////////////////////
  

  int addLinkInequalCon(VectorFunctionalX lc,
      RegVec regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      std::vector<std::string> regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      LinkFlags regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      std::string regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      RegVec regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      std::vector<std::string> regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      LinkFlags regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc,
      std::string regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkConstraint(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkInequalCon(LC);
  }


  ////////////////////////////////////////////////



  int addLinkInequalCon(VectorFunctionalX lc, LinkFlags regs,
      std::vector<VectorXi> ptl, VectorXi xtuvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc, LinkFlags regs,
      std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
      return this->addLinkInequalCon(lc, regs, ptl_from_phases(ptl), xtuvs);
  }
  
  int addLinkInequalCon(VectorFunctionalX lc, std::string regs,
      std::vector<VectorXi> ptl, VectorXi xtuvs) {
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs);
      return this->addLinkInequalCon(LC);
  }

  int addLinkInequalCon(VectorFunctionalX lc, std::string regs,
      std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
      return this->addLinkInequalCon(lc, regs, ptl_from_phases(ptl), xtuvs);
  }


  int addLinkInequalCon(VectorFunctionalX lc, RegVec regs,
      std::vector<VectorXi> ptl, std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> lpvs) {
      std::vector<Eigen::VectorXi> empty;
      empty.resize(regs.size());
      auto LC = LinkConstraint(lc, regs, ptl, xtuvs, empty, empty, lpvs);
      return this->addLinkInequalCon(LC);
  }
  
  int addLinkInequalCon(VectorFunctionalX lc, RegVec regs,
      std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> lpvs) {
      return this->addLinkInequalCon(lc, regs, ptl_from_phases(ptl), xtuvs, lpvs);
  }


  int addLinkParamInequalCon(VectorFunctionalX lc, std::vector<VectorXi> lpvs) {
    std::vector<Eigen::VectorXi> empty;
    return this->addLinkInequalCon(LinkConstraint(
        lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs));
  }
  int addLinkParamInequalCon(VectorFunctionalX lc, VectorXi lpvs) {
    std::vector<Eigen::VectorXi> lpvss;
    lpvss.push_back(lpvs);
    return this->addLinkParamInequalCon(lc, lpvss);
  }
  ////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////

  int addLinkObjective(LinkObjective lc) {
    this->resetTranscription();
    this->LinkObjectives.emplace_back(lc);
    int index = int(this->LinkObjectives.size()) - 1;
    this->LinkObjectives.back().StorageIndex = index;
    check_function_size(this->LinkObjectives.back(), "Link Objective");
    return index;
  }
  /////////////// THE NEW INEQUALCON INTERFACE//////////////////////////////

  int addLinkObjective(ScalarFunctionalX lc, std::vector<PhaseIndexPack> packs, VectorXi lv) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, PhaseIndexPack, LinkObjective>(lc, packs, lv);
      return this->addLinkObjective(Func);
  }
  int addLinkObjective(ScalarFunctionalX lc, std::vector<PhaseIndexPackPtr> packs, VectorXi lv) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, PhaseIndexPackPtr, LinkObjective>(lc, packs, lv);
      return this->addLinkObjective(Func);
  }


  int addLinkObjective(ScalarFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, int, LinkObjective>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1,
          lv);
      return this->addLinkObjective(Func);
  }
  int addLinkObjective(ScalarFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1,
      VectorXi lv) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, PhasePtr, LinkObjective>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1,
          lv);
      return this->addLinkObjective(Func);
  }


  int addLinkObjective(ScalarFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, int, LinkObjective>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1);
      return this->addLinkObjective(Func);
  }
  int addLinkObjective(ScalarFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0, Eigen::VectorXi opv0, Eigen::VectorXi spv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi opv1, Eigen::VectorXi spv1) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, PhasePtr, LinkObjective>(lc,
          phase0, reg0, xtv0, opv0, spv0,
          phase1, reg1, xtv1, opv1, spv1);
      return this->addLinkObjective(Func);
  }



  int addLinkObjective(ScalarFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi lv) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, int, LinkObjective>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1, lv);
      return this->addLinkObjective(Func);
  }
  int addLinkObjective(ScalarFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1, Eigen::VectorXi lv) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, PhasePtr, LinkObjective>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1, lv);
      return this->addLinkObjective(Func);
  }


  int addLinkObjective(ScalarFunctionalX lc,
      int phase0, std::string reg0, Eigen::VectorXi xtv0,
      int phase1, std::string reg1, Eigen::VectorXi xtv1) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, int, LinkObjective>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1);
      return this->addLinkObjective(Func);
  }
  int addLinkObjective(ScalarFunctionalX lc,
      PhasePtr phase0, std::string reg0, Eigen::VectorXi xtv0,
      PhasePtr phase1, std::string reg1, Eigen::VectorXi xtv1) {
      auto Func = this->makeLinkFunc< ScalarFunctionalX, PhasePtr, LinkObjective>(lc,
          phase0, reg0, xtv0,
          phase1, reg1, xtv1);
      return this->addLinkObjective(Func);
  }


  //////////////////////////








  int addLinkObjective(ScalarFunctionalX lc,
      RegVec regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      std::vector<std::string> regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      LinkFlags regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      std::string regs,
      std::vector<VectorXi> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl, xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      RegVec regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      std::vector<std::string> regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      LinkFlags regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc,
      std::string regs,
      std::vector<std::vector<PhasePtr>> ptl,
      std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> opvs,
      std::vector<VectorXi> spvs,
      std::vector<VectorXi> lpvs) {
      auto LC = LinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, opvs, spvs, lpvs);
      return this->addLinkObjective(LC);
  }


  ////////////////////////////////////////////////


  int addLinkObjective(ScalarFunctionalX lc, LinkFlags regs,
      std::vector<VectorXi> ptl, VectorXi xtuvs) {
      auto LC = LinkObjective(lc, regs, ptl, xtuvs);
      return this->addLinkObjective(LC);
  }
  
  int addLinkObjective(ScalarFunctionalX lc, LinkFlags regs,
      std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
      return this->addLinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs);
  }

  int addLinkObjective(ScalarFunctionalX lc, std::string regs,
      std::vector<VectorXi> ptl, VectorXi xtuvs) {
      auto LC = LinkObjective(lc, regs, ptl, xtuvs);
      return this->addLinkObjective(LC);
  }

  int addLinkObjective(ScalarFunctionalX lc, std::string regs,
      std::vector<std::vector<PhasePtr>> ptl, VectorXi xtuvs) {
      return this->addLinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs);
  }

  int addLinkObjective(ScalarFunctionalX lc, RegVec regs,
      std::vector<VectorXi> ptl, std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> lpvs) {
      std::vector<Eigen::VectorXi> empty;
      empty.resize(regs.size());
      auto LC = LinkObjective(lc, regs, ptl, xtuvs, empty, empty, lpvs);
      return this->addLinkObjective(LC);
  }
  
  int addLinkObjective(ScalarFunctionalX lc, RegVec regs,
      std::vector<std::vector<PhasePtr>> ptl, std::vector<VectorXi> xtuvs,
      std::vector<VectorXi> lpvs) {
      return this->addLinkObjective(lc, regs, ptl_from_phases(ptl), xtuvs, lpvs);
  }


  int addLinkParamObjective(ScalarFunctionalX lc, std::vector<VectorXi> lpvs) {
    std::vector<Eigen::VectorXi> empty;
    return this->addLinkObjective(LinkObjective(
        lc, LinkFlags::LinkParams, empty, empty, empty, empty, lpvs));
  }
  int addLinkParamObjective(ScalarFunctionalX lc, VectorXi lpvs) {
    std::vector<Eigen::VectorXi> lpvss;
    lpvss.push_back(lpvs);
    return this->addLinkParamObjective(lc, lpvss);
  }

  ///////////////////////////////////////////////////

  void removeLinkEqualCon(int ith) {
    this->resetTranscription();
    if (ith < 0) ith = (this->LinkEqualities.size() + ith);
    this->LinkEqualities.erase(this->LinkEqualities.begin() + ith);
  }
  void removeLinkInequalCon(int ith) {
    this->resetTranscription();
    if (ith < 0) ith = (this->LinkInequalities.size() + ith);
    this->LinkInequalities.erase(this->LinkInequalities.begin() + ith);
  }
  void removeLinkObjective(int ith) {
    this->resetTranscription();
    if (ith < 0) ith = (this->LinkObjectives.size() + ith);
    this->LinkObjectives.erase(this->LinkObjectives.begin() + ith);
  }

  ///////////////////////////////////////////////////
  void checkTranscriptions() {
    for (int i = 0; i < this->phases.size(); i++) {
      if (this->phases[i]->doTranscription) {
        this->doTranscription = true;
      }
    }
  }

  void transcribe_phases();


  void check_dupilcate_phases() {
      for (int i = 0; i<this->phases.size(); i++) {
          for (int j = 0;j< this->phases.size(); j++) {
              if (j != i) {

                  if (this->phase_names[i] == this->phase_names[j]) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "OptimalControlProblem contains Two phases with identical names\n");
                      throw std::invalid_argument("");
                  }

                  if (this->phases[i].get() == this->phases[j].get()) {

                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Same phase detected more than once in optimal control problem. \n");
                      throw std::invalid_argument("");
                  }
              }
          }
      }
  }

  void check_functions();

  template<class T>
  void check_function_size(const T & func, std::string ftype) {
      int irows = func.Func.IRows();
      switch (func.LinkFlag) {
      case BackToFront:
      case FrontToBack:
      case FrontToFront:
      case BackToBack:
      case ParamsToParams:
      case LinkParams:
      case PathToPath:
      case ReadRegions: {

          if (func.LinkParams.size() != func.PhasesTolink.size() && func.PhasesTolink.size()>0) {
              fmt::print(fmt::fg(fmt::color::red),
                  "Transcription Error!!!\n"
                  "LinkParam Vector Must be same size as PTL Vector "
                  "even if each element of LinkParam Vector is empty (See Docs).");
              throw std::invalid_argument("");

          }

          if (func.LinkParams.size() > 0 && func.PhasesTolink.size() == 0) {
              for (int i = 0; i < func.LinkParams.size(); i++) {
                  int isize = func.LinkParams[i].size();
                  if (irows != isize) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Input size of {0:} (IRows = {1:}) does not match that implied by indexing parameters (IRows = {2:}).\n",
                          ftype, irows, isize);
                      throw std::invalid_argument("");
                  }
              }
          }
          else {
              for (int i = 0; i < func.PhasesTolink.size(); i++) {
                  int isize = func.LinkParams[i].size();

                  if (func.PhasesTolink[i].size() != func.XtUVars.size()) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Size of PTL vector element must equal size of Phase State Variables Vector");
                      throw std::invalid_argument("");
                  }
                  if (func.PhasesTolink[i].size() != func.OPVars.size()) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Size of PTL vector element must equal size of Phase ODEParam Variables Vector");
                      throw std::invalid_argument("");
                  }
                  if (func.PhasesTolink[i].size() != func.SPVars.size()) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Size of PTL vector element must equal size of Phase StaticParam Variables Vector");
                      throw std::invalid_argument("");
                  }
                  if (func.PhasesTolink[i].size() != func.PhaseRegFlags.size()) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Size of PTL vector element must equal size of Phase Region Flag Vector");
                      throw std::invalid_argument("");
                  }
                  for (int j = 0; j < func.PhasesTolink[i].size(); j++) {
                      auto flag = func.PhaseRegFlags[j];
                      int xmult = 1;
                      switch (flag) {
                      case PhaseRegionFlags::Front:
                      case PhaseRegionFlags::Back:
                      case PhaseRegionFlags::Path:
                      case PhaseRegionFlags::ODEParams:
                      case PhaseRegionFlags::StaticParams:
                      case PhaseRegionFlags::Params: xmult = 1; break;
                      case PhaseRegionFlags::FrontandBack:
                      case PhaseRegionFlags::BackandFront:xmult = 2; break;
                      default: {

                          fmt::print(fmt::fg(fmt::color::red),
                              "Transcription Error!!!\n"
                              "Invalid Phase Region requested in link function\n"
                              "Only the following regions are supported\n"\
                              "    Front, Back, Path,ODEParams,StaticParams, Params, FrontandBack, BackAndFront\n");

                          throw std::invalid_argument("");
                          break;
                      }
                      }
                      isize += func.XtUVars[j].size() * xmult + func.OPVars[j].size() + func.SPVars[j].size();
                  }
                  if (irows != isize) {
                      fmt::print(fmt::fg(fmt::color::red),
                          "Transcription Error!!!\n"
                          "Input size of {0:} (IRows = {1:}) does not match that implied by indexing parameters (IRows = {2:}).\n",
                          ftype, irows, isize);
                      throw std::invalid_argument("");
                  }
              }
          }
          
          break;
      }
      default: {
          break;
      }
      }
  }

  void transcribe_links();

  void transcribe(bool showstats, bool showfuns);

  void transcribe() { this->transcribe(false, false); }

  void jet_initialize() {
      this->setThreads(1, 1);
      this->optimizer->PrintLevel = 10;
      this->PrintMeshInfo = false;
      this->transcribe();
  }
  void jet_release() {
      this->optimizer->release();
      this->initThreads();
      this->optimizer->PrintLevel = 0;
      this->PrintMeshInfo = true;
      this->nlp = std::shared_ptr<NonLinearProgram>();
      for (auto& phase : this->phases) phase->jet_release();
      this->resetTranscription();
  }


  

  ////////////////////////////////////////////////////////////
  protected:
  void initMeshs() {
      this->MeshConverged = false;
      for (auto& phase : this->phases) {
          if (phase->AdaptiveMesh) {
              phase->initMeshRefinement();
          }
      }
  }

  bool checkMeshs(bool printinfo) {
      this->MeshConverged = true;
      for (auto& phase : this->phases) {
          if (phase->AdaptiveMesh) {
              if (!phase->checkMesh()) MeshConverged = false;
          }
      }
     
      return this->MeshConverged;
  }
  void updateMeshs(bool printinfo) {
      for (auto& phase : this->phases) {
          if (phase->AdaptiveMesh) {
              if (!phase->MeshConverged) {
                  phase->updateMesh();
              }
          }
          
      }
  }

  void printMeshs(int iter) {
      MeshIterateInfo::print_header(iter);
      for (int i = 0; i < this->phases.size(); i++) {
          if (this->phases[i]->AdaptiveMesh) {
              this->phases[i]->MeshIters.back().print(i);
          }
      }
  }

  std::array<Eigen::MatrixXi, 2> make_link_Vindex_Cindex(
      LinkFlags Reg, const Eigen::Matrix<PhaseRegionFlags, -1, 1>& PhaseRegs,
      const std::vector<Eigen::VectorXi>& PTL,
      const std::vector<Eigen::VectorXi>& xtv,
      const std::vector<Eigen::VectorXi>& opv,
      const std::vector<Eigen::VectorXi>& spv,
      const std::vector<Eigen::VectorXi>& lv, int orows, int& NextCLoc) const;


  PSIOPT::ConvergenceFlags psipot_call_impl(std::string mode);

  
  PSIOPT::ConvergenceFlags ocp_call_impl(std::string mode);

  VectorXd makeSolverInput() const {
      VectorXd Vars(this->numProbVars);

      for (int i = 0; i < this->phases.size(); i++) {
          int Start = 0;
          if (i > 0) Start = this->numPhaseVars.segment(0, i).sum();
          Vars.segment(Start, this->numPhaseVars[i]) =
              this->phases[i]->makeSolverInput();
      }
      Vars.tail(this->numLinkParams) = this->ActiveLinkParams;

      return Vars;
  }

  void collectSolverOutput(const VectorXd& Vars) {
      for (int i = 0; i < this->phases.size(); i++) {
          int Start = 0;
          if (i > 0) Start = this->numPhaseVars.segment(0, i).sum();
          this->phases[i]->collectSolverOutput(
              Vars.segment(Start, this->numPhaseVars[i]));
      }
      this->ActiveLinkParams = Vars.tail(this->numLinkParams);
  }
  void collectSolverMultipliers(const VectorXd& EM, const VectorXd& IM) {
      this->MultipliersLoaded = true;
      for (int i = 0; i < this->phases.size(); i++) {
          int EStart = 0;
          if (i > 0) EStart = this->numPhaseEqCons.segment(0, i).sum();
          int IStart = 0;
          if (i > 0) IStart = this->numPhaseIqCons.segment(0, i).sum();
          this->phases[i]->collectSolverMultipliers(
              EM.segment(EStart, this->numPhaseEqCons[i]),
              IM.segment(IStart, this->numPhaseIqCons[i]));
      }
      this->ActiveEqLmults = EM.tail(this->numLinkEqCons);
      this->ActiveIqLmults = IM.tail(this->numLinkIqCons);
  }



 public:
  PSIOPT::ConvergenceFlags solve() {
      return ocp_call_impl("solve");
  }
  PSIOPT::ConvergenceFlags optimize() {
      return ocp_call_impl("optimize");
  }
  PSIOPT::ConvergenceFlags solve_optimize() {
      return ocp_call_impl("solve_optimize");
  }
  PSIOPT::ConvergenceFlags solve_optimize_solve() {
      return ocp_call_impl("solve_optimize_solve");
  }
  PSIOPT::ConvergenceFlags optimize_solve() {
      return ocp_call_impl("optimize_solve");
  }



  void print_stats(bool showfuns);

 

  
  static void Build(py::module& m);
};

}  // namespace ASSET
