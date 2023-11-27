#include "OptimalControlProblem.h"
#include "OptimalControlProblem.h"
#include "OptimalControlProblem.h"
#include "AutoScalingUtils.h"

#include "PyDocString/OptimalControl/OptimalControlProblem_doc.h"

Eigen::VectorXd ASSET::OptimalControlProblem::get_input_scale(LinkFlags lflag, 
    Eigen::Vector<PhaseRegionFlags, -1> regs,
    std::vector<VectorXi> phases_to_link, 
    std::vector<VectorXi> XtUVars, 
    std::vector<VectorXi> OPVars, 
    std::vector<VectorXi> SPVars, 
    std::vector<VectorXi> LVars)
{
    

    std::vector<VectorXd> scales;
    int size = 0;

    if (phases_to_link.size() > 0) {

        for (int i = 0; i < phases_to_link[0].size(); i++) {

            int pnum = phases_to_link[0][i];
            auto flag = regs[i];
            auto XtUV = XtUVars[i];
            auto OPV = OPVars[i];
            auto SPV = SPVars[i];

            VectorXd scale = this->phases[pnum]->get_input_scale(flag, XtUV, OPV, SPV);
            scales.push_back(scale);
            size += scale.size();
        }
    }

    if (LVars.size() > 0) {

        VectorXd lscales(LVars[0].size());
        for (int i = 0; i < LVars[0].size(); i++) {
            lscales[i] = this->LPUnits[LVars[0][i]];
            size++;
        }
        scales.push_back(lscales);
    }

    VectorXd input_scales(size);

    int start = 0;

    for (int i = 0; i < scales.size(); i++) {
        size = scales[i].size();
        input_scales.segment(start, size) = scales[i];
        start += size;
    }



    return input_scales;
}

std::vector<Eigen::VectorXd> ASSET::OptimalControlProblem::get_test_inputs(LinkFlags lflag, 
    Eigen::Vector<PhaseRegionFlags, -1> regs, 
    std::vector<VectorXi> phases_to_link,
    std::vector<VectorXi> XtUVars, 
    std::vector<VectorXi> OPVars, std::vector<VectorXi> SPVars, std::vector<VectorXi> LVars)
{

    int nappl = std::max(phases_to_link.size(), LVars.size());

    std::vector<Eigen::VectorXd> test_inputs;


    for (int j = 0; j < nappl; j++) {
        int size = 0;

        std::vector<VectorXd> inputs;


        for (int i = 0; i < phases_to_link[j].size(); i++) {

            int pnum = phases_to_link[j][i];
            auto flag = regs[i];
            auto XtUV = XtUVars[i];
            auto OPV = OPVars[i];
            auto SPV = SPVars[i];

            VectorXd input = this->phases[pnum]->get_test_inputs(flag, XtUV, OPV, SPV)[0];
            inputs.push_back(input);
            size += input.size();
        }

        VectorXd linput(LVars[j].size());
        for (int i = 0; i < LVars[j].size(); i++) {
            linput[i] = this->ActiveLinkParams[LVars[j][i]];
            size++;
        }
        inputs.push_back(linput);

        VectorXd test_input(size);

        int start = 0;

        for (int i = 0; i < inputs.size(); i++) {
            size = inputs[i].size();
            test_input.segment(start, size) = inputs[i];
            start += size;
        }

        test_inputs.push_back(test_input);
    }


    return test_inputs;
}

void ASSET::OptimalControlProblem::transcribe_phases() {


  if (this->phases.size() > 0) {

    this->numPhaseVars.resize(this->phases.size());
    this->numPhaseEqCons.resize(this->phases.size());
    this->numPhaseIqCons.resize(this->phases.size());

    for (int i = 0; i < this->phases.size(); i++) {
      this->phases[i]->Threads = this->Threads;
      this->phases[i]->initIndexing();
      this->numPhaseVars[i] = this->phases[i]->indexer.numPhaseVars;
    }

    this->phases[0]->transcribe_phase(0, 0, 0, this->nlp, 0);
    this->numPhaseEqCons[0] = this->phases[0]->indexer.numPhaseEqCons;
    this->numPhaseIqCons[0] = this->phases[0]->indexer.numPhaseIqCons;

    for (int i = 1; i < this->phases.size(); i++) {
      int Vstart = this->numPhaseVars.segment(0, i).sum();
      int Estart = this->numPhaseEqCons.segment(0, i).sum();
      int Istart = this->numPhaseIqCons.segment(0, i).sum();

      this->phases[i]->transcribe_phase(Vstart, Estart, Istart, this->nlp, i);
      this->numPhaseEqCons[i] = this->phases[i]->indexer.numPhaseEqCons;
      this->numPhaseIqCons[i] = this->phases[i]->indexer.numPhaseIqCons;
    }


  } else {

    this->numPhaseVars.resize(1);
    this->numPhaseEqCons.resize(1);
    this->numPhaseIqCons.resize(1);

    this->numPhaseVars[0] = 0;
    this->numPhaseEqCons[0] = 0;
    this->numPhaseIqCons[0] = 0;
  }
}

void ASSET::OptimalControlProblem::check_functions() {
  auto CheckFunc = [&](std::string type, auto& func) {
    for (int i = 0; i < func.PhasesTolink.size(); i++) {
      for (int j = 0; j < func.PhasesTolink[i].size(); j++) {
        int pnum = func.PhasesTolink[i][j];
        if (pnum >= this->phases.size() || pnum < 0) {
          fmt::print(fmt::fg(fmt::color::red),
                     "Transcription Error!!!\n"
                     "{0:} references non-existant phase:{1:}\n"
                     " Function Storage Index:{2:}\n"
                     " Function Name:{3:}\n",
                     type,
                     pnum,
                     func.StorageIndex,
                     func.Func.name());
          throw std::invalid_argument("");
        }

        if (func.XtUVars[j].size() > 0) {
          if (func.XtUVars[j].maxCoeff() >= this->Phase(pnum)->XtUPVars() || func.XtUVars[j].minCoeff() < 0) {

            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "{0:} function state variable indices out of bounds in phase:{1:}\n"
                       " Function Storage Index:{2:}\n"
                       " Function Name:{3:}\n",
                       type,
                       pnum,
                       func.StorageIndex,
                       func.Func.name());
            throw std::invalid_argument("");
          }
        }
        if (func.OPVars[j].size() > 0) {
          if (func.OPVars[j].maxCoeff() >= this->Phase(pnum)->PVars() || func.OPVars[j].minCoeff() < 0) {

            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "{0:} function ODE Param variable indices out of bounds in phase:{1:}\n"
                       " Function Storage Index:{2:}\n"
                       " Function Name:{3:}\n",
                       type,
                       pnum,
                       func.StorageIndex,
                       func.Func.name());
            throw std::invalid_argument("");
          }
        }
        if (func.SPVars[j].size() > 0) {
          if (func.SPVars[j].maxCoeff() >= this->Phase(pnum)->numStatParams
              || func.SPVars[j].minCoeff() < 0) {
            fmt::print(fmt::fg(fmt::color::red),
                       "Transcription Error!!!\n"
                       "{0:} function Static Param variable indices out of bounds in phase:{1:}\n"
                       " Function Storage Index:{2:}\n"
                       " Function Name:{3:}\n",
                       type,
                       pnum,
                       func.StorageIndex,
                       func.Func.name());
            throw std::invalid_argument("");
          }
        }
      }
    }

    for (int i = 0; i < func.LinkParams.size(); i++) {
      if (func.LinkParams[i].size() > 0) {


      if (func.LinkParams[i].maxCoeff() >= this->numLinkParams || func.LinkParams[i].minCoeff() < 0) {

          fmt::print(fmt::fg(fmt::color::red),
                     "Transcription Error!!!\n"
                     "{0:} function link parameter variable indices out of bounds\n"
                     " Function Storage Index:{1:}\n"
                     " Function Name:{2:}\n",
                     type,
                     func.StorageIndex,
                     func.Func.name());
          throw std::invalid_argument("");
        }

      }
    
    }
  };

  std::string eq = "Link Equality constraint";
  std::string iq = "Link Inequality constraint";
  std::string obj = "Link Objective";


  for (auto& [key,f]: this->LinkEqualities)
    CheckFunc(eq, f);
  for (auto& [key, f]: this->LinkInequalities)
    CheckFunc(iq, f);
  for (auto& [key, f]: this->LinkObjectives)
    CheckFunc(obj, f);
}

void ASSET::OptimalControlProblem::transcribe_links() {


  int NextEq = this->numPhaseEqCons.sum();
  int NextIq = this->numPhaseIqCons.sum();

  int LinkVarStart = this->numPhaseVars.sum();
  this->LinkParamLocs.resize(this->numLinkParams);
  for (int i = 0; i < this->numLinkParams; i++) {
    this->LinkParamLocs[i] = LinkVarStart + i;
  }

  this->StartObj = int(this->nlp->Objectives.size());
  this->StartEq = int(this->nlp->EqualityConstraints.size());
  this->StartIq = int(this->nlp->InequalityConstraints.size());
  this->numEqFuns = 0;
  this->numIqFuns = 0;
  this->numObjFuns = 0;
  for (auto& [key,Eq]: this->LinkEqualities) {
    auto VC = this->make_link_Vindex_Cindex(Eq.LinkFlag,
                                            Eq.PhaseRegFlags,
                                            Eq.PhasesTolink,
                                            Eq.XtUVars,
                                            Eq.OPVars,
                                            Eq.SPVars,
                                            Eq.LinkParams,
                                            Eq.Func.ORows(),
                                            NextEq);


    auto Func = Eq.Func;

    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(Eq.LinkFlag,
            Eq.PhaseRegFlags,
            Eq.PhasesTolink,
            Eq.XtUVars,
            Eq.OPVars,
            Eq.SPVars,
            Eq.LinkParams);
        VectorXd output_scales(Func.ORows());
        output_scales = Eq.OutputScales;
        Func = IOScaled<decltype(Func)>(Eq.Func, input_scales, output_scales);
    }

    this->nlp->EqualityConstraints.emplace_back(ConstraintFunction(Func, VC[0], VC[1]));
    Eq.GlobalIndex = this->nlp->EqualityConstraints.size() - 1;
    this->numEqFuns++;
  }
  for (auto& [key,Iq]: this->LinkInequalities) {
    auto VC = this->make_link_Vindex_Cindex(Iq.LinkFlag,
                                            Iq.PhaseRegFlags,
                                            Iq.PhasesTolink,
                                            Iq.XtUVars,
                                            Iq.OPVars,
                                            Iq.SPVars,
                                            Iq.LinkParams,
                                            Iq.Func.ORows(),
                                            NextIq);

    auto Func = Iq.Func;

    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(Iq.LinkFlag,
            Iq.PhaseRegFlags,
            Iq.PhasesTolink,
            Iq.XtUVars,
            Iq.OPVars,
            Iq.SPVars,
            Iq.LinkParams);
        VectorXd output_scales(Func.ORows());
        output_scales = Iq.OutputScales;
        Func = IOScaled<decltype(Func)>(Iq.Func, input_scales, output_scales);
    }


    this->nlp->InequalityConstraints.emplace_back(ConstraintFunction(Func, VC[0], VC[1]));
    Iq.GlobalIndex = this->nlp->InequalityConstraints.size() - 1;
    this->numIqFuns++;
  }
  for (auto& [key,Ob]: this->LinkObjectives) {
    int dummy = 0;
    auto VC = this->make_link_Vindex_Cindex(Ob.LinkFlag,
                                            Ob.PhaseRegFlags,
                                            Ob.PhasesTolink,
                                            Ob.XtUVars,
                                            Ob.OPVars,
                                            Ob.SPVars,
                                            Ob.LinkParams,
                                            Ob.Func.ORows(),
                                            dummy);

    auto Func = Ob.Func;

    if (this->AutoScaling) {
        VectorXd input_scales = this->get_input_scale(Ob.LinkFlag,
            Ob.PhaseRegFlags,
            Ob.PhasesTolink,
            Ob.XtUVars,
            Ob.OPVars,
            Ob.SPVars,
            Ob.LinkParams);
        VectorXd output_scales(Func.ORows());
        output_scales = Ob.OutputScales;
        Func = IOScaled<decltype(Func)>(Ob.Func, input_scales, output_scales);
    }


    this->nlp->Objectives.emplace_back(ObjectiveFunction(Func, VC[0]));
    Ob.GlobalIndex = this->nlp->Objectives.size() - 1;

    this->numObjFuns++;
  }

  this->numLinkEqCons = NextEq - this->numPhaseEqCons.sum();
  this->numLinkIqCons = NextIq - this->numPhaseIqCons.sum();
}

void ASSET::OptimalControlProblem::calc_auto_scales()
{
    auto calc_impl = [&](auto& funcmap) {
        for (auto& [key, func] : funcmap) {
            if (func.ScaleMode == "auto" || true) {
                VectorXd input_scales = this->get_input_scale(func.LinkFlag,
                    func.PhaseRegFlags,
                    func.PhasesTolink,
                    func.XtUVars,
                    func.OPVars,
                    func.SPVars,
                    func.LinkParams);
                std::vector<VectorXd> test_inputs = this->get_test_inputs(func.LinkFlag,
                    func.PhaseRegFlags,
                    func.PhasesTolink,
                    func.XtUVars,
                    func.OPVars,
                    func.SPVars,
                    func.LinkParams);
                VectorXd output_scales = calc_jacobian_row_scales(func.Func, input_scales, test_inputs, "norm", "mean");
                func.OutputScales = output_scales;
                fmt::print("##############\n");
                std::cout << output_scales << std::endl;
            }
            else {


            }

        }
    };
    calc_impl(this->LinkEqualities);
    calc_impl(this->LinkInequalities);
    calc_impl(this->LinkObjectives);

}

void ASSET::OptimalControlProblem::transcribe(bool showstats, bool showfuns) {

  this->nlp = std::make_shared<NonLinearProgram>(this->Threads);
  check_functions();

  if (this->AutoScaling) {
      this->calc_auto_scales();
  }

  this->transcribe_phases();
  this->transcribe_links();

  this->numProbVars = this->numPhaseVars.sum() + this->numLinkParams;
  this->numProbEqCons = this->numPhaseEqCons.sum() + this->numLinkEqCons;
  this->numProbIqCons = this->numPhaseIqCons.sum() + this->numLinkIqCons;
  if (showstats)
    this->print_stats(showfuns);
  this->nlp->make_NLP(this->numProbVars, this->numProbEqCons, this->numProbIqCons);
  this->optimizer->setNLP(this->nlp);

  //////DO NOT GET RID OF THIS!!!!!!//
  this->doTranscription = false;
}

ASSET::PSIOPT::ConvergenceFlags ASSET::OptimalControlProblem::psipot_call_impl(std::string mode) {


  this->checkTranscriptions();
  if (this->doTranscription)
    this->transcribe();
  VectorXd Input = this->makeSolverInput();
  VectorXd Output;

  if (mode == "solve") {
    Output = this->optimizer->solve(Input);
  } else if (mode == "optimize") {
    Output = this->optimizer->optimize(Input);
  } else if (mode == "solve_optimize") {
    Output = this->optimizer->solve_optimize(Input);
  } else if (mode == "solve_optimize_solve") {
    Output = this->optimizer->solve_optimize_solve(Input);
  } else if (mode == "optimize_solve") {
    Output = this->optimizer->optimize_solve(Input);
  } else {
    throw std::invalid_argument("Unrecognized PSIOPT mode");
  }


  this->collectSolverOutput(Output);

  this->collectPostOptInfo(this->optimizer->LastEqCons,
                           this->optimizer->LastEqLmults,
                           this->optimizer->LastIqCons,
                           this->optimizer->LastIqLmults);


  return this->optimizer->ConvergeFlag;
}

ASSET::PSIOPT::ConvergenceFlags ASSET::OptimalControlProblem::ocp_call_impl(std::string mode) {
  if (this->PrintMeshInfo && this->AdaptiveMesh) {
    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
    fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
    fmt::print("\n");
  }

  Utils::Timer Runtimer;
  Runtimer.start();

  PSIOPT::ConvergenceFlags flag = this->psipot_call_impl(mode);

  std::string nextmode = mode;
  if (this->SolveOnlyFirst) {
    if (nextmode.find(std::string("solve_")) != std::string::npos) {
      nextmode.erase(0, 6);
    }
  }

  if (this->AdaptiveMesh) {

    if (flag >= this->MeshAbortFlag) {
      if (this->PrintMeshInfo) {
        fmt::print(fmt::fg(fmt::color::red), "Mesh Iteration 0 Failed to Solve: Aborting\n");
      }
    } else {
      initMeshs();
      for (int i = 0; i < this->MaxMeshIters; i++) {
        if (checkMeshs(this->PrintMeshInfo)) {
          if (this->PrintMeshInfo) {
            this->printMeshs(i);
            fmt::print(fmt::fg(fmt::color::lime_green), "All Meshes Converged\n");
          }

          break;
        } else if (i == this->MaxMeshIters - 1) {
          if (this->PrintMeshInfo) {
            this->printMeshs(i);
            fmt::print(fmt::fg(fmt::color::red), "All Meshes Not Converged\n");
          }
          break;
        } else {
          updateMeshs(this->PrintMeshInfo);
          if (this->PrintMeshInfo)
            this->printMeshs(i);
        }
        flag = this->psipot_call_impl(nextmode);
        if (flag >= this->MeshAbortFlag) {
          if (this->PrintMeshInfo) {
            fmt::print(fmt::fg(fmt::color::red), "Mesh Iteration {0:} Failed to Solve: Aborting\n", i + 1);
          }
          break;
        }
      }
    }
  }

  if (this->PrintMeshInfo && this->AdaptiveMesh) {

    Runtimer.stop();
    double tseconds = double(Runtimer.count<std::chrono::microseconds>()) / 1000000;
    fmt::print("Total Time:");
    if (tseconds > 0.5) {
      fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.4f} s\n", tseconds);
    } else {
      fmt::print(fmt::fg(fmt::color::cyan), "{0:>10.2f} ms\n", tseconds * 1000);
    }


    fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
    fmt::print(": ");
    fmt::print(fmt::fg(fmt::color::royal_blue), "Adaptive Mesh Refinement");
    fmt::print("\n");
    fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
  }

  return flag;
}


void ASSET::OptimalControlProblem::print_stats(bool showfuns) {
  using std::cout;
  using std::endl;
  cout << "Problem Statistics" << endl << endl;

  cout << "# Variables:    " << numProbVars << endl;
  cout << "# EqualCons:    " << numProbEqCons << endl;
  cout << "# InEqualCons:  " << numProbIqCons << endl;
  cout << "# Phases:       " << this->phases.size() << endl << endl;

  for (int i = 0; i < this->phases.size(); i++) {
    cout << "____________________________________________________________" << endl << endl;
    cout << "Phase: " << i << " Statistics" << endl << endl;
    this->phases[i]->indexer.print_stats(showfuns);
    cout << "____________________________________________________________" << endl << endl;
  }

  cout << "____________________________________________________________" << endl << endl;
  cout << "Link Statistics" << endl << endl;
  cout << "# Link Params:    " << numLinkParams << endl;
  cout << "# Link EqualCons:    " << numLinkEqCons << endl;
  cout << "# Link InEqualCons:  " << numLinkIqCons << endl;

  cout << "____________________________________________________________" << endl << endl;

  if (showfuns) {
    cout << "Objective Functions" << endl << endl;
    cout << "____________________________________________________________" << endl << endl;
    for (int i = 0; i < this->numObjFuns; i++) {
      cout << "************************************************************" << endl << endl;
      this->nlp->Objectives[this->StartObj + i].print_data();
    }
    cout << "Equality Constraints" << endl << endl;
    cout << "____________________________________________________________" << endl << endl;
    for (int i = 0; i < this->numEqFuns; i++) {
      cout << "************************************************************" << endl << endl;
      this->nlp->EqualityConstraints[this->StartEq + i].print_data();
    }
    cout << "Inequality Constraints" << endl << endl;
    cout << "____________________________________________________________" << endl << endl;
    for (int i = 0; i < this->numIqFuns; i++) {
      cout << "************************************************************" << endl << endl;
      this->nlp->InequalityConstraints[this->StartIq + i].print_data();
    }
  }
}

std::array<Eigen::MatrixXi, 2> ASSET::OptimalControlProblem::make_link_Vindex_Cindex(
    LinkFlags Reg,
    const Eigen::Matrix<PhaseRegionFlags, -1, 1>& PhaseRegs,
    const std::vector<Eigen::VectorXi>& PTL,
    const std::vector<Eigen::VectorXi>& xtv,
    const std::vector<Eigen::VectorXi>& opv,
    const std::vector<Eigen::VectorXi>& spv,
    const std::vector<Eigen::VectorXi>& lv,
    int orows,
    int& NextCLoc) const {
  using std::cout;
  using std::endl;
  MatrixXi Vindex;
  MatrixXi Cindex;

  switch (Reg) {
    case LinkFlags::PathToPath: {

      int cols = 0;
      std::vector<MatrixXi> vtemps(PTL.size());
      std::vector<MatrixXi> vtemps2(PhaseRegs.size());

      int irows = 0;
      for (int i = 0; i < PTL.size(); i++) {
        int sz = 0;
        irows = 0;

        for (int j = 0; j < PhaseRegs.size(); j++) {
          auto VinTemp =
              this->phases[PTL[i][j]]->indexer.make_Vindex_Cindex(PhaseRegs[j], xtv[j], opv[j], spv[j], 1)[0];
          vtemps2[j] = VinTemp;
          irows += VinTemp.rows();
          if (j == 0)
            sz = VinTemp.cols();
          else {
            if (sz != VinTemp.cols()) {
              throw std::invalid_argument("Phases cannot be linked path to path");
            }
          }
        }
        cols += sz;
        vtemps[i].resize(irows, sz);

        int start = 0;
        for (int j = 0; j < PhaseRegs.size(); j++) {
          vtemps[i].middleRows(start, vtemps2[j].rows()) = vtemps2[j];
          start += vtemps2[j].rows();
        }
      }

      Vindex.resize(irows, cols);

      int start = 0;
      for (int i = 0; i < PTL.size(); i++) {
        Vindex.middleCols(start, vtemps[i].cols()) = vtemps[i];
        start += vtemps[i].cols();
      }


      Cindex.resize(orows, cols);
      for (int i = 0; i < cols; i++) {
        for (int j = 0; j < orows; j++) {
          Cindex(j, i) = NextCLoc;
          NextCLoc++;
        }
      }

      break;
    }
    case LinkFlags::ReadRegions: {
    }
    default: {
      int cols = std::max(PTL.size(), lv.size());
      Cindex.resize(orows, cols);

      for (int i = 0; i < cols; i++) {
        for (int j = 0; j < orows; j++) {
          Cindex(j, i) = NextCLoc;
          NextCLoc++;
        }
      }
      int irows = lv[0].size();
      for (int i = 0; i < PhaseRegs.size(); i++) {
        irows += xtv[i].size() + opv[i].size() + spv[i].size();
      }
      Vindex.resize(irows, cols);
      for (int i = 0; i < cols; i++) {
        int start = 0;
        for (int j = 0; j < PhaseRegs.size(); j++) {

          auto VinTemp =
              this->phases[PTL[i][j]]->indexer.make_Vindex_Cindex(PhaseRegs[j], xtv[j], opv[j], spv[j], 1)[0];
          Vindex.col(i).segment(start, VinTemp.rows()) = VinTemp;
          start += VinTemp.rows();
        }
        for (int j = 0; j < lv[i].size(); j++) {
          Vindex(start, i) = this->LinkParamLocs[lv[i][j]];
          start++;
        }
      }
      break;
    }
  }

  return std::array<Eigen::MatrixXi, 2> {Vindex, Cindex};
}

void ASSET::OptimalControlProblem::Build(py::module& m) {
  using namespace doc;
  auto obj =
      py::class_<OptimalControlProblem, std::shared_ptr<OptimalControlProblem>, OptimizationProblemBase>(
          m, "OptimalControlProblem");
  obj.def(py::init<>());


  ///////////////////// NEW INTERFACE ///////////////////////////

  obj.def("addLinkEqualConNEW",
      py::overload_cast<
      VectorFunctionalX,
      PhaseRefType,
      RegionType,
      VarIndexType,
      VarIndexType,
      VarIndexType,
      PhaseRefType,
      RegionType,
      VarIndexType,
      VarIndexType,
      VarIndexType,
      VarIndexType,
      ScaleType>(&OptimalControlProblem::addLinkEqualCon),
      py::arg("func"),
      py::arg("phase0"),
      py::arg("reg0"),
      py::arg("XtUVars0"),
      py::arg("OPVars0"),
      py::arg("SPVars0"),
      py::arg("phase1"),
      py::arg("reg1"),
      py::arg("XtUVars1"),
      py::arg("OPVars1"),
      py::arg("SPVars1"),
      py::arg("linkparams") = VectorXi(),
      py::arg("AutoScale") = std::string("auto")
  );



  obj.def("addLinkEqualConNEW",
      py::overload_cast<
      VectorFunctionalX ,
      PhaseRefType , 
      RegionType , 
      VarIndexType ,
      PhaseRefType , 
      RegionType , 
      VarIndexType ,
      VarIndexType ,
      ScaleType >(&OptimalControlProblem::addLinkEqualCon),
      py::arg("func"),
      py::arg("phase0"),
      py::arg("reg0"),
      py::arg("v0"),
      py::arg("phase1"),
      py::arg("reg1"),
      py::arg("v1"),
      py::arg("linkparams") = VectorXi(),
      py::arg("AutoScale") = std::string("auto")
      );

  obj.def("addForwardLinkEqualConNEW",
      py::overload_cast<
      PhaseRefType,
      PhaseRefType,
      VarIndexType,
      ScaleType>(&OptimalControlProblem::addForwardLinkEqualCon),
      py::arg("phase0"),
      py::arg("phase1"),
      py::arg("vars"),
      py::arg("AutoScale") = std::string("auto")
  );

  obj.def("addDirectLinkEqualConNEW",
      py::overload_cast<
      PhaseRefType,
      RegionType,
      VarIndexType,
      PhaseRefType,
      RegionType,
      VarIndexType,
      ScaleType>(&OptimalControlProblem::addDirectLinkEqualCon),
      py::arg("phase0"),
      py::arg("reg0"),
      py::arg("v0"),
      py::arg("phase1"),
      py::arg("reg1"),
      py::arg("v1"),
      py::arg("AutoScale") = std::string("auto")
  );
  obj.def("addLinkParamEqualConNEW",
      py::overload_cast<
      VectorFunctionalX, 
      std::vector<VectorXi>,ScaleType>(
          &OptimalControlProblem::addLinkParamEqualCon),
      py::arg("func"),
      py::arg("LinkParms"),
      py::arg("AutoScale") = std::string("auto")
      );
  obj.def("addLinkParamEqualConNEW",
      py::overload_cast<
      VectorFunctionalX,
      VectorXi, 
      ScaleType>(
          &OptimalControlProblem::addLinkParamEqualCon),
      py::arg("func"),
      py::arg("LinkParms"),
      py::arg("AutoScale") = std::string("auto")
  );

  //////////////////
  obj.def("addLinkEqualCon",
          py::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon1);

  ////////////////////////////

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            std::vector<std::string>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            LinkFlags,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            std::string,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

  /// ///
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            std::vector<std::string>,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            LinkFlags,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            std::string,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon));

  ////////////////////////////


  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon2);


  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<std::vector<std::shared_ptr<ODEPhaseBase>>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon2);


  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon2);
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon2);

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon2);
  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon),
          OptimalControlProblem_addLinkEqualCon2);


  obj.def("addForwardLinkEqualCon",
          py::overload_cast<int, int, VectorXi>(&OptimalControlProblem::addForwardLinkEqualCon),
          OptimalControlProblem_addForwardLinkEqualCon);


  obj.def("addForwardLinkEqualCon",
          py::overload_cast<PhasePtr, PhasePtr, VectorXi>(&OptimalControlProblem::addForwardLinkEqualCon),
          OptimalControlProblem_addForwardLinkEqualCon);


  obj.def("addDirectLinkEqualCon",
          py::overload_cast<LinkFlags, int, VectorXi, int, VectorXi>(
              &OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);

  obj.def(
      "addDirectLinkEqualCon",
      py::overload_cast<VectorFunctionalX, int, PhaseRegionFlags, VectorXi, int, PhaseRegionFlags, VectorXi>(
          &OptimalControlProblem::addDirectLinkEqualCon),
      OptimalControlProblem_addDirectLinkEqualCon);
  obj.def("addDirectLinkEqualCon",
          py::overload_cast<int, PhaseRegionFlags, VectorXi, int, PhaseRegionFlags, VectorXi>(
              &OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);

  obj.def("addDirectLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            PhaseRegionFlags,
                            VectorXi,
                            PhasePtr,
                            PhaseRegionFlags,
                            VectorXi>(&OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);

  obj.def("addDirectLinkEqualCon",
          py::overload_cast<PhasePtr, PhaseRegionFlags, VectorXi, PhasePtr, PhaseRegionFlags, VectorXi>(
              &OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);

  //
  obj.def("addDirectLinkEqualCon",
          py::overload_cast<VectorFunctionalX, int, std::string, VectorXi, int, std::string, VectorXi>(
              &OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);
  obj.def("addDirectLinkEqualCon",
          py::overload_cast<int, std::string, VectorXi, int, std::string, VectorXi>(
              &OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);

  obj.def(
      "addDirectLinkEqualCon",
      py::overload_cast<VectorFunctionalX, PhasePtr, std::string, VectorXi, PhasePtr, std::string, VectorXi>(
          &OptimalControlProblem::addDirectLinkEqualCon),
      OptimalControlProblem_addDirectLinkEqualCon);

  obj.def("addDirectLinkEqualCon",
          py::overload_cast<PhasePtr, std::string, VectorXi, PhasePtr, std::string, VectorXi>(
              &OptimalControlProblem::addDirectLinkEqualCon),
          OptimalControlProblem_addDirectLinkEqualCon);


  //////////////////////////////////////////////////
  //////////////////////////////////////////////////

  obj.def("addLinkInequalCon",
          py::overload_cast<LinkConstraint>(&OptimalControlProblem::addLinkInequalCon),
          OptimalControlProblem_addLinkInequalCon);

  ////////////////////////////

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            std::vector<std::string>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            LinkFlags,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            std::string,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

  /// ///
  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            std::vector<std::string>,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            LinkFlags,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));
  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            std::string,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));

  ////////////////////////////


  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon),
          OptimalControlProblem_addLinkInequalCon);


  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon),
          OptimalControlProblem_addLinkInequalCon);


  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon),
          OptimalControlProblem_addLinkInequalCon);


  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, std::string, std::vector<std::vector<PhasePtr>>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon),
          OptimalControlProblem_addLinkInequalCon);


  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));


  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            RegVec,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkInequalCon));


  ////////////////////////////////////////
  obj.def("addLinkObjective",
          py::overload_cast<LinkObjective>(&OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);


  ////////////////////////////

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            RegVec,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            std::vector<std::string>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            LinkFlags,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            std::string,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

  /// ///
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            RegVec,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            std::vector<std::string>,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            LinkFlags,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            std::string,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective));

  ////////////////////////////


  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<VectorXi>, VectorXi>(
              &OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, LinkFlags, std::vector<std::vector<PhasePtr>>, VectorXi>(
              &OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, std::string, std::vector<VectorXi>, VectorXi>(
              &OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);
  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, std::string, std::vector<std::vector<PhasePtr>>, VectorXi>(
              &OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);


  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            RegVec,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);


  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            RegVec,
                            std::vector<std::vector<PhasePtr>>,
                            std::vector<VectorXi>,
                            std::vector<VectorXi>>(&OptimalControlProblem::addLinkObjective),
          OptimalControlProblem_addLinkObjective);


  //////////////////////////////////////////////////////////////////////////////

  obj.def("addLinkParamEqualCon",
          py::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
              &OptimalControlProblem::addLinkParamEqualCon),
          OptimalControlProblem_addLinkParamEqualCon1);
  obj.def("addLinkParamEqualCon",
          py::overload_cast<VectorFunctionalX, VectorXi>(&OptimalControlProblem::addLinkParamEqualCon),
          OptimalControlProblem_addLinkParamEqualCon2);
  obj.def("addLinkParamInequalCon",
          py::overload_cast<VectorFunctionalX, std::vector<VectorXi>>(
              &OptimalControlProblem::addLinkParamInequalCon),
          OptimalControlProblem_addLinkParamInequalCon1);
  obj.def("addLinkParamInequalCon",
          py::overload_cast<VectorFunctionalX, VectorXi>(&OptimalControlProblem::addLinkParamInequalCon),
          OptimalControlProblem_addLinkParamInequalCon2);
  obj.def("addLinkParamObjective",
          py::overload_cast<ScalarFunctionalX, std::vector<VectorXi>>(
              &OptimalControlProblem::addLinkParamObjective),
          OptimalControlProblem_addLinkParamObjective1);
  obj.def("addLinkParamObjective",
          py::overload_cast<ScalarFunctionalX, VectorXi>(&OptimalControlProblem::addLinkParamObjective),
          OptimalControlProblem_addLinkParamObjective2);

  //////////////////////////////////////////////////////////////////////////////

  obj.def("removeLinkEqualCon",
          &OptimalControlProblem::removeLinkEqualCon,
          OptimalControlProblem_removeLinkEqualCon);
  obj.def("removeLinkInequalCon",
          &OptimalControlProblem::removeLinkInequalCon,
          OptimalControlProblem_removeLinkEqualCon);
  obj.def("removeLinkObjective",
          &OptimalControlProblem::removeLinkObjective,
          OptimalControlProblem_removeLinkObjective);

  obj.def("addPhase",
          py::overload_cast<PhasePtr>(&OptimalControlProblem::addPhase),
          OptimalControlProblem_addPhase);


  obj.def("addPhases", &OptimalControlProblem::addPhases);

  obj.def("getPhaseNum", py::overload_cast<PhasePtr>(&OptimalControlProblem::getPhaseNum));


  obj.def("removePhase", &OptimalControlProblem::removePhase, OptimalControlProblem_removePhase);
  obj.def("Phase", &OptimalControlProblem::Phase, OptimalControlProblem_Phase);
  obj.def("setLinkParams", &OptimalControlProblem::setLinkParams, OptimalControlProblem_setLinkParams);
  obj.def(
      "returnLinkParams", &OptimalControlProblem::returnLinkParams, OptimalControlProblem_returnLinkParams);


  obj.def("transcribe",
          py::overload_cast<bool, bool>(&OptimalControlProblem::transcribe),
          OptimalControlProblem_transcribe);

  obj.def_readonly("Phases", &OptimalControlProblem::phases, OptimalControlProblem_Phases);


  ///////////////////////////////////////////////////////////////
  /////////////// New Link Interface/////////////////////////////

  ///
  /// EqualCons

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, std::vector<PhaseIndexPack>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, std::vector<PhaseIndexPackPtr>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon));


  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX, std::vector<PhaseIndexPack>, VectorXi>(
              &OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkEqualCon));

  obj.def("addLinkEqualCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkEqualCon));


  ///
  /// InequalCons

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, std::vector<PhaseIndexPack>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, std::vector<PhaseIndexPackPtr>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX, std::vector<PhaseIndexPack>, VectorXi>(
              &OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkInequalCon));

  obj.def("addLinkInequalCon",
          py::overload_cast<VectorFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkInequalCon));


  ///
  /// Objectives

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, std::vector<PhaseIndexPack>, VectorXi>(
              &OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, std::vector<PhaseIndexPackPtr>, VectorXi>(
              &OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX, std::vector<PhaseIndexPack>, VectorXi>(
              &OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            int,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkObjective));

  obj.def("addLinkObjective",
          py::overload_cast<ScalarFunctionalX,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            PhasePtr,
                            std::string,
                            Eigen::VectorXi,
                            Eigen::VectorXi>(&OptimalControlProblem::addLinkObjective));


  ///////////////////////
  obj.def("returnLinkEqualConVals", &OptimalControlProblem::returnLinkEqualConVals);
  obj.def("returnLinkEqualConLmults", &OptimalControlProblem::returnLinkEqualConLmults);

  obj.def("returnLinkInequalConVals", &OptimalControlProblem::returnLinkInequalConVals);
  obj.def("returnLinkInequalConLmults", &OptimalControlProblem::returnLinkInequalConLmults);


  ///////////////////////

  obj.def_readwrite("AutoScaling", &OptimalControlProblem::AutoScaling);


  obj.def_readwrite("AdaptiveMesh", &OptimalControlProblem::AdaptiveMesh);
  obj.def_readwrite("PrintMeshInfo", &OptimalControlProblem::PrintMeshInfo);
  obj.def_readwrite("MaxMeshIters", &OptimalControlProblem::MaxMeshIters);
  obj.def_readonly("MeshConverged", &OptimalControlProblem::MeshConverged);
  obj.def_readwrite("SolveOnlyFirst", &OptimalControlProblem::SolveOnlyFirst);

  obj.def("setAdaptiveMesh",
          &OptimalControlProblem::setAdaptiveMesh,
          py::arg("AdaptiveMesh") = true,
          py::arg("ApplyToPhases") = true);
  obj.def("setMeshTol", &OptimalControlProblem::setMeshTol);
  obj.def("setMeshRedFactor", &OptimalControlProblem::setMeshRedFactor);
  obj.def("setMeshIncFactor", &OptimalControlProblem::setMeshIncFactor);
  obj.def("setMeshErrFactor", &OptimalControlProblem::setMeshErrFactor);
  obj.def("setMaxMeshIters", &OptimalControlProblem::setMaxMeshIters);
  obj.def("setMinSegments", &OptimalControlProblem::setMinSegments);
  obj.def("setMaxSegments", &OptimalControlProblem::setMaxSegments);
  obj.def("setMeshErrorCriteria", &OptimalControlProblem::setMeshErrorCriteria);
  obj.def("setMeshErrorEstimator", &OptimalControlProblem::setMeshErrorEstimator);
}
