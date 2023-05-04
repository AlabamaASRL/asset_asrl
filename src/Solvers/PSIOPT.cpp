#include "PSIOPT.h"

#include <mkl.h>

#include "PyDocString/Solvers/PSIOPT_doc.h"

void ASSET::PSIOPT::setNLP(std::shared_ptr<NonLinearProgram> np) {
  this->nlp = np;
  this->PrimalVars = this->nlp->PrimalVars;
  this->EqualCons = this->nlp->EqualCons;
  this->InequalCons = this->nlp->InequalCons;
  this->SlackVars = this->nlp->SlackVars;
  this->KKTdim = this->nlp->KKTdim;
  this->setQPParams();
  mkl_set_num_threads(QPThreads);


  this->nlp->analyzeSparsity(this->KKTSol.getMatrix());
  if (storespmat)
    spmat = this->KKTSol.getMatrix();
  this->QPanalyzed = false;
}

void ASSET::PSIOPT::max_primal_dual_step(Eigen::Ref<Eigen::VectorXd> XSL,
                                         Eigen::Ref<Eigen::VectorXd> DXSL,
                                         double bfrac,
                                         double& alphap,
                                         double& alphad) {
  double Smax = this->max_step_to_boundary(this->getSlacks(XSL), this->getSlacks(DXSL), bfrac);
  double Lmax = this->max_step_to_boundary(this->getIqLmults(XSL), this->getIqLmults(DXSL), bfrac);

  double primstep = Smax;
  double slackstep = Smax;
  double eqmultstep = Smax;
  double iqmultstep = Lmax;

  if (this->PDStepStrategy == PDStepStrategies::PrimSlackEq_Iq) {
  } else if (this->PDStepStrategy == PDStepStrategies::AllMinimum) {
    double step = std::min(Smax, Lmax);
    primstep = step;
    slackstep = step;
    eqmultstep = step;
    iqmultstep = step;
  } else if (this->PDStepStrategy == PDStepStrategies::PrimSlack_EqIq) {
    eqmultstep = Lmax;
  } else if (this->PDStepStrategy == PDStepStrategies::MaxEq) {
    double step = std::max(Smax, Lmax);
    eqmultstep = step;
  }
  this->getPrimals(DXSL) *= primstep;
  if (InequalCons > 0)
    this->getSlacks(DXSL) *= slackstep;
  if (EqualCons > 0)
    this->getEqLmults(DXSL) *= eqmultstep;
  if (InequalCons > 0)
    this->getIqLmults(DXSL) *= iqmultstep;

  alphap = Smax;
  alphad = Lmax;
}

void ASSET::PSIOPT::fill_iter_info(Eigen::Ref<Eigen::VectorXd> XSL,
                                   Eigen::Ref<Eigen::VectorXd> RHS,
                                   double pobj,
                                   double bobj,
                                   double mu,
                                   IterateInfo& iter) const {


  iter.PrimObj = pobj;
  iter.BarrObj = bobj;
  iter.Mu = mu;
  iter.PPivots = this->KKTSol.ppivs();
  iter.KKTInf = this->getPrimGrad(RHS).lpNorm<Eigen::Infinity>();

  double avgcomp = 0;
  double mincomp = 0;
  double maxcomp = 0;
  if (InequalCons > 0) {
    iter.IConInf = this->getIqCons(RHS).lpNorm<Eigen::Infinity>();
    iter.IConNormErr = this->getIqCons(RHS).norm();
    iter.MaxIMult = this->getIqLmults(XSL).lpNorm<Eigen::Infinity>();
    this->complementarity(this->getSlacks(XSL), this->getIqLmults(XSL), avgcomp, mincomp, maxcomp);

    iter.BarrInf = maxcomp;
    iter.BarrNormErr = avgcomp;
  }
  if (EqualCons > 0) {
    iter.EConInf = this->getEqCons(RHS).lpNorm<Eigen::Infinity>();
    iter.EConNormErr = this->getEqCons(RHS).norm();
    iter.MaxEMult = this->getEqLmults(XSL).lpNorm<Eigen::Infinity>();
  }

  iter.KKTNormErr = this->getPrimGrad(RHS).norm();

  if (EqualCons > 0 || InequalCons > 0)
    iter.AllConNormErr = this->getAllCons(RHS).norm();
}

void ASSET::PSIOPT::evalNLP(int algmode,
                            double ObjScale,
                            ConstEigenRef<VectorXd> XSL,
                            double& val,
                            EigenRef<VectorXd> GX,
                            EigenRef<VectorXd> AGXS_FX,
                            Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat) {
  std::fill_n(KKTmat.valuePtr(), KKTmat.nonZeros(), 0.0);

  switch (algmode) {
    case AlgorithmModes::OPT:
      evalKKT(ObjScale, XSL, val, GX, AGXS_FX, KKTmat);
      break;
    case AlgorithmModes::OPTNO:
      evalKKTNO(ObjScale, XSL, val, GX, AGXS_FX, KKTmat);

      break;
    case AlgorithmModes::INIT:
      evalAUG(ObjScale, XSL, val, GX, AGXS_FX, KKTmat);
      break;
    case AlgorithmModes::SOE:
      this->nlp->setPrimalDiags(1.0);
      evalSOE(0.0, XSL, val, GX, AGXS_FX, KKTmat);
      this->nlp->setPrimalDiags(0.0);
      this->getPrimGrad(GX).setZero();
      this->getPrimGrad(AGXS_FX).setZero();
      break;
  }
}

ASSET::PSIOPT::ConvergenceFlags ASSET::PSIOPT::convergeCheck(std::vector<IterateInfo>& iters) {
  ConvergenceFlags Flag = CONVERGED;
  IterateInfo last = iters.back();
  bool KKTFeas = (last.KKTInf < this->KKTtol);
  bool EConFeas = (last.EConInf < this->EContol);
  bool IConFeas = (last.IConInf < this->IContol);
  bool BarFeas = (last.BarrInf < this->Bartol);

  bool KKTDiv = (last.KKTInf > this->DivKKTtol) || !std::isfinite(last.KKTInf);
  bool EConDiv = (last.EConInf > this->DivEContol) || !std::isfinite(last.EConInf);
  bool IConDiv = (last.IConInf > this->DivIContol) || !std::isfinite(last.IConInf);
  bool BarDiv = (last.BarrInf > this->DivBartol) || !std::isfinite(last.BarrInf);


  if (KKTDiv || EConDiv || IConDiv || BarDiv) {
    Flag = ConvergenceFlags::DIVERGING;
    return Flag;
  } else if (KKTFeas && EConFeas && IConFeas && BarFeas) {
    Flag = ConvergenceFlags::CONVERGED;
    return Flag;
  } else if (int(iters.size()) > this->MaxAccIters) {
    int nfeas = 0;
    for (int i = 0; i < this->MaxAccIters; i++) {
      last = iters[int(iters.size()) - i - 1];
      KKTFeas = (last.KKTInf < this->AccKKTtol);
      EConFeas = (last.EConInf < this->AccEContol);
      IConFeas = (last.IConInf < this->AccIContol);
      BarFeas = (last.BarrInf < this->AccBartol);
      if (KKTFeas && EConFeas && IConFeas && BarFeas)
        nfeas++;
      else
        break;
    }
    if (nfeas == this->MaxAccIters) {
      Flag = ConvergenceFlags::ACCEPTABLE;
      return Flag;
    }
  }
  Flag = ConvergenceFlags::NOTCONVERGED;
  return Flag;
}

void ASSET::PSIOPT::printPSIOPT() {

  std::string PsioptStr = "       ____    _____    ____          ____     ____   ______\n"
                          "      / __ \\  / ___/   /  _/         / __ \\   / __ \\ /_  __/\n"
                          "     / /_/ /  \\__ \\    / /   ______ / / / /  / /_/ /  / /   \n"
                          "    / ____/  ___/ /  _/ /   /_____// /_/ /  / ____/  / /    \n"
                          "   /_/      /____/  /___/          \\____/  /_/      /_/    \n";
  print_Header();
  fmt::print(fmt::fg(fmt::color::crimson), PsioptStr);
  fmt::print(fmt::fg(fmt::color::crimson), " \n       Parallel Sparse Interior-point Optimizer\n");
  print_Header();
}

void ASSET::PSIOPT::print_settings() {
  using std::cout;
  using std::endl;

  auto cyan = fmt::fg(fmt::color::cyan);
  auto magenta = fmt::fg(fmt::color::magenta);

  fmt::print(magenta, "Convergence Criteria\n\n");

  fmt::print("{0:_^{1}}\n", "", 39);
  fmt::print("|------|   tol   | Acctol  | Divtol  |\n");
  fmt::print(
      "|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "KKT", this->KKTtol, this->AccKKTtol, this->DivKKTtol);
  fmt::print(
      "|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "Bar", this->Bartol, this->AccBartol, this->DivBartol);
  fmt::print(
      "|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "ECons", this->EContol, this->AccEContol, this->DivEContol);
  fmt::print(
      "|{0:<6}|{1:>8.3e}|{2:>8.3e}|{3:>8.3e}|\n", "ICons", this->IContol, this->AccIContol, this->DivIContol);
}

void ASSET::PSIOPT::print_matrixinfo() {
}

void ASSET::PSIOPT::print_stats() {
  printPSIOPT();

  auto cyan = fmt::fg(fmt::color::cyan);
  auto magenta = fmt::fg(fmt::color::magenta);

  fmt::print(magenta, "Problem Statistics\n\n");

  fmt::print(" Primal Variables         : ");
  fmt::print(cyan, "{:<10}\n", this->PrimalVars);
  fmt::print(" Equality Constraints     : ");
  fmt::print(cyan, "{:<10}\n", this->EqualCons);
  fmt::print(" Inequality Constraints   : ");
  fmt::print(cyan, "{:<10}\n", this->InequalCons);
  fmt::print("\n");
  fmt::print(" KKT-Matrix DIM (P+E+2*I) : ");
  fmt::print(cyan, "{:<10}\n", this->KKTdim);
  fmt::print(" KKT-Matrix NNZs          : ");
  fmt::print(cyan, "{:<10}\n", this->KKTSol.getMatrix().nonZeros());
  fmt::print(" KKT-Matrix NNZ%          : ");
  fmt::print(
      cyan,
      "{:.6f}%\n",
      100.0 * double(this->KKTSol.getMatrix().nonZeros()) / (double(this->KKTdim) * double(this->KKTdim)));
  fmt::print("\n");

  // print_settings();

  // fmt::print("\n");
}

void ASSET::PSIOPT::print_last_iterate(const std::vector<IterateInfo>& iters) {
  auto last = iters.back();
  bool wide = false;

  if (last.iter % 10 == 0) {
    if (WideConsole) {
      fmt::print("{0:=^{1}}\n", "", 159);
      fmt::print("|Iter| Mu Val | Prim Obj |  Bar Obj |  KKT Inf |  Bar Inf | ECons Inf| ICons Inf|Max "
                 "EMult|Max IMult| AlphaP | AlphaD | AlphaT | Merit Val|LSI|PPS|HFI| HPert |\n");
    } else {
      fmt::print("{0:=^{1}}\n", "", 119);
      fmt::print("|Iter| Mu Val | Prim Obj |  Bar Obj |  KKT Inf |  Bar Inf | ECons Inf| ICons Inf| AlphaP | "
                 "AlphaD |LS| PPS |HF| HPert |\n");
    }
    auto tst = "|Iter|Mu Val | Prim Obj |KKT Inf |ECon Inf|ICon Inf|AlphaM |LS|PPS|HF| HPert |\n";
  }


  fmt::text_style PHashcol = fmt::text_style();
  fmt::text_style EHashcol = fmt::text_style();
  fmt::text_style IHashcol = fmt::text_style();
  fmt::text_style KHashcol = fmt::text_style();
  fmt::text_style BHashcol = fmt::text_style();
  fmt::text_style BOHashcol = fmt::text_style();


  fmt::text_style Kcol = calculate_color(last.KKTInf, this->KKTtol, this->AccKKTtol);
  fmt::text_style Bcol = calculate_color(last.BarrInf, this->Bartol, this->AccBartol);
  fmt::text_style Ecol = calculate_color(last.EConInf, this->EContol, this->AccEContol);
  fmt::text_style Icol = calculate_color(last.IConInf, this->IContol, this->AccIContol);

  if (iters.size() > 1) {

    auto GCol = fmt::fg(fmt::color::lime_green);
    auto BCol = fmt::fg(fmt::color::red);

    PHashcol = (iters.back().PrimObj <= iters[iters.size() - 2].PrimObj) ? GCol : BCol;
    BOHashcol = (iters.back().BarrObj <= iters[iters.size() - 2].BarrObj) ? GCol : BCol;
    BHashcol = (iters.back().BarrInf <= iters[iters.size() - 2].BarrInf) ? GCol : BCol;
    EHashcol = (iters.back().EConInf <= iters[iters.size() - 2].EConInf) ? GCol : BCol;
    IHashcol = (iters.back().IConInf <= iters[iters.size() - 2].IConInf) ? GCol : BCol;
    KHashcol = (iters.back().KKTInf <= iters[iters.size() - 2].KKTInf) ? GCol : BCol;
  }


  auto hash = []() { fmt::print("|"); };
  auto chash = [](fmt::text_style c) { fmt::print(c, "|"); };

  hash();
  fmt::print("{:<4}", last.iter);
  hash();
  fmt::print("{:.2e}", last.Mu);
  hash();
  fmt::print("{:>10.3e}", last.PrimObj);
  chash(PHashcol);
  fmt::print("{:>10.3e}", last.BarrObj);
  chash(BOHashcol);
  fmt::print(Kcol, "{:>10.4e}", last.KKTInf);
  chash(KHashcol);
  fmt::print(Bcol, "{:>10.4e}", last.BarrInf);
  chash(BHashcol);
  fmt::print(Ecol, "{:>10.4e}", last.EConInf);
  chash(EHashcol);
  fmt::print(Icol, "{:>10.4e}", last.IConInf);
  chash(IHashcol);

  if (WideConsole) {
    fmt::print("{:>9.3e}|{:>9.3e}|{:>8.2e}|{:>8.2e}|{:>8.2e}|{:>10.3e}|{:>3}|{:>3}|{:>3}|{:>6.1e}|\n",
               last.MaxEMult,
               last.MaxIMult,
               last.alphaP,
               last.alphaD,
               last.alphaT,
               last.MeritVal,
               last.LSiters,
               last.PPivots,
               last.Hfacs,
               last.Hpert);
  } else {
    fmt::print("{:>8.2e}|{:>8.2e}|{:>2}|{:>5}|{:>2}|{:>6.1e}|\n",
               last.alphaT * last.alphaP,
               last.alphaT * last.alphaD,
               last.LSiters,
               last.PPivots,
               last.Hfacs,
               last.Hpert);
  }
}

void ASSET::PSIOPT::print_Beginning(std::string msg) const {
  fmt::print(fmt::fg(fmt::color::dim_gray), "Beginning");
  fmt::print(": ");
  fmt::print(fmt::fg(fmt::color::royal_blue), msg);
  fmt::print("\n");
}

void ASSET::PSIOPT::print_Finished(std::string msg) const {

  fmt::print(fmt::fg(fmt::color::dim_gray), "Finished ");
  fmt::print(": ");
  fmt::print(fmt::fg(fmt::color::royal_blue), msg);
  fmt::print("\n");
}

void ASSET::PSIOPT::print_ExitStats(ConvergenceFlags ExitCode,
                                    const std::vector<IterateInfo>& iters,
                                    double tottime,
                                    double nlptime,
                                    double qptime) {
  auto last = iters.back();
  fmt::text_style Kcol = calculate_color(last.KKTInf, this->KKTtol, this->AccKKTtol);
  fmt::text_style Bcol = calculate_color(last.BarrInf, this->Bartol, this->AccBartol);
  fmt::text_style Ecol = calculate_color(last.EConInf, this->EContol, this->AccEContol);
  fmt::text_style Icol = calculate_color(last.IConInf, this->IContol, this->AccIContol);


  int iternum = int(iters.size());
  double printtime = tottime - nlptime - qptime;
  auto TColor = fmt::fg(fmt::color::cyan);
  auto Printtime = [&](const char* msg, double t1) {
    fmt::print(msg);
    fmt::print(TColor, "{0:>10.3f} ms {1:>10.3f} ms/iter\n", t1, double(t1 / iternum));
  };

  if (this->PrintLevel < 3) {
    if (ExitCode == ConvergenceFlags::CONVERGED) {
      fmt::print(fmt::fg(fmt::color::lime_green), "\nOptimal Solution Found\n");
    } else if (ExitCode == ConvergenceFlags::ACCEPTABLE) {
      fmt::print(fmt::fg(fmt::color::yellow), "\nAcceptable Solution Found\n");
    } else if (ExitCode == ConvergenceFlags::DIVERGING) {
      fmt::print(fmt::fg(fmt::color::dark_red), "\nSolution Diverging\n");
    } else if (ExitCode == ConvergenceFlags::NOTCONVERGED) {
      fmt::print(fmt::fg(fmt::color::red), "\nNo Solution Found\n");
    }
  }

  if (this->PrintLevel < 2) {

    fmt::print(" Iterations : ");
    fmt::print("{:<5}\n", iternum);
    fmt::print(" Prim Obj   : ");
    fmt::print("{:<15.8e}\n", last.PrimObj);
    fmt::print(" KKT Inf    : ");
    fmt::print(Kcol, "{:<15.8e}\n", last.KKTInf);
    fmt::print(" Bar Inf    : ");
    fmt::print(Bcol, "{:<15.8e}\n", last.BarrInf);
    fmt::print(" ECons Inf  : ");
    fmt::print(Ecol, "{:<15.8e}\n", last.EConInf);
    fmt::print(" ICons Inf  : ");
    fmt::print(Icol, "{:<15.8e}\n", last.IConInf);

    fmt::print("\n");

    Printtime(" NLP Function Evaluation Time : ", nlptime);
    Printtime(" KKT Matrix Factor/Solve Time : ", qptime);
    Printtime(" Console Print Time           : ", printtime);
    Printtime(" Total Time (NLP+KKT+Print)   : ", tottime);

    fmt::print("\n");
  }
}

fmt::text_style ASSET::PSIOPT::calculate_color(double val, double targ, double acc) {
  auto level1 = std::log(targ);
  auto level3 = std::log(acc);
  auto level5 = std::log(acc * 1000.0);
  auto level2 = (level1 + level3) / 2.0;
  auto level4 = (level3 + level5) / 2.0;

  auto logval = std::log(val);
  fmt::color c;

  if (logval < level1)
    c = fmt::color::lime_green;
  else if (logval < level2)
    c = fmt::color::yellow;
  else if (logval < level3)
    c = fmt::color::orange;
  else if (logval < level4)
    c = fmt::color::red;
  else
    c = fmt::color::dark_red;
  return fmt::fg(c);
}

int ASSET::PSIOPT::factor_impl(
    bool docompute, bool Zfac, double ipurt, double incpurt0, double incpurt, double& finalpert) {
  auto Inertia = [&]() { return this->KKTSol.neigs() - (this->EqualCons + this->InequalCons); };
  auto RankDef = [&]() {
    if ((this->KKTSol.neigs() + this->KKTSol.peigs() - this->KKTdim) != 0) {
      std::cout << "Potential Rank Deficiency Detected!!!" << std::endl;
    }
  };
  auto Perturb = [&](double p) { this->nlp->perturbKKTPDiags(p, this->KKTSol.getMatrix()); };
  auto Factor = [&]() { this->KKTSol.factorize_internal(); };
  auto Compute = [&]() { this->KKTSol.compute_internal(); };
  int IncEigs;

  if (Zfac || docompute) {
    if (!docompute)
      Factor();
    else
      Compute();
    RankDef();
    IncEigs = Inertia();
    finalpert = 0.0;
    if (IncEigs <= 0)
      return 0;
  }
  double p = ipurt;

  for (int i = 0; i < this->MaxRefac; i++) {
    Perturb(p);
    Factor();
    RankDef();
    IncEigs = Inertia();
    finalpert += p;

    if (IncEigs <= 0)
      return i + 1;
    if (i == 0)
      p *= incpurt0;
    else
      p *= incpurt;
  }
  return this->MaxRefac;
}

Eigen::VectorXd ASSET::PSIOPT::alg_impl(AlgorithmModes algmode,
                                        BarrierModes barmode,
                                        LineSearchModes lsmode,
                                        double ObjScale,
                                        double MuI,
                                        Eigen::Ref<Eigen::VectorXd> xsl) {
  Eigen::VectorXd XSL = xsl;
  Eigen::VectorXd RHS(this->KKTdim);
  Eigen::VectorXd DXSL(this->KKTdim);
  Eigen::VectorXd RHS2(this->KKTdim);
  Eigen::VectorXd PGX(this->PrimalVars);

  Eigen::VectorXd Temp(this->KKTdim);
  Eigen::VectorXd Err;

  double Mu = MuI;

  Utils::Timer Runtimer;
  Utils::Timer Funtimer;
  Utils::Timer LStimer;
  Utils::Timer QPtimer;
  Utils::Timer CBtimer;

  double Hpert0 = this->deltaH;
  std::vector<IterateInfo> iters;
  iters.reserve(this->MaxIters);
  ConvergenceFlags ExitCode;
  bool FirstPert = true;

  Runtimer.start();
  for (int i = 0; i < this->MaxIters; i++) {
    IterateInfo Citer;
    Citer.iter = i;

    double avgcomp = 0;
    double mincomp = 0;
    double maxcomp = 0;
    double alpha = 1.0;
    double alphap = 1.0;
    double alphad = 1.0;

    RHS.setZero();
    PGX.setZero();
    double PrimObj = 0;
    double BarrObj = 0;

    Funtimer.start();
    /////////////////////////////////////////////////////////////

    this->evalNLP(algmode, ObjScale, XSL, PrimObj, PGX, RHS, this->KKTSol.getMatrix());


    if (this->InequalCons > 0) {
      this->apply_reset_slacks(this->getSlacks(XSL), this->getIqCons(RHS));
      this->barrier_hessian(this->KKTSol.getMatrix(), this->getSlacks(XSL), this->getIqLmults(XSL), Mu);
      this->complementarity(this->getSlacks(XSL), this->getIqLmults(XSL), avgcomp, mincomp, maxcomp);
    }

    ///////////////////////////////////////////////////////////////
    Funtimer.stop();
    if (this->EarlyCallBackEnabled) {
      CBtimer.start();
      this->EarlyCallBack(i, ObjScale, XSL, PrimObj, PGX, RHS, this->KKTSol.getMatrix());
      CBtimer.stop();
    }
    QPtimer.start();
    ////////////////////////////////////////////////////////////////
    RHS.head(this->PrimalVars) += PGX;

    ////////////////////////////////////////////////////////////////
    double nhpert = 0;
    double Incr = this->incrH;
    double Incr2 = this->incrH;
    if (FirstPert)
      Incr2 *= this->incrH;
    bool Zfac = true;
    if (this->FastFactorAlg && i > 6 && ((i * 3) % 4) != 0) {
      bool cycling = true;
      for (int j = 0; j < 4; j++) {
        int ns = iters[iters.size() - 1 - j].Hfacs;
        if (ns == 0) {
          cycling = false;
          break;
        }
      }
      Zfac = !cycling;
    }

    Citer.Hfacs = this->factor_impl(false, Zfac, Hpert0, Incr, Incr2, nhpert);

    if (Citer.Hfacs > 0) {
      Hpert0 = std::max(this->deltaH, nhpert * decrH);
      FirstPert = false;
    }
    Citer.Hpert = nhpert;
    ///////////////////////////////////////////////////////////////////

    if (this->InequalCons > 0) {
      switch (barmode) {
        case BarrierModes::PROBE:
          this->barrier_gradient(this->getIqLmults(XSL), this->getDualGrad(RHS));
          DXSL = -this->KKTSol.solve(RHS);
          this->max_primal_dual_step(XSL, DXSL, this->BoundFraction, alphap, alphad);
          Temp = XSL + DXSL;
          Mu = this->MPCMu(this->getSlacks(Temp), this->getIqLmults(Temp), avgcomp, mincomp);

          break;
        case BarrierModes::LOQO:
          Mu = this->LOQOMu(this->getSlacks(XSL), this->getIqLmults(XSL), avgcomp, mincomp);
          break;
        case BarrierModes::FIACCO:
          break;
        default:
          break;
      }

      Mu = std::max(Mu, this->MinMu);
      Mu = std::min(Mu, this->MaxMu);
      BarrObj = this->barrier_objective(this->getSlacks(XSL), Mu);
      this->barrier_gradient(this->getSlacks(XSL), this->getIqLmults(XSL), Mu, this->getDualGrad(RHS));
    }

    DXSL = -this->KKTSol.solve(RHS);

    if (Diagnostic) {
    }

    if (this->InequalCons > 0)
      this->max_primal_dual_step(XSL, DXSL, this->BoundFraction, alphap, alphad);
    /////////////////////////////////////////////////////////////////////

    QPtimer.stop();

    Funtimer.start();
    //////////////////////////////////////////////////////////////////////

    alpha = ls_impl(lsmode, ObjScale, Mu, PrimObj, BarrObj, XSL, DXSL, Temp, RHS, RHS2, Citer, iters);


    //////////////////////////////////////////////////////////////////////
    Funtimer.stop();

    Citer.alphaP = alphap;
    Citer.alphaD = alphad;
    Citer.alphaT = alpha;

    this->fill_iter_info(XSL, RHS, PrimObj, BarrObj, Mu, Citer);
    iters.push_back(Citer);

    if (this->LateCallBackEnabled) {
      CBtimer.start();
      this->LateCallBack(iters.back(), XSL, RHS);
      CBtimer.stop();
    }

    ExitCode = this->convergeCheck(iters);
    if (this->PrintLevel == 0) {
      this->print_last_iterate(iters);
    }

    if (ExitCode == ConvergenceFlags::CONVERGED || ExitCode == ConvergenceFlags::ACCEPTABLE
        || ExitCode == ConvergenceFlags::DIVERGING || i == (this->MaxIters - 1)) {

      this->ConvergeFlag = ExitCode;
      break;
    }

    /////////Very Important/////////
    XSL += alpha * DXSL;
    ///////////////////////////////
  }

  if (algmode == AlgorithmModes::OPT) {
    this->LastObjVal = iters.back().PrimObj;
  } else {
    Funtimer.start();
    this->LastObjVal = 0;
    this->nlp->evalOBJ(ObjScale, XSL.head(this->PrimalVars), this->LastObjVal);
    Funtimer.stop();
  }


  if (this->EqualCons > 0) {
    this->LastEqCons = this->getEqCons(RHS);
    this->LastEqLmults = this->getEqLmults(XSL);

  }
  if (this->InequalCons > 0) {
    this->LastIqCons = this->getIqCons(RHS) - this->getSlacks(XSL);
    this->LastIqLmults = this->getIqLmults(XSL);
  }


  Runtimer.stop();
  this->LastIterNum += iters.size();
  double qptime = double(QPtimer.count<std::chrono::microseconds>()) / 1000000.0;
  double nlptime = double(Funtimer.count<std::chrono::microseconds>()) / 1000000.0;
  double tottime = double(Runtimer.count<std::chrono::microseconds>()) / 1000000.0;

  this->LastFuncTime += nlptime;
  this->LastKKTTime += qptime;

  ///////////////////////////////////////////////////////////////////////////

  print_ExitStats(ExitCode, iters, tottime * 1000, nlptime * 1000, qptime * 1000);
  ////////////////////////////////////////////////////////////////////////////

  return XSL;
}

Eigen::VectorXd ASSET::PSIOPT::init_impl(const Eigen::VectorXd& x, double Mu, bool docompute) {


  Utils::Timer kktt;
  kktt.start();

  Eigen::VectorXd XSL(this->KKTdim);
  XSL.setZero();
  XSL.head(this->PrimalVars) = x;

  Eigen::VectorXd RHS(this->KKTdim);
  RHS.setZero();
  double val = 0;
  this->nlp->setPrimalDiags(1.0);
  if (this->InequalCons > 0) {
    this->nlp->setSlacksOnes();
  }
  this->evalNLP(AlgorithmModes::INIT,
                this->ObjScale,
                XSL,
                val,
                RHS.head(this->PrimalVars),
                RHS,
                this->KKTSol.getMatrix());

  Eigen::VectorXd hp(this->SlackVars);

  for (int i = 0; i < this->SlackVars; i++) {
    double fxi = this->getIqCons(RHS)[i];
    if (fxi < -this->BoundPush) {
      this->getSlacks(XSL)[i] = abs(fxi);
    } else {
      this->getSlacks(XSL)[i] = this->BoundPush;
    }
    hp[i] = 1.0;
    this->getIqLmults(XSL)[i] = Mu / this->getSlacks(XSL)[i];
  }

  RHS.tail(this->EqualCons + this->InequalCons).setZero();

  if (this->InequalCons > 0)
    this->nlp->assignKKTSlackHessian(hp, this->KKTSol.getMatrix());
  if (this->PrintLevel < 2) {
    print_Beginning("KKT-Matrix Analysis ");
  }

  if (docompute)
    this->KKTSol.compute_internal();
  else
    this->KKTSol.factorize_internal();
  kktt.stop();

  double pretime = double(kktt.count<std::chrono::microseconds>()) / 1000000.0;
  this->LastPreTime += pretime;

  this->FactorFlops = this->KKTSol.m_flops;
  this->FactorMem = this->KKTSol.m_mem;

  if (this->PrintLevel < 2) {
    auto cyan = fmt::fg(fmt::color::cyan);
    if (docompute) {
      fmt::print(" LDLT Factor NNZs      : ");
      fmt::print(cyan, "{0:<10}\n", this->FactorMem);
      fmt::print(" LDLT Factor FLOPs     : ");
      fmt::print(cyan, "{0} MFLOPs\n", this->FactorFlops);
    }
    fmt::print(" Analysis/Reorder Time : ");
    fmt::print(cyan, "{0:.3f} ms\n", pretime * 1000);
    print_Finished("KKT-Matrix Analysis ");
  }

  Eigen::VectorXd dx = -this->KKTSol.solve(RHS);

  if (EqualCons > 0)
    this->getEqLmults(XSL) = this->getEqLmults(dx);
  if (this->InequalCons > 0)
    this->nlp->setSlackDiags(0.0);
  this->nlp->setPrimalDiags(0.0);

  return XSL;
}


double ASSET::PSIOPT::ls_impl(LineSearchModes lsmode,
                              double ObjScale,
                              double Mu,
                              double PrimObj,
                              double BarrObj,
                              EigenRef<VectorXd> XSL,
                              EigenRef<VectorXd> DXSL,
                              EigenRef<VectorXd> XSL2,
                              EigenRef<VectorXd> RHS,
                              EigenRef<VectorXd> RHS2,
                              IterateInfo& Citer,
                              const std::vector<IterateInfo>& iters) {


  // Do not modify RHS,XSL,DXSL. EigenRef<VectorXd> doesnt like to be explicitly const in this instance,
  // I will fix this later in refactor

  double alpha = 1.0;


  if (lsmode == LineSearchModes::LANG) {
    double LangInit = PrimObj + BarrObj + this->getLmults(XSL).dot(this->getAllCons(RHS));

    for (int j = 0; j < this->MaxLSIters; j++) {
      double ptest = 0;
      double btest = 0;
      XSL2 = XSL + alpha * DXSL;
      RHS2.setZero();
      this->evalRHS(ObjScale, XSL2, ptest, RHS2, RHS2);
      this->apply_reset_slacks(this->getSlacks(XSL2), this->getIqCons(RHS2));
      btest = this->barrier_objective(this->getSlacks(XSL2), Mu);
      this->barrier_gradient(this->getSlacks(XSL2), this->getIqLmults(XSL2), Mu, this->getDualGrad(RHS2));
      double LangTest = ptest + btest + this->getLmults(XSL2).dot(this->getAllCons(RHS2));
      Citer.LSiters = j;
      if (LangTest < LangInit) {
        break;
      } else {
        alpha = alpha / this->alphaRed;
      }
    }
  } else if (lsmode == LineSearchModes::L1) {
    double vv =
        RHS.head(this->PrimalVars + this->SlackVars).dot(DXSL.head(this->PrimalVars + this->SlackVars));
    double cv = this->getLmults(DXSL).dot(this->getAllCons(RHS));

    double LangInit = PrimObj + BarrObj;
    double InitL1Pen = this->getLmults(XSL).cwiseAbs().dot(this->getAllCons(RHS).cwiseAbs());
    double InitL2Pen = this->getAllCons(RHS).squaredNorm();

    double sc = .1 + std::abs(vv - cv) / InitL2Pen;
    if (InitL2Pen == 0.0)
      sc = 1.0;

    LangInit += InitL1Pen + InitL2Pen * sc;

    for (int j = 0; j < this->MaxLSIters; j++) {
      double ptest = 0;
      double btest = 0;
      XSL2 = XSL + alpha * DXSL;
      RHS2.setZero();
      this->nlp->evalOCC(ObjScale,
                         XSL2.head(this->PrimalVars),
                         ptest,
                         RHS2.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                         RHS2.tail(this->InequalCons));

      this->apply_reset_slacks(this->getSlacks(XSL2), this->getIqCons(RHS2));
      btest = this->barrier_objective(this->getSlacks(XSL2), Mu);

      double LangTest = ptest + btest;
      double TestL1Pen = this->getLmults(XSL).cwiseAbs().dot(this->getAllCons(RHS2).cwiseAbs());
      double TestL2Pen = this->getAllCons(RHS2).squaredNorm();
      LangTest += TestL1Pen + TestL2Pen * sc;

      Citer.MeritVal = LangTest;
      if (LangTest < LangInit) {
        Citer.LSiters = j;
        break;
      } else {
        Citer.LSiters = j + 1;
        alpha = alpha / this->alphaRed;
      }
    }
  } else if (lsmode == LineSearchModes::AUGLANG) {


    double vv = this->getPrimDualGrad(RHS).dot(this->getPrimalsSlacks(DXSL));
    double cv = this->getLmults(DXSL).dot(this->getAllCons(RHS));

    double LangInit = PrimObj + BarrObj;
    double InitL1Pen = this->getLmults(XSL).cwiseAbs().dot(this->getAllCons(RHS).cwiseAbs());
    double InitL2Pen = this->getAllCons(RHS).squaredNorm();

    double sc = .01 + std::abs(vv - cv) / InitL2Pen;
    if (InitL2Pen == 0.0)
      sc = 1.0;

    LangInit += InitL1Pen + InitL2Pen * sc;

    for (int j = 0; j < this->MaxLSIters; j++) {
      double ptest = 0;
      double btest = 0;
      XSL2 = XSL + alpha * DXSL;
      RHS2.setZero();
      this->nlp->evalOCC(ObjScale,
                         XSL2.head(this->PrimalVars),
                         ptest,
                         RHS2.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                         RHS2.tail(this->InequalCons));

      this->apply_reset_slacks(this->getSlacks(XSL2), this->getIqCons(RHS2));
      btest = this->barrier_objective(this->getSlacks(XSL2), Mu);

      double LangTest = ptest + btest;

      double TestL1Pen = this->getLmults(XSL).cwiseAbs().dot(this->getAllCons(RHS2).cwiseAbs());

      TestL1Pen = 0;

      for (int i = 0; i < this->EqualCons; i++) {
        double eqerr = abs(this->getEqCons(RHS2)[i]);
        double eqmul = abs(this->getEqLmults(XSL)[i]);
        if (eqerr > this->EContol * 10) {
          TestL1Pen += eqerr * eqmul;
        }
      }

      for (int i = 0; i < this->InequalCons; i++) {
        double iqerr = abs(this->getIqCons(RHS2)[i]);
        double iqmul = abs(this->getIqLmults(XSL)[i]);
        if (iqerr > this->IContol * 10) {
          TestL1Pen += iqerr * iqmul;
        }
      }


      double TestL2Pen = this->getAllCons(RHS2).squaredNorm();

      if (TestL2Pen < EContol * EContol * EqualCons + IContol * IContol * InequalCons) {
        TestL2Pen = 0;
      }

      LangTest += TestL1Pen + TestL2Pen * sc;

      Citer.MeritVal = LangTest;
      if (LangTest < LangInit) {
        Citer.LSiters = j;
        break;
      } else {
        Citer.LSiters = j + 1;
        alpha = alpha / this->alphaRed;
      }
    }
  } else
    Citer.LSiters = 0;


  return alpha;
}


Eigen::VectorXd ASSET::PSIOPT::optimize(const Eigen::VectorXd& x) {


  this->zero_timing_stats();

  if (this->PrintLevel == 0)
    print_stats();
  if (this->PrintLevel < 2) {
    print_Header();
    print_Beginning("PSIOPT ");
  }
  Utils::Timer t;
  t.start();

  bool docompute = analyze_KKT_Matrix();

  Eigen::VectorXd XSL = this->init_impl(x, this->initMu, docompute);

  Eigen::VectorXd XSLans(this->KKTdim);
  XSLans.setZero();
  if (this->PrintLevel < 2) {
    print_Beginning("Optimization Algorithm ");
  }
  XSLans = this->alg_impl(OPT, this->OptBarMode, this->OptLSMode, this->ObjScale, this->initMu, XSL);
  if (this->PrintLevel < 2) {
    print_Finished("Optimization Algorithm ");
  }

 

  t.stop();
  double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
  this->LastTotalTime = tottime / 1000.0;
  this->LastMiscTime = this->LastTotalTime - this->LastPreTime - this->LastKKTTime - this->LastFuncTime;

  if (this->PrintLevel < 2) {
    fmt::print(" PSIOPT Total Time : ");
    fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", tottime);
    print_Finished("PSIOPT ");
    print_Header();
  }
  return this->getPrimals(XSLans);
}

Eigen::VectorXd ASSET::PSIOPT::solve_optimize(const Eigen::VectorXd& x) {

  this->zero_timing_stats();
  if (this->PrintLevel == 0)
    print_stats();
  if (this->PrintLevel < 2) {
    print_Header();
    print_Beginning("PSIOPT ");
  }
  Utils::Timer t;
  t.start();

  bool docompute = analyze_KKT_Matrix();

  Eigen::VectorXd XSL = this->init_impl(x, this->initMu, docompute);
  Eigen::VectorXd XSLans(this->KKTdim);
  XSLans.setZero();

  if (this->PrintLevel < 2) {
    print_Beginning("Solve Algorithm ");
  }

  XSLans =
      this->alg_impl(this->SoeMode, this->SoeBarMode, this->SoeLSMode, this->ObjScale, this->initMu, XSL);
  if (this->PrintLevel < 2) {
    print_Finished("Solve Algorithm ");
  }
  Eigen::VectorXd Xt = this->getPrimals(XSLans);
  XSL = this->init_impl(Xt, this->initMu, false);

  if (this->PrintLevel < 2) {
    print_Beginning("Optimization Algorithm ");
  }
  XSLans = this->alg_impl(OPT, this->OptBarMode, this->OptLSMode, this->ObjScale, this->initMu, XSL);

  t.stop();
  double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
  this->LastTotalTime = tottime / 1000.0;
  this->LastMiscTime = this->LastTotalTime - this->LastPreTime - this->LastKKTTime - this->LastFuncTime;

  if (this->PrintLevel < 2) {
    print_Finished("Optimization Algorithm ");
    fmt::print(" PSIOPT Total Time : ");
    fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", tottime);
    print_Finished("PSIOPT ");
    print_Header();
  }

  
  return this->getPrimals(XSLans);
}

Eigen::VectorXd ASSET::PSIOPT::solve_optimize_solve(const Eigen::VectorXd& x) {
  this->zero_timing_stats();
  if (this->PrintLevel == 0)
    print_stats();
  if (this->PrintLevel < 2) {
    print_Header();
    print_Beginning("PSIOPT ");
  }
  Utils::Timer t;
  t.start();

  bool docompute = analyze_KKT_Matrix();

  Eigen::VectorXd XSL = this->init_impl(x, this->initMu, docompute);
  Eigen::VectorXd XSLans(this->KKTdim);
  XSLans.setZero();

  if (this->PrintLevel < 2) {
    print_Beginning("Solve Algorithm ");
  }

  XSLans =
      this->alg_impl(this->SoeMode, this->SoeBarMode, this->SoeLSMode, this->ObjScale, this->initMu, XSL);
  if (this->PrintLevel < 2) {
    print_Finished("Solve Algorithm ");
  }
  Eigen::VectorXd Xt = this->getPrimals(XSLans);
  XSL = this->init_impl(Xt, this->initMu, false);

  if (this->PrintLevel < 2) {
    print_Beginning("Optimization Algorithm ");
  }
  XSLans = this->alg_impl(OPT, this->OptBarMode, this->OptLSMode, this->ObjScale, this->initMu, XSL);
  if (this->PrintLevel < 2) {
    print_Finished("Optimization Algorithm ");
  }
  if (this->ConvergeFlag == ConvergenceFlags::CONVERGED) {

  } else {
    Xt = this->getPrimals(XSLans);
    XSL = this->init_impl(Xt, this->initMu, false);

    if (this->PrintLevel < 2) {
      print_Beginning("Solve Algorithm ");
    }
    XSLans =
        this->alg_impl(this->SoeMode, this->SoeBarMode, this->SoeLSMode, this->ObjScale, this->initMu, XSL);

    if (this->PrintLevel < 2) {
      print_Finished("Solve Algorithm ");
    }
  }
  t.stop();
  double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
  this->LastTotalTime = tottime / 1000.0;
  this->LastMiscTime = this->LastTotalTime - this->LastPreTime - this->LastKKTTime - this->LastFuncTime;

  if (this->PrintLevel < 2) {
    fmt::print(" PSIOPT Total Time : ");
    fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", tottime);
    print_Finished("PSIOPT ");
    print_Header();
  }

 
  return this->getPrimals(XSLans);
}

Eigen::VectorXd ASSET::PSIOPT::optimize_solve(const Eigen::VectorXd& x) {
  this->zero_timing_stats();
  if (this->PrintLevel == 0)
    print_stats();
  if (this->PrintLevel < 2) {
    print_Header();
    print_Beginning("PSIOPT ");
  }
  Utils::Timer t;
  t.start();

  bool docompute = analyze_KKT_Matrix();

  Eigen::VectorXd XSL = this->init_impl(x, this->initMu, docompute);
  Eigen::VectorXd XSLans(this->KKTdim);

  if (this->PrintLevel < 2) {
    print_Beginning("Optimization Algorithm ");
  }

  XSLans = this->alg_impl(OPT, this->OptBarMode, this->OptLSMode, this->ObjScale, this->initMu, XSL);

  if (this->PrintLevel < 2) {
    print_Finished("Optimization Algorithm ");
  }

  if (this->ConvergeFlag == ConvergenceFlags::CONVERGED) {

  } else {
    Eigen::VectorXd Xt = this->getPrimals(XSLans);
    XSL = this->init_impl(Xt, this->initMu, false);

    if (this->PrintLevel < 2) {
      print_Beginning("Solve Algorithm ");
    }
    XSLans =
        this->alg_impl(this->SoeMode, this->SoeBarMode, this->SoeLSMode, this->ObjScale, this->initMu, XSL);

    if (this->PrintLevel < 2) {
      print_Finished("Solve Algorithm ");
    }
  }
  t.stop();
  double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
  this->LastTotalTime = tottime / 1000.0;
  this->LastMiscTime = this->LastTotalTime - this->LastPreTime - this->LastKKTTime - this->LastFuncTime;

  if (this->PrintLevel < 2) {
    fmt::print(" PSIOPT Total Time : ");
    fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", tottime);
    print_Finished("PSIOPT ");
    print_Header();
  }

  
  return this->getPrimals(XSLans);
}

Eigen::VectorXd ASSET::PSIOPT::solve(const Eigen::VectorXd& x) {

  this->zero_timing_stats();
  if (this->PrintLevel == 0)
    print_stats();
  if (this->PrintLevel < 2) {
    print_Header();
    print_Beginning("PSIOPT ");
  }
  Utils::Timer t;
  t.start();
  bool docompute = analyze_KKT_Matrix();

  Eigen::VectorXd XSL = this->init_impl(x, this->initMu, docompute);
  Eigen::VectorXd XSLans(this->KKTdim);
  XSLans.setZero();
  if (this->PrintLevel < 2) {
    print_Beginning("Solve Algorithm ");
  }
  XSLans =
      this->alg_impl(this->SoeMode, this->SoeBarMode, this->SoeLSMode, this->ObjScale, this->initMu, XSL);

  t.stop();
  double tottime = double(t.count<std::chrono::microseconds>()) / 1000.0;
  this->LastTotalTime = tottime / 1000.0;
  this->LastMiscTime = this->LastTotalTime - this->LastPreTime - this->LastKKTTime - this->LastFuncTime;

  if (this->PrintLevel < 2) {
    print_Finished("Solve Algorithm ");
    fmt::print(" PSIOPT Total Time : ");
    fmt::print(fmt::fg(fmt::color::cyan), "{0:.3f} ms\n", tottime);
    print_Finished("PSIOPT ");
    print_Header();
  }
  

  return this->getPrimals(XSLans);
}

void ASSET::PSIOPT::Build(py::module& m) {
  using namespace doc;
  auto obj = py::class_<PSIOPT, std::shared_ptr<PSIOPT>>(m, "PSIOPT");
  obj.def(py::init<std::shared_ptr<NonLinearProgram>>());
  obj.def(py::init<>());

  obj.def("optimize", &PSIOPT::optimize, PSIOPT_optimize);
  obj.def("solve_optimize", &PSIOPT::solve_optimize, PSIOPT_solve_optimize);
  obj.def("solve", &PSIOPT::solve, PSIOPT_solve);
  obj.def("setQPParams", &PSIOPT::setQPParams);

  obj.def_readwrite("MaxIters", &PSIOPT::MaxIters, PSIOPT_MaxIters);
  obj.def_readwrite("MaxAccIters", &PSIOPT::MaxAccIters, PSIOPT_MaxAccIters);
  obj.def_readwrite("MaxLSIters", &PSIOPT::MaxLSIters, PSIOPT_MaxLSIters);

  obj.def("set_MaxIters", &PSIOPT::set_MaxIters);
  obj.def("set_MaxAccIters", &PSIOPT::set_MaxAccIters);
  obj.def("set_MaxLSIters", &PSIOPT::set_MaxLSIters);


  obj.def_readwrite("alphaRed", &PSIOPT::alphaRed, PSIOPT_alphaRed);
  obj.def("set_alphaRed", &PSIOPT::set_alphaRed);


  obj.def_readwrite("WideConsole", &PSIOPT::WideConsole);


  obj.def_readwrite("FastFactorAlg", &PSIOPT::FastFactorAlg, PSIOPT_FastFactorAlg);


  obj.def_readwrite("LastTotalTime", &PSIOPT::LastTotalTime, PSIOPT_LastUserTime);
  obj.def_readwrite("LastPreTime", &PSIOPT::LastPreTime, PSIOPT_LastUserTime);
  obj.def_readwrite("LastFuncTime", &PSIOPT::LastFuncTime, PSIOPT_LastUserTime);
  obj.def_readwrite("LastKKTTime", &PSIOPT::LastKKTTime, PSIOPT_LastQPTime);
  obj.def_readwrite("LastMiscTime", &PSIOPT::LastMiscTime, PSIOPT_LastQPTime);
  obj.def_readwrite("LastIterNum", &PSIOPT::LastIterNum, PSIOPT_LastIterNum);
  obj.def_readwrite("LastObjVal", &PSIOPT::LastObjVal);


  obj.def_readwrite("ObjScale", &PSIOPT::ObjScale, PSIOPT_ObjScale);
  obj.def_readwrite("PrintLevel", &PSIOPT::PrintLevel, PSIOPT_PrintLevel);
  obj.def("set_PrintLevel", &PSIOPT::set_PrintLevel);


  obj.def_readwrite("ConvergeFlag", &PSIOPT::ConvergeFlag);

  obj.def("get_ConvergenceFlag", &PSIOPT::get_ConvergenceFlag);


  obj.def_readwrite("KKTtol", &PSIOPT::KKTtol, PSIOPT_KKTtol);
  obj.def_readwrite("Bartol", &PSIOPT::Bartol, PSIOPT_Bartol);
  obj.def_readwrite("EContol", &PSIOPT::EContol, PSIOPT_EContol);
  obj.def_readwrite("IContol", &PSIOPT::IContol, PSIOPT_IContol);

  obj.def("set_KKTtol", &PSIOPT::set_KKTtol);
  obj.def("set_Bartol", &PSIOPT::set_Bartol);
  obj.def("set_EContol", &PSIOPT::set_EContol);
  obj.def("set_IContol", &PSIOPT::set_IContol);

  obj.def("set_tols",
          &PSIOPT::set_tols,
          py::arg("KKTtol") = 1.0e-6,
          py::arg("EContol") = 1.0e-6,
          py::arg("IContol") = 1.0e-6,
          py::arg("Bartol") = 1.0e-6);


  obj.def_readwrite("AccKKTtol", &PSIOPT::AccKKTtol, PSIOPT_AccKKTtol);
  obj.def_readwrite("AccBartol", &PSIOPT::AccBartol, PSIOPT_AccBartol);
  obj.def_readwrite("AccEContol", &PSIOPT::AccEContol, PSIOPT_AccEContol);
  obj.def_readwrite("AccIContol", &PSIOPT::AccIContol, PSIOPT_AccIContol);

  obj.def("set_AccKKTtol", &PSIOPT::set_AccKKTtol);
  obj.def("set_AccBartol", &PSIOPT::set_AccBartol);
  obj.def("set_AccEContol", &PSIOPT::set_AccEContol);
  obj.def("set_AccIContol", &PSIOPT::set_AccIContol);


  obj.def("set_Acctols",
          &PSIOPT::set_Acctols,
          py::arg("AccKKTtol") = 1.0e-2,
          py::arg("AccEContol") = 1.0e-3,
          py::arg("AccIContol") = 1.0e-3,
          py::arg("AccBartol") = 1.0e-3);


  obj.def_readwrite("DivKKTtol", &PSIOPT::DivKKTtol, PSIOPT_DivKKTtol);
  obj.def_readwrite("DivBartol", &PSIOPT::DivBartol, PSIOPT_DivBartol);
  obj.def_readwrite("DivEContol", &PSIOPT::DivEContol, PSIOPT_DivEContol);
  obj.def_readwrite("DivIContol", &PSIOPT::DivIContol, PSIOPT_DivIContol);

  obj.def("set_DivKKTtol", &PSIOPT::set_DivKKTtol);
  obj.def("set_DivBartol", &PSIOPT::set_DivBartol);
  obj.def("set_DivEContol", &PSIOPT::set_DivEContol);
  obj.def("set_DivIContol", &PSIOPT::set_DivIContol);


  obj.def_readwrite("NegSlackReset", &PSIOPT::NegSlackReset, PSIOPT_NegSlackReset);

  obj.def_readwrite("BoundFraction", &PSIOPT::BoundFraction, PSIOPT_BoundFraction);
  obj.def("set_BoundFraction", &PSIOPT::set_BoundFraction);

  obj.def_readwrite("BoundPush", &PSIOPT::BoundPush, PSIOPT_BoundPush);

  /////////////////////////////////////////////////////////////

  obj.def_readwrite("deltaH", &PSIOPT::deltaH, PSIOPT_deltaH);
  obj.def_readwrite("incrH", &PSIOPT::incrH, PSIOPT_incrH);
  obj.def_readwrite("decrH", &PSIOPT::decrH, PSIOPT_decrH);

  obj.def("set_deltaH", &PSIOPT::set_deltaH);
  obj.def("set_incrH", &PSIOPT::set_incrH);
  obj.def("set_decrH", &PSIOPT::set_decrH);

  obj.def("set_HpertParams", &PSIOPT::set_HpertParams, py::arg("deltaH"), py::arg("incrH"), py::arg("decrH"));


  /////////////////////////////////////////////////////////////
  obj.def_readwrite("initMu", &PSIOPT::initMu, PSIOPT_initMu);
  obj.def_readwrite("MinMu", &PSIOPT::MinMu, PSIOPT_MinMu);
  obj.def_readwrite("MaxMu", &PSIOPT::MaxMu, PSIOPT_MaxMu);

  obj.def_readwrite("MaxSOC", &PSIOPT::MaxSOC, PSIOPT_MaxSOC);


  obj.def_readwrite("PDStepStrategy", &PSIOPT::PDStepStrategy, PSIOPT_PDStepStrategy);
  obj.def_readwrite("SOEboundRelax", &PSIOPT::SOEboundRelax, PSIOPT_SOEboundRelax);
  obj.def_readwrite("QPParSolve", &PSIOPT::QPParSolve, PSIOPT_QPParSolve);

  obj.def_readwrite("SoeMode", &PSIOPT::SoeMode, PSIOPT_SoeMode);

  //////////////////////////////////////////////////////////////////////////////////////////////////

  obj.def_readwrite("OptBarMode", &PSIOPT::OptBarMode, PSIOPT_OptBarMode);
  obj.def_readwrite("SoeBarMode", &PSIOPT::SoeBarMode, PSIOPT_SoeBarMode);

  obj.def("set_OptBarMode", py::overload_cast<BarrierModes>(&PSIOPT::set_OptBarMode));
  obj.def("set_OptBarMode", py::overload_cast<const std::string&>(&PSIOPT::set_OptBarMode));
  obj.def("set_SoeBarMode", py::overload_cast<BarrierModes>(&PSIOPT::set_SoeBarMode));
  obj.def("set_SoeBarMode", py::overload_cast<const std::string&>(&PSIOPT::set_SoeBarMode));


  //////////////////////////////////////////////////////////////////////////////////////////////////
  obj.def_readwrite("OptLSMode", &PSIOPT::OptLSMode, PSIOPT_OptLSMode);
  obj.def_readwrite("SoeLSMode", &PSIOPT::SoeLSMode, PSIOPT_SoeLSMode);

  obj.def("set_OptLSMode", py::overload_cast<LineSearchModes>(&PSIOPT::set_OptLSMode));
  obj.def("set_OptLSMode", py::overload_cast<const std::string&>(&PSIOPT::set_OptLSMode));
  obj.def("set_SoeLSMode", py::overload_cast<LineSearchModes>(&PSIOPT::set_SoeLSMode));
  obj.def("set_SoeLSMode", py::overload_cast<const std::string&>(&PSIOPT::set_SoeLSMode));

  //////////////////////////////////////////////////////////////////////////////////////////////////

  obj.def_readwrite("ForceQPanalysis", &PSIOPT::ForceQPanalysis, PSIOPT_ForceQPanalysis);
  obj.def_readwrite("QPRefSteps", &PSIOPT::QPRefSteps, PSIOPT_QPRefSteps);

  obj.def_readwrite("QPPivotPerturb", &PSIOPT::QPPivotPerturb, PSIOPT_QPPivotPerturb);
  obj.def_readwrite("QPThreads", &PSIOPT::QPThreads, PSIOPT_QPThreads);
  obj.def_readwrite("QPPivotStrategy", &PSIOPT::QPPivotStrategy, PSIOPT_QPPivotStrategy);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  obj.def_readwrite("QPOrderingMode", &PSIOPT::QPOrd, PSIOPT_QPOrd);

  obj.def("set_QPOrderingMode", py::overload_cast<QPOrderingModes>(&PSIOPT::set_QPOrderingMode));
  obj.def("set_QPOrderingMode", py::overload_cast<const std::string&>(&PSIOPT::set_QPOrderingMode));

  //////////////////////////////////////////////////////////////////////////////////////////////////
  obj.def_readwrite("QPPrint", &PSIOPT::QPPrint);

  obj.def_readwrite("Diagnostic", &PSIOPT::Diagnostic);


  obj.def_readwrite("storespmat", &PSIOPT::storespmat, PSIOPT_storespmat);
  obj.def("getSPmat", &PSIOPT::getSPmat, PSIOPT_getSPmat);
  obj.def("getSPmat2", &PSIOPT::getSPmat2, PSIOPT_getSPmat2);

  obj.def_readwrite("CNRMode", &PSIOPT::CNRMode, PSIOPT_CNRMode);


  py::enum_<BarrierModes>(m, "BarrierModes")
      .value("PROBE", BarrierModes::PROBE)
      .value("LOQO", BarrierModes::LOQO);
  py::enum_<LineSearchModes>(m, "LineSearchModes")
      .value("AUGLANG", LineSearchModes::AUGLANG)
      .value("LANG", LineSearchModes::LANG)
      .value("L1", LineSearchModes::L1)
      .value("NOLS", LineSearchModes::NOLS);
  py::enum_<QPPivotModes>(m, "QPPivotModes")
      .value("OneByOne", QPPivotModes::OneByOne)
      .value("TwoByTwo", QPPivotModes::TwoByTwo);
  py::enum_<PDStepStrategies>(m, "PDStepStrategies")
      .value("PrimSlackEq_Iq", PDStepStrategies::PrimSlackEq_Iq)
      .value("AllMinimum", PDStepStrategies::AllMinimum)
      .value("PrimSlack_EqIq", PDStepStrategies::PrimSlack_EqIq)
      .value("MaxEq", PDStepStrategies::MaxEq);
  py::enum_<ConvergenceFlags>(m, "ConvergenceFlags", py::arithmetic())
      .value("CONVERGED", ConvergenceFlags::CONVERGED)
      .value("ACCEPTABLE", ConvergenceFlags::ACCEPTABLE)
      .value("NOTCONVERGED", ConvergenceFlags::NOTCONVERGED)
      .value("DIVERGING", ConvergenceFlags::DIVERGING);
  py::enum_<AlgorithmModes>(m, "AlgorithmModes")
      .value("OPT", AlgorithmModes::OPT)
      .value("OPTNO", AlgorithmModes::OPTNO)
      .value("SOE", AlgorithmModes::SOE)
      .value("INIT", AlgorithmModes::INIT);

  py::enum_<QPOrderingModes>(m, "QPOrderingModes")
      .value("MINDEG", QPOrderingModes::MINDEG)
      .value("METIS", QPOrderingModes::METIS)
      .value("PARMETIS", QPOrderingModes::PARMETIS);
}
