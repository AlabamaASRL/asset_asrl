
#pragma once
#include "IterateInfo.h"
#include "NonLinearProgram.h"
#include "PardisoInterface.h"
#include "Utils/ColorText.h"
#include "pch.h"

namespace ASSET {

  struct IterateInfo;

  struct PSIOPT {


    enum BarrierModes {
      PROBE,
      LOQO,
      FIACCO,
      BARDISABLED
    };
    enum LineSearchModes {
      AUGLANG,
      LANG,
      L1,
      L2,
      NOLS
    };
    enum AlgorithmModes {
      OPT,
      OPTNO,
      SOE,
      INIT
    };
    enum ConvergenceFlags {
      CONVERGED,
      ACCEPTABLE,
      NOTCONVERGED,
      DIVERGING,
    };

    enum QPAlgModes {
      Classic = 0,
      TwoLevel = 1,
    };

    enum QPOrderingModes {
      MINDEG = 0,
      METIS = 2,
      PARMETIS = 3
    };

    enum QPPivotModes {
      OneByOne = 0,
      TwoByTwo = 1,
      E4 = 4,
      E6 = 6,
      E8 = 8,
      E13 = 13,
    };
    enum PDStepStrategies {
      PrimSlackEq_Iq,
      AllMinimum,
      PrimSlack_EqIq,
      MaxEq
    };


    static QPOrderingModes strto_OrderingMode(const std::string& str) {

      if (str.compare("MINDEG") == 0)
        return MINDEG;
      else if (str.compare("METIS") == 0)
        return METIS;
      else if (str.compare("PARMETIS") == 0)
        return PARMETIS;
      else {
        auto msg = fmt::format("Unrecognized QPOrderingMode: {0}\n"
                               "Valid Options Are: MINDEG , METIS, PARMETIS ",
                               str);
        throw std::invalid_argument(msg);
        return MINDEG;
      }
    }
    static LineSearchModes strto_LineSearchMode(const std::string& str) {

      if (str.compare("L1") == 0)
        return L1;
      else if (str.compare("NOLS") == 0)
        return NOLS;
      else if (str.compare("LANG") == 0)
        return LANG;
      else if (str.compare("AUGLANG") == 0)
        return AUGLANG;
      else {
        auto msg = fmt::format("Unrecognized LineSearchMode: {0}\n"
                               "Valid Options Are: AUGLANG, LANG, L1, NOLS ",
                               str);
        throw std::invalid_argument(msg);
        return L1;
      }
    }
    static BarrierModes strto_BarrierMode(const std::string& str) {

      if (str.compare("PROBE") == 0)
        return PROBE;
      else if (str.compare("LOQO") == 0)
        return LOQO;
      else {
        auto msg = fmt::format("Unrecognized BarrierMode: {0}\n"
                               "Valid Options Are: LOQO, PROBE ",
                               str);
        throw std::invalid_argument(msg);
        return LOQO;
      }
    }


    using VectorXd = Eigen::VectorXd;
    std::shared_ptr<NonLinearProgram> nlp;

    int PrimalVars = 0;
    int SlackVars = 0;
    int EqualCons = 0;
    int InequalCons = 0;
    int KKTdim = 0;

    Eigen::PardisoLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper> KKTSol;

    int QPThreads = ASSET_DEFAULT_QP_THREADS;
    QPAlgModes QPAlg = QPAlgModes::Classic;
    QPOrderingModes QPOrd = QPOrderingModes::METIS;
    QPPivotModes QPPivotStrategy = QPPivotModes::TwoByTwo;

    void set_QPOrderingMode(QPOrderingModes mode) {
      this->QPOrd = mode;
    }
    void set_QPOrderingMode(const std::string& str) {
      this->QPOrd = strto_OrderingMode(str);
    }


    int QPMatching = 1;
    int QPScaling = 0;
    int QPPivotPerturb = 8;
    int QPRefSteps = 0;
    bool QPPrint = false;
    bool QPanalyzed = false;
    bool ForceQPanalysis = false;
    bool Diagnostic = false;
    int QPParSolve = 0;


    /////////////////////////////////////////////////////////////////////
    int MaxIters = 500;
    int MaxLSIters = 2;
    int MaxAccIters = 50;

    void set_MaxIters(int MaxIters) {
      if (MaxIters < 1) {
        throw std::invalid_argument("MaxIters must be greater than 0.");
      }
      this->MaxIters = MaxIters;
    }
    void set_MaxAccIters(int MaxAccIters) {
      if (MaxAccIters < 1) {
        throw std::invalid_argument("MaxAccIters must be greater than 0.");
      }
      this->MaxAccIters = MaxAccIters;
    }
    void set_MaxLSIters(int MaxLSIters) {
      if (MaxLSIters < 0) {
        throw std::invalid_argument("MaxLSIters must be positive.");
      }
      this->MaxLSIters = MaxLSIters;
    }
    void set_AllMaxIters(int m1, int m2) {
      set_MaxIters(m1);
      set_MaxAccIters(m2);
    }


    int MaxRefac = 15;

    int MaxSOC = 1;
    int MaxFeasRest = 2;

    AlgorithmModes SoeMode = AlgorithmModes::SOE;

    BarrierModes OptBarMode = BarrierModes::LOQO;
    BarrierModes SoeBarMode = BarrierModes::LOQO;

    void set_OptBarMode(BarrierModes mode) {
      this->OptBarMode = mode;
    }
    void set_OptBarMode(const std::string& str) {
      this->OptBarMode = strto_BarrierMode(str);
    }
    void set_SoeBarMode(BarrierModes mode) {
      this->SoeBarMode = mode;
    }
    void set_SoeBarMode(const std::string& str) {
      this->SoeBarMode = strto_BarrierMode(str);
    }


    LineSearchModes OptLSMode = LineSearchModes::NOLS;
    LineSearchModes SoeLSMode = LineSearchModes::NOLS;

    void set_OptLSMode(LineSearchModes mode) {
      this->OptLSMode = mode;
    }
    void set_OptLSMode(const std::string& str) {
      this->OptLSMode = strto_LineSearchMode(str);
    }
    void set_SoeLSMode(LineSearchModes mode) {
      this->SoeLSMode = mode;
    }
    void set_SoeLSMode(const std::string& str) {
      this->SoeLSMode = strto_LineSearchMode(str);
    }


    double MaxCPUtime = 1200;
    double ObjScale = 1.0;

    /////////////////////////////////////////////////////////////////////////
    double KKTtol = 1.0e-6;
    double EContol = 1.0e-6;
    double IContol = 1.0e-6;
    double Bartol = 1.0e-6;

    void set_KKTtol(double KKTtol) {
      this->KKTtol = std::abs(KKTtol);
    }
    void set_Bartol(double Bartol) {
      this->Bartol = std::abs(Bartol);
    }
    void set_EContol(double EContol) {
      this->EContol = std::abs(EContol);
    }
    void set_IContol(double IContol) {
      this->IContol = std::abs(IContol);
    }
    void set_tols(double KKTtol, double EContol, double IContol, double Bartol) {
      this->set_KKTtol(KKTtol);
      this->set_EContol(EContol);
      this->set_IContol(IContol);
      this->set_Bartol(Bartol);
    }

    double AccKKTtol = 1.0e-2;
    double AccEContol = 1.0e-3;
    double AccIContol = 1.0e-3;
    double AccBartol = 1.0e-3;

    void set_AccKKTtol(double AccKKTtol) {
      this->AccKKTtol = std::abs(AccKKTtol);
    }
    void set_AccBartol(double AccBartol) {
      this->AccBartol = std::abs(AccBartol);
    }
    void set_AccEContol(double AccEContol) {
      this->AccEContol = std::abs(AccEContol);
    }
    void set_AccIContol(double AccIContol) {
      this->AccIContol = std::abs(AccIContol);
    }
    void set_Acctols(double AccKKTtol, double AccEContol, double AccIContol, double AccBartol) {
      this->set_AccKKTtol(AccKKTtol);
      this->set_AccEContol(AccEContol);
      this->set_AccIContol(AccIContol);
      this->set_AccBartol(AccBartol);
    }

    double UnAccKKTtol = 10;
    double UnAccEContol = 2;
    double UnAccIContol = 2;
    double UnAccBartol = 2;

    void set_UnAcctols(double kktol, double etol, double itol, double bartol) {
      this->UnAccKKTtol = kktol;
      this->UnAccBartol = bartol;
      this->UnAccEContol = etol;
      this->UnAccIContol = itol;
    }

    double DivKKTtol = 1.0e15;
    double DivEContol = 1.0e15;
    double DivIContol = 1.0e15;
    double DivBartol = 1.0e15;

    void set_DivKKTtol(double DivKKTtol) {
      this->DivKKTtol = std::abs(DivKKTtol);
    }
    void set_DivBartol(double DivBartol) {
      this->DivBartol = std::abs(DivBartol);
    }
    void set_DivEContol(double DivEContol) {
      this->DivEContol = std::abs(DivEContol);
    }
    void set_DivIContol(double DivIContol) {
      this->DivIContol = std::abs(DivIContol);
    }
    void set_Divtols(double DivKKTtol, double DivEContol, double DivIContol, double DivBartol) {
      this->set_DivKKTtol(DivKKTtol);
      this->set_DivEContol(DivEContol);
      this->set_DivIContol(DivIContol);
      this->set_DivBartol(DivBartol);
    }


    /////////////////////////////////////////////////////////////////////////

    double ExObjVal = -1.0e20;


    double BoundFraction = 0.98;
    void set_BoundFraction(double BoundFraction) {
      if (BoundFraction >= 1.0 || BoundFraction <= 0.0) {
        throw std::invalid_argument("BoundFraction must be between 0 and 1.");
      }
      this->BoundFraction = BoundFraction;
    }

    double BoundPush = 1.0e-3;
    void set_BoundPush(double BoundPush) {
      if (BoundPush <= 0.0) {
        throw std::invalid_argument("BoundPush must be greater than 0.");
      }
      this->BoundPush = BoundPush;
    }

    double NegSlackReset = 1.0e-12;

    double SOEboundRelax = 1.0e-8;
    double minLSstep = .01;
    double alphaRed = 2.0;

    void set_alphaRed(double ared) {
      if (ared <= 1.0) {
        throw std::invalid_argument("alphaRed must be greater than 1.0");
      }
      this->alphaRed = ared;
    }

    /////////////////////////////////////////////////////////////////////////
    double deltaH = 1.0e-5;
    double incrH = 8.00;
    double decrH = 0.333333;

    void set_deltaH(double deltaH) {
      if (deltaH <= 0.0) {
        throw std::invalid_argument("deltaH must be greater than 0.");
      }
      this->deltaH = deltaH;
    }
    void set_incrH(double incrH) {
      if (incrH <= 1.0) {
        throw std::invalid_argument("incrH must  greater than 1.0.");
      }
      this->incrH = incrH;
    }
    void set_decrH(double decrH) {
      if (decrH >= 1.0 || decrH <= 0) {
        throw std::invalid_argument("decrH must be between 0 and 1.");
      }
      this->decrH = decrH;
    }
    void set_HpertParams(double deltaH, double incrH, double decrH) {
      this->set_deltaH(deltaH);
      this->set_incrH(incrH);
      this->set_decrH(decrH);
    }
    /////////////////////////////////////////////////////////////////////////
    ConvergenceFlags ConvergeFlag = ConvergenceFlags::NOTCONVERGED;
    ConvergenceFlags get_ConvergenceFlag() const {
      return this->ConvergeFlag;
    }


    double initMu = 0.001;
    double MaxMu = 100.0;
    double MinMu = 1.0e-12;

    bool CNRMode = false;
    int PrintLevel = 0;
    void set_PrintLevel(int plevel) {
      this->PrintLevel = plevel;
    }

    PDStepStrategies PDStepStrategy = PrimSlackEq_Iq;
    bool storespmat = false;
    Eigen::SparseMatrix<double, Eigen::RowMajor> spmat;
    double LastObjVal = 0.0;
    bool FastFactorAlg = true;

    double LastTotalTime = 0;
    double LastPreTime = 0;
    double LastMiscTime = 0;
    double LastFuncTime = 0;
    double LastKKTTime = 0;
    int LastIterNum = 0;

    void zero_timing_stats() {
      this->LastTotalTime = 0;
      this->LastPreTime = 0;
      this->LastMiscTime = 0;
      this->LastFuncTime = 0;
      this->LastKKTTime = 0;
      this->LastIterNum = 0;
    }


    bool WideConsole = false;
    int FactorMem = 0;
    int FactorFlops = 0;
    Eigen::VectorXd LastEqLmults;
    Eigen::VectorXd LastIqLmults;

    Eigen::VectorXd LastEqCons;
    Eigen::VectorXd LastIqCons;



    /////////////////////////////////////////////////////////////////////

    using EarlyCallBackType = std::function<int(int,
                                                double,
                                                EigenRef<VectorXd>,
                                                double,
                                                EigenRef<VectorXd>,
                                                EigenRef<VectorXd>,
                                                Eigen::SparseMatrix<double, Eigen::RowMajor>&)>;

    using LateCallBackType =
        std::function<int(const IterateInfo&, ConstEigenRef<VectorXd>, ConstEigenRef<VectorXd>)>;

    EarlyCallBackType EarlyCallBack;  // = [](int i, EigenRef<VectorXd> XSL, EigenRef<VectorXd>
                                      // GX, EigenRef<VectorXd> AGXFX) {return 0; };
    bool EarlyCallBackEnabled = false;
    LateCallBackType LateCallBack;  // = [](const IterateInfo& i, EigenRef<VectorXd> XSL,
                                    // EigenRef<VectorXd> AGXFX) {return 0; };
    bool LateCallBackEnabled = false;

    ////////////////////////////////////////////////////////////////////

    PSIOPT() {
      this->QPThreads = std::min(ASSET_DEFAULT_QP_THREADS, get_core_count());
    }
    PSIOPT(std::shared_ptr<NonLinearProgram> np) {
      this->QPThreads = std::min(ASSET_DEFAULT_QP_THREADS, get_core_count());
      this->setNLP(np);
    }

    void release() {
      this->KKTSol.release();
      this->QPanalyzed = false;
      this->nlp = std::shared_ptr<NonLinearProgram>();
      this->LastEqLmults.resize(0);
      this->LastIqLmults.resize(0);
    }


    void setNLP(std::shared_ptr<NonLinearProgram> np);
    Eigen::MatrixXd getSPmat() {
      return this->spmat.toDense();
    }
    Eigen::MatrixXd getSPmat2() {
      return this->KKTSol.getMatrixTwisted(this->spmat);
    }

    void setQPParams() {
      this->KKTSol.m_ord = QPOrd;
      this->KKTSol.m_pivotstrat = QPPivotStrategy;
      this->KKTSol.m_pivotpert = QPPivotPerturb;
      this->KKTSol.m_matching = QPMatching;
      this->KKTSol.m_scaling = QPScaling;
      this->KKTSol.m_iterref = QPRefSteps;
      this->KKTSol.m_alg = QPAlg;
      this->KKTSol.m_msglvl = QPPrint;

      if (this->CNRMode)
        this->KKTSol.m_threads = this->QPThreads;
      this->KKTSol.m_parsolve = this->QPParSolve;
      // mkl_set_num_threads(QPThreads);
      this->KKTSol.setParams();
    }

    void set_early_callback(const EarlyCallBackType& f) {
      this->EarlyCallBackEnabled = true;
      this->EarlyCallBack = f;
    }
    void disable_early_callback() {
      this->EarlyCallBackEnabled = false;
    }
    void set_late_callback(const LateCallBackType& f) {
      this->LateCallBackEnabled = true;
      this->LateCallBack = f;
    }
    void disable_late_callback() {
      this->LateCallBackEnabled = false;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> getPrimals(EigenRef<VectorXd> XSL) const {
      return XSL.head(this->PrimalVars);
    }
    EigenRef<VectorXd> getSlacks(EigenRef<VectorXd> XSL) const {
      return XSL.segment(this->PrimalVars, this->SlackVars);
    }
    EigenRef<VectorXd> getPrimalsSlacks(EigenRef<VectorXd> XSL) const {
      return XSL.head(this->PrimalVars + this->SlackVars);
    }

    EigenRef<VectorXd> getLmults(EigenRef<VectorXd> XSL) const {
      return XSL.tail(this->EqualCons + this->InequalCons);
    }

    EigenRef<VectorXd> getEqLmults(EigenRef<VectorXd> XSL) const {
      return XSL.segment(this->PrimalVars + this->SlackVars, this->EqualCons);
    }
    EigenRef<VectorXd> getIqLmults(EigenRef<VectorXd> XSL) const {
      return XSL.tail(this->InequalCons);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////
    EigenRef<VectorXd> getPrimGrad(EigenRef<VectorXd> GX_or_AGX_FX) const {
      return GX_or_AGX_FX.head(this->PrimalVars);
    }
    EigenRef<VectorXd> getDualGrad(EigenRef<VectorXd> AGX_FX) const {
      return AGX_FX.segment(this->PrimalVars, this->SlackVars);
    }
    EigenRef<VectorXd> getPrimDualGrad(EigenRef<VectorXd> AGX_FX) const {
      return AGX_FX.head(this->PrimalVars + this->SlackVars);
    }
    EigenRef<VectorXd> getEqCons(EigenRef<VectorXd> AGX_FX) const {
      return AGX_FX.segment(this->PrimalVars + this->SlackVars, this->EqualCons);
    }
    EigenRef<VectorXd> getIqCons(EigenRef<VectorXd> AGX_FX) const {
      return AGX_FX.tail(this->InequalCons);
    }
    EigenRef<VectorXd> getAllCons(EigenRef<VectorXd> AGX_FX) const {
      return AGX_FX.tail(this->InequalCons + this->EqualCons);
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    void apply_reset_slacks(Eigen::Ref<Eigen::VectorXd> S, Eigen::Ref<Eigen::VectorXd> FXI) const {
      for (int i = 0; i < this->SlackVars; i++) {
        double fxi = FXI[i];
        double si = S[i];
        if (si < NegSlackReset) {
          si = NegSlackReset;
        }

        if (fxi < 0.0) {
          FXI[i] = 0.0;
          S[i] = std::max(std::abs(fxi), NegSlackReset);
        } else {
          FXI[i] += si;
        }
      }
    }
    double max_step_to_boundary(Eigen::Ref<Eigen::VectorXd> SLI,
                                Eigen::Ref<Eigen::VectorXd> dSLI,
                                double bfrac) const {
      double alpha = 1.0;
      for (int i = 0; i < this->InequalCons; i++) {
        if (dSLI[i] < -bfrac * SLI[i]) {
          double an = -bfrac * SLI[i] / dSLI[i];
          if (an < alpha)
            alpha = an;
        }
      }
      return alpha;
    }
    void complementarity(Eigen::Ref<Eigen::VectorXd> S,
                         Eigen::Ref<Eigen::VectorXd> LI,
                         double& avgcomp,
                         double& mincomp,
                         double& maxcomp) const {
      Eigen::VectorXd StLI = S.cwiseProduct(LI);
      mincomp = StLI.minCoeff();
      maxcomp = StLI.maxCoeff();
      avgcomp = StLI.sum() / double(StLI.size());
    }

    double barrier_objective(Eigen::Ref<Eigen::VectorXd> S, double mu) const {
      double psi = 0;
      for (int i = 0; i < this->InequalCons; i++) {
        psi += -mu * std::log(S[i]);
      }
      return psi;
    }
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> S,
                          Eigen::Ref<Eigen::VectorXd> LI,
                          double mu,
                          Eigen::Ref<Eigen::VectorXd> AGS) const {
      AGS = LI - mu * (S.cwiseInverse());
    }
    void barrier_gradient(Eigen::Ref<Eigen::VectorXd> LI, Eigen::Ref<Eigen::VectorXd> AGS) const {
      AGS = LI;
    }

    void barrier_hessian(Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat,
                         Eigen::Ref<Eigen::VectorXd> S,
                         Eigen::Ref<Eigen::VectorXd> LI,
                         double mu) {
      Eigen::VectorXd hp = LI.cwiseQuotient(S);
      for (int i = 0; i < this->InequalCons; i++) {
        if (hp[i] < 0.0) {
          hp[i] = mu / (S[i] * S[i]);
        }
      }
      this->nlp->assignKKTSlackHessian(hp, KKTmat);
    }

    double LOQOMu(Eigen::Ref<Eigen::VectorXd> S,
                  Eigen::Ref<Eigen::VectorXd> LI,
                  double avgcomp,
                  double mincomp) const {
      double eta = mincomp / avgcomp;
      double sigmat = .1 * std::pow(0.05 * (1.0 - eta) / eta, 3);
      double sigma = std::min(0.8, abs(sigmat));
      return sigma * avgcomp;
    }
    double MPCMu(Eigen::Ref<Eigen::VectorXd> S,
                 Eigen::Ref<Eigen::VectorXd> LI,
                 double avgcomp,
                 double mincomp) const {
      double navgcomp = 0;
      double nmincomp = 0;
      double nmaxcomp = 0;
      this->complementarity(S, LI, navgcomp, nmincomp, nmaxcomp);
      return std::pow(navgcomp / avgcomp, 3) * avgcomp;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    void evalKKT(double ObjScale,
                 ConstEigenRef<VectorXd> XSL,
                 double& val,
                 EigenRef<VectorXd> GX,
                 EigenRef<VectorXd> AGXS_FX,
                 Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat) {
      this->nlp->evalKKT(ObjScale,
                         XSL.head(this->PrimalVars),
                         XSL.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                         XSL.tail(this->InequalCons),
                         val,
                         this->getPrimGrad(GX),
                         this->getPrimGrad(AGXS_FX),
                         this->getEqCons(AGXS_FX),
                         this->getIqCons(AGXS_FX),
                         KKTmat);
    }

    void evalKKTNO(double ObjScale,
                   ConstEigenRef<VectorXd> XSL,
                   double& val,
                   EigenRef<VectorXd> GX,
                   EigenRef<VectorXd> AGXS_FX,
                   Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat) {
      this->nlp->evalKKTNO(ObjScale,
                           XSL.head(this->PrimalVars),
                           XSL.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                           XSL.tail(this->InequalCons),
                           val,
                           this->getPrimGrad(GX),
                           this->getPrimGrad(AGXS_FX),
                           this->getEqCons(AGXS_FX),
                           this->getIqCons(AGXS_FX),
                           KKTmat);
    }

    void evalAUG(double ObjScale,
                 ConstEigenRef<VectorXd> XSL,
                 double& val,
                 EigenRef<VectorXd> GX,
                 EigenRef<VectorXd> AGXS_FX,
                 Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat) {
      this->nlp->evalAUG(ObjScale,
                         XSL.head(this->PrimalVars),
                         XSL.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                         XSL.tail(this->InequalCons),
                         val,
                         this->getPrimGrad(GX),
                         this->getPrimGrad(AGXS_FX),
                         this->getEqCons(AGXS_FX),
                         this->getIqCons(AGXS_FX),
                         KKTmat);
    }

    void evalSOE(double ObjScale,
                 ConstEigenRef<VectorXd> XSL,
                 double& val,
                 EigenRef<VectorXd> GX,
                 EigenRef<VectorXd> AGXS_FX,
                 Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat) {
      this->nlp->evalSOE(ObjScale,
                         XSL.head(this->PrimalVars),
                         XSL.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                         XSL.tail(this->InequalCons),
                         val,
                         this->getPrimGrad(GX),
                         this->getPrimGrad(AGXS_FX),
                         this->getEqCons(AGXS_FX),
                         this->getIqCons(AGXS_FX),
                         KKTmat);
    }

    void evalRHS(double ObjScale,
                 const Eigen::Ref<const Eigen::VectorXd>& XSL,
                 double& val,
                 Eigen::Ref<Eigen::VectorXd> GX,
                 Eigen::Ref<Eigen::VectorXd> AGXS_FX) {
      this->nlp->evalRHS(ObjScale,
                         XSL.head(this->PrimalVars),
                         XSL.segment(this->PrimalVars + this->SlackVars, this->EqualCons),
                         XSL.tail(this->InequalCons),
                         val,
                         this->getPrimGrad(GX),
                         this->getPrimGrad(AGXS_FX),
                         this->getEqCons(AGXS_FX),
                         this->getIqCons(AGXS_FX));
    }

    void max_primal_dual_step(Eigen::Ref<Eigen::VectorXd> XSL,
                              Eigen::Ref<Eigen::VectorXd> DXSL,
                              double bfrac,
                              double& alphap,
                              double& alphad);

    void fill_iter_info(Eigen::Ref<Eigen::VectorXd> XSL,
                        Eigen::Ref<Eigen::VectorXd> RHS,
                        double pobj,
                        double bobj,
                        double mu,
                        IterateInfo& iter) const;

    void evalNLP(int algmode,
                 double ObjScale,
                 ConstEigenRef<VectorXd> XSL,
                 double& val,
                 EigenRef<VectorXd> GX,
                 EigenRef<VectorXd> AGXS_FX,
                 Eigen::SparseMatrix<double, Eigen::RowMajor>& KKTmat);

    ConvergenceFlags convergeCheck(std::vector<IterateInfo>& iters);

    static void printPSIOPT();

    void print_settings();
    void print_matrixinfo();
    void print_stats();
    void print_last_iterate(const std::vector<IterateInfo>& iters);

    static void print_Header() {
      fmt::print(fmt::fg(fmt::color::white), "{0:=^{1}}\n", "", 65);
    }
    void print_Beginning(std::string msg) const;
    void print_Finished(std::string msg) const;
    void print_ExitStats(ConvergenceFlags ExitCode,
                         const std::vector<IterateInfo>& iters,
                         double tottime,
                         double nlptime,
                         double qptime);

    fmt::text_style calculate_color(double val, double targ, double acc);

    int factor_impl(
        bool docompute, bool ZFac, double ipurt, double incpurt0, double incpurt, double& finalpert);

    bool analyze_KKT_Matrix() {
      bool docompute = true;
      if (this->QPanalyzed && !(this->ForceQPanalysis)) {
        docompute = false;
      } else {
        this->QPanalyzed = true;
        docompute = true;
      }
      return docompute;
    }

    Eigen::VectorXd alg_impl(AlgorithmModes algmode,
                             BarrierModes barmode,
                             LineSearchModes lsmode,
                             double ObjScale,
                             double MuI,
                             Eigen::Ref<Eigen::VectorXd> xsl);


    Eigen::VectorXd init_impl(const Eigen::VectorXd& x, double Mu, bool docompute);


    double ls_impl(LineSearchModes lsmode,
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
                   const std::vector<IterateInfo>& iters);


    Eigen::VectorXd optimize(const Eigen::VectorXd& x);

    Eigen::VectorXd solve_optimize(const Eigen::VectorXd& x);

    Eigen::VectorXd solve_optimize_solve(const Eigen::VectorXd& x);

    Eigen::VectorXd optimize_solve(const Eigen::VectorXd& x);


    Eigen::VectorXd solve(const Eigen::VectorXd& x);

    static void Build(py::module& m);
  };

}  // namespace ASSET
