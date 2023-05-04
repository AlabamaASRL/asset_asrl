#pragma once
#include "Solvers/NonLinearProgram.h"
#include "Solvers/PSIOPT.h"
#include "pch.h"


namespace ASSET {


  struct OptimizationProblemBase {


    enum JetJobModes {
      NotSet,
      DoNothing,
      Solve,
      Optimize,
      SolveOptimize,
      SolveOptimizeSolve,
      OptimizeSolve
    };

    int Threads = ASSET_DEFAULT_FUNC_THREADS;
    int JetJobMode = JetJobModes::NotSet;

    std::shared_ptr<NonLinearProgram> nlp;
    std::shared_ptr<PSIOPT> optimizer;


    virtual ~OptimizationProblemBase() = default;

    OptimizationProblemBase() {
      this->optimizer = std::make_shared<PSIOPT>();
      this->initThreads();  // must called after initing optimizer
    }

    virtual PSIOPT::ConvergenceFlags solve() = 0;
    virtual PSIOPT::ConvergenceFlags optimize() = 0;
    virtual PSIOPT::ConvergenceFlags solve_optimize() = 0;
    virtual PSIOPT::ConvergenceFlags solve_optimize_solve() = 0;
    virtual PSIOPT::ConvergenceFlags optimize_solve() = 0;


    virtual void initThreads() {
      this->Threads = std::min(ASSET_DEFAULT_FUNC_THREADS, int(std::thread::hardware_concurrency()));
      this->optimizer->QPThreads = std::min(ASSET_DEFAULT_QP_THREADS, get_core_count());
      ;
    }

    virtual void setThreads(int functhreads, int qpthreads) {
      if (functhreads < 1 || qpthreads < 1) {
        throw std::invalid_argument("Number of threads must be positive");
      }
      this->Threads = functhreads;
      this->optimizer->QPThreads = qpthreads;
    }
    virtual void setThreads(int functhreads) {
      if (functhreads < 1) {
        throw std::invalid_argument("Number of threads must be positive");
      }
      this->Threads = functhreads;
    }

    virtual void jet_initialize() = 0;
    virtual void jet_release() = 0;

    virtual PSIOPT::ConvergenceFlags jet_run() {
      this->jet_initialize();

      PSIOPT::ConvergenceFlags flag;

      switch (this->JetJobMode) {
        case JetJobModes::Solve: {
          flag = this->solve();
          break;
        }
        case JetJobModes::Optimize: {
          flag = this->optimize();
          break;
        }
        case JetJobModes::SolveOptimize: {
          flag = this->solve_optimize();
          break;
        }
        case JetJobModes::SolveOptimizeSolve: {
          flag = this->solve_optimize_solve();
          break;
        }
        case JetJobModes::OptimizeSolve: {
          flag = this->optimize_solve();
          break;
        }
        case NotSet: {
          throw ::std::invalid_argument("JetJobMode not set");
          break;
        }
        default:
          break;
          flag = PSIOPT::ConvergenceFlags::CONVERGED;
      }

      this->jet_release();
      return flag;
    }


    static JetJobModes strto_JetJobMode(const std::string& str) {

      if (str == "solve" || str == "Solve")
        return JetJobModes::Solve;
      else if (str == "optimize" || str == "Optimize")
        return JetJobModes::Optimize;
      else if (str == "solve_optimize" || str == "SolveOptimize" || str == "Solve_Optimize")
        return JetJobModes::SolveOptimize;
      else if (str == "solve_optimize_solve" || str == "SolveOptimizeSolve" || str == "Solve_Optimize_Solve")
        return JetJobModes::SolveOptimizeSolve;
      else if (str == "optimize_solve" || str == "OptimizeSolve" || str == "Optimize_Solve")
        return JetJobModes::OptimizeSolve;
      else if (str == "DoNothing" || str == "do_nothing" || str == "Do_Nothing")
        return JetJobModes::DoNothing;
      else {
        auto msg = fmt::format("Unrecognized JetJobMode: {0}\n", str);
        throw std::invalid_argument(msg);
        return JetJobModes::NotSet;
      }
    }

    void setJetJobMode(JetJobModes m) {
      this->JetJobMode = m;
    }
    void setJetJobMode(const std::string& str) {
      this->setJetJobMode(strto_JetJobMode(str));
    }

    static void Build(py::module& m);
  };


}  // namespace ASSET
