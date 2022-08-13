#pragma once
#include "pch.h"
#include "Solvers/NonLinearProgram.h"
#include "Solvers/PSIOPT.h"


namespace ASSET {


	struct OptimizationProblemBase {


		enum JetJobModes {
			NotSet,
			DoNothing,
			Solve,
			Optimize,
			SolveOptimize,
			SolveOptimizeSolve,
		};

		int Threads = ASSET_DEFAULT_FUNC_THREADS;
		int JetJobMode = JetJobModes::NotSet;

		std::shared_ptr<NonLinearProgram> nlp;
		std::shared_ptr<PSIOPT> optimizer;


		virtual ~OptimizationProblemBase() = default;


		virtual PSIOPT::ConvergenceFlags solve() = 0;
		virtual PSIOPT::ConvergenceFlags optimize() = 0;
		virtual PSIOPT::ConvergenceFlags solve_optimize() = 0;
		virtual PSIOPT::ConvergenceFlags solve_optimize_solve() = 0;


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
				flag = this->solve_optimize();
				if (flag != PSIOPT::ConvergenceFlags::CONVERGED) {
					flag = this->solve();
				}
				break;
			}
			case NotSet: {
				throw::std::invalid_argument("JetJobMode not set");
				break;
			}
			default: break;
				flag = PSIOPT::ConvergenceFlags::CONVERGED;
			}
			
			this->jet_release();
			return flag;
		}



		static JetJobModes strto_JetJobMode(const std::string& str) {

			if (str == "solve"||str=="Solve")
				return JetJobModes::Solve;
			else if (str == "optimize"||str=="Optimize")
				return JetJobModes::Optimize;
			else if (str == "solve_optimize" || str == "SolveOptimize" || str == "Solve_Optimize")
				return JetJobModes::SolveOptimize;
			else if (str == "solve_optimize_solve" || str == "SolveOptimizeSolve" || str == "Solve_Optimize_Solve")
				return JetJobModes::SolveOptimizeSolve;
			else if (str == "DoNothing"||str=="do_nothing"||str=="Do_Nothing")
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




}
