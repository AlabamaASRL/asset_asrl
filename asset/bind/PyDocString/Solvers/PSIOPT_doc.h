#pragma once

namespace ASSET {
namespace doc {

const char* const PSIOPT_optimize =
    "Find the optimal solution to the NonLinearProgram given by nlp.\n\n"
    ":param arg0: Initial guess for design variables (trajectory)\n"
    ":returns: Optimal design variables (trajectory)";

const char* const PSIOPT_solve_optimize =
    "Two-stage solution: Satisfy constraints first, then optimize "
    "objective.\n\n"
    ":param arg0: Initial guess for design variables (trajectory)\n"
    ":returns: Optimal design variables (trajectory)";

const char* const PSIOPT_solve =
    "Find a solution that satisfies all constraints. Ignores objective "
    "functions.\n\n"
    ":param arg0: Initial guess for design variables (trajectory)\n"
    ":returns: Valid design variables (trajectory)";

const char* const PSIOPT_setQPParams =
    "Apply Pardiso parameters (QPPivotPerturb, QPThreads, QPPivotStrategy, "
    "QPOrd) which were directly assigned.\n\n";

const char* const PSIOPT_MaxIters =
    "int: Maximum allowed iterations. The optimization/solve loop will exit "
    "at this number regardless of whether it has found an acceptable answer.";

const char* const PSIOPT_MaxLSIters =
    "int: Maximum number of line search iterations in each optimize/solve "
    "loop";

const char* const PSIOPT_alphaRed = "";

const char* const PSIOPT_FastFactorAlg = "";

const char* const PSIOPT_LastUserTime =
    "double: How long user function evaluations took on the previous "
    "iteration.";

const char* const PSIOPT_LastQPTime =
    "double: How long optimization functions (Pardiso) took on the previous "
    "iteration.";

const char* const PSIOPT_LastIterNum =
    "int: The iteration counter of the previous iteration.";

const char* const PSIOPT_MaxAccIters = "";

const char* const PSIOPT_ObjScale =
    "double: Scaling factor for the objective function";

const char* const PSIOPT_PrintLevel =
    "int: Defines the amount of output. Level 0 prints everything; Level 1 "
    "prints progress, but does not scroll through iteration history; Level 2 "
    "outputs results only; Level 3 prints nothing";

// const char* const PSIOPT_MaxFeasRest = "int: ";

const char* const PSIOPT_KKTtol =
    "double: Tolerance for norm of primal gradient vector to be considered "
    "CONVERGED. Default = 1e-6";

const char* const PSIOPT_Bartol =
    "double: Tolerance for barrier complementarity to be considered CONVERGED. "
    "Default = 1e-6";

const char* const PSIOPT_EContol =
    "double: Tolerance for Equality Constraints to be considered CONVERGED. "
    "Default = 1e-6";

const char* const PSIOPT_IContol =
    "double: Tolerance for Inequality Constraints to be considered CONVERGED. "
    "Default = 1e-6";

const char* const PSIOPT_AccKKTtol =
    "double: Tolerance for norm of primal gradient vector to be considered "
    "ACCEPTABLE. Default = 1e-2";

const char* const PSIOPT_AccBartol =
    "double: Tolerance for barrier complementarity to be considered "
    "ACCEPTABLE. Default = 1e-3";

const char* const PSIOPT_AccEContol =
    "double: Tolerance for Equality Constraints to be considered ACCEPTABLE. "
    "Default = 1e-3";

const char* const PSIOPT_AccIContol =
    "double: Tolerance for Inequality Constraints to be considered ACCEPTABLE. "
    "Default = 1e-3";

const char* const PSIOPT_DivKKTtol =
    "double: Tolerance for norm of primal gradient vector to be considered "
    "DIVERGING. Default = 1e15";

const char* const PSIOPT_DivBartol =
    "double: Tolerance for barrier complementarity to be considered "
    "DIVERGING. Default = 1e15";

const char* const PSIOPT_DivEContol =
    "double: Tolerance for Equality Constraints to be considered DIVERGING. "
    "Default = 1e15";

const char* const PSIOPT_DivIContol =
    "double: Tolerance for Inequality Constraints to be considered DIVERGING. "
    "Default = 1e15";

const char* const PSIOPT_NegSlackReset =
    "double: Threshold for resetting slack variables. Default = 1e-12";

const char* const PSIOPT_BoundFraction = "double: ";

const char* const PSIOPT_BoundPush = "double: ";

const char* const PSIOPT_deltaH = "double: ";

const char* const PSIOPT_incrH = "double: ";

const char* const PSIOPT_decrH = "double: ";

const char* const PSIOPT_initMu = "double: ";

const char* const PSIOPT_MinMu = "double: Default = 1e-12";

const char* const PSIOPT_MaxMu = "double: Default = 100";

const char* const PSIOPT_MaxSOC = "double: ";

const char* const PSIOPT_coupleDualStep = "bool: ";

const char* const PSIOPT_PDStepStrategy =
    "PDStepStrategies: An enum value defining the primal-dual step strategy.";

const char* const PSIOPT_SOEboundRelax = "";

const char* const PSIOPT_QPParSolve = "";

const char* const PSIOPT_SoeMode = "AlgorithmModes: The optimization algorithm";

const char* const PSIOPT_OptBarMode =
    "BarrierModes: Barrier mode used when 'optimize' is called.";

const char* const PSIOPT_SoeBarMode =
    "BarrierModes: Barrier mode used when 'solve' is called";

const char* const PSIOPT_OptLSMode =
    "LineSearchModes: Line search algorithm used when 'optimize' is called";

const char* const PSIOPT_SoeLSMode =
    "LineSearchModes: Line search algorithm used when 'solve' is called";

const char* const PSIOPT_ForceQPanalysis = "";

const char* const PSIOPT_QPRefSteps = "";

const char* const PSIOPT_QPPivotPerturb = "";

const char* const PSIOPT_QPThreads =
    "int: Number of CPU threads for Pardiso. Default = 8.";

const char* const PSIOPT_QPPivotStrategy =
    "QPPivotModes: An enum value for the Pardiso pivoting strategy.";

const char* const PSIOPT_QPOrd = "";

const char* const PSIOPT_storespmat =
    "bool: Whether or not to store the SP Matrix.";

const char* const PSIOPT_getSPmat = "";

const char* const PSIOPT_getSPmat2 = "";

const char* const PSIOPT_CNRMode = "";

// const char* const PSIOPT_BarrierModes = "";

// const char* const PSIOPT_LineSearchModes = "";

// const char* const PSIOPT_QPPivotModes = "";

// const char* const PSIOPT_PDStepStrategies = "";

// const char* const PSIOPT_ConvergenceFlags = "";

}  // namespace doc
}  // namespace ASSET
