#pragma once

namespace ASSET {
namespace doc {

const char* const OptimalControlProblem_addLinkEqualCon1 =
    "Adds an equality link constraint to the optimal control problem\n\n"
    ":param arg0: Predefined LinkConstraint object\n"
    ":type arg0: LinkConstraint\n"
    ":returns: Index of the equality link constraint in the optimal control "
    "problem";

const char* const OptimalControlProblem_addLinkEqualCon2 =
    "Adds an equality link constraint to the optimal control problem\n\n"
    ":param arg0: Vector of Predefined LinkConstraint object\n"
    ":type arg0: LinkConstraint\n"
    ":param arg1: Vector of PhaseRegionFlags\n"
    "type arg1: PhaseRegionFlags\n"
    ":param arg2: Vector of phases to link. Must be same length as arg1\n"
    ":param arg3: Vector of indices of state variables to link. Must be same "
    "length as arg1\n"
    ":param arg4: Vector of link indices to perform phase linking. Must be "
    "same length as arg1\n"
    ":returns: Index of the equality link constraint in the optimal control "
    "problem";

const char* const OptimalControlProblem_addForwardLinkEqualCon =
    "Adds a forward PhaseRegion Flag equality link constraint to the optimal "
    "control problem\n\n"
    ":param arg0: Index of phase for the front of link equality constraint\n"
    ":type arg0: int\n"
    ":param arg1: Index of end phase for link equality constraint\n"
    ":type arg1: int\n"
    ":param arg2: Vector of indices of state variables to apply link "
    "constraint to\n"
    ":type arg2: Vector<int>\n"
    ":returns: Index of the forward equality link constraint in the optimal "
    "control problem";

const char* const OptimalControlProblem_addDirectLinkEqualCon =
    "Adds a direct equality link constraint to the optimal control problem\n\n"
    ":param arg0: LinkFlag defining what type of link constraint\n"
    ":type arg0: LinkFlags\n"
    ":param arg1: Index of phase for the front of link equality constraint\n"
    ":type arg1: int\n"
    ":param arg2: Vector of indices of front phase state variables to apply "
    "link constraint to\n"
    ":type arg2: Vector<int>\n"
    ":param arg3: Index of end phase for link equality constraint\n"
    ":type arg3: int\n"
    ":param arg2: Vector of indices of end phase state variables to apply link "
    "constraint to\n"
    ":type arg4: Vector<int>\n"
    ":returns: Index of the direct equality link constraint in the optimal "
    "control problem";

const char* const OptimalControlProblem_addLinkInequalCon =
    "Adds a inequality link constraint to the optimal control problem\n\n"
    ":param arg0: Fully formed LinkConstraint object\n"
    ":type arg0: LinkConstraint\n"
    ":returns: Index of the inequality link constraint in the optimal control "
    "problem";

const char* const OptimalControlProblem_addLinkObjective =
    "Adds a link objective to the optimal control problem\n\n"
    ":param arg0: Fully formed LinkObjective object\n"
    ":type arg0: LinkObjective\n"
    ":returns: Index of the link objective in the optimal control problem";

const char* const OptimalControlProblem_addLinkParamEqualCon1 =
    "Adds an equality parameter link constraint to the optimal control problem "
    "with different link parameter indices\n\n"
    ":param arg0: VectorFunctional defining link parameter constraint\n"
    ":param arg1: Vector of vectors link parameter indices that are inputs to "
    "arg1\n"
    ":returns: Index of equality parameter link constraint in the optimal "
    "control problem";

const char* const OptimalControlProblem_addLinkParamEqualCon2 =
    "Adds an equality parameter link constraint to the optimal control problem "
    "that shares the same vector of indices for the link parameter "
    "constraint\n\n"
    ":param arg0: VectorFunctional defining link parameter constraint\n"
    ":param arg1: Vector of link parameter indices that are inputs to arg1\n"
    ":returns: Index of equality parameter link constraint in the optimal "
    "control problem";

const char* const OptimalControlProblem_addLinkParamInequalCon1 =
    "Adds an inequality parameter link constraint to the optimal control "
    "problem with different link parameter indices\n\n"
    ":param arg0: VectorFunctional defining link parameter constraint\n"
    ":param arg1: Vector of vectors link parameter indices that are inputs to "
    "arg1\n"
    ":returns: Index of equality parameter link constraint in the optimal "
    "control problem";

const char* const OptimalControlProblem_addLinkParamInequalCon2 =
    "Adds an inequality parameter link constraint to the optimal control "
    "problem that shares the same vector of indices for the link parameter "
    "constraint\n\n"
    ":param arg0: VectorFunctional defining link parameter constraint\n"
    ":param arg1: Vector of link parameter indices that are inputs to arg1\n"
    ":returns: Index of equality parameter link constraint in the optimal "
    "control problem";

const char* const OptimalControlProblem_addLinkParamObjective1 =
    "Adds a parameter link objective(s) to the optimal control problem with "
    "different link parameter indices\n\n"
    ":param arg0: ScalarFunctional defining link parameter objectives\n"
    ":param arg1: Vector of vectors of link parameter indices that are inputs "
    "to arg1\n"
    ":returns: Index of link parameter objective in the optimal control "
    "problem";

const char* const OptimalControlProblem_addLinkParamObjective2 =
    "Adds a parameter link objective(s) to the optimal control problem that "
    "shares the same vector of indices for the link parameter constraint\n\n"
    ":param arg0: ScalarFunctional defining link parameter objectives\n"
    ":param arg1: Vector of link parameter indices that are inputs to arg1\n"
    ":returns: Index of link parameter objective in the optimal control "
    "problem";

const char* const OptimalControlProblem_removeLinkEqualCon =
    "Discard the specified link equality constraint.\n\n"
    ":param arg0: The index of the link equality constraint you are "
    "removing. Allows negative indexing.\n"
    ":type arg0: int\n"
    ":rtype: void";

const char* const OptimalControlProblem_removeLinkInequalCon =
    "Discard the specified link inequality constraint.\n\n"
    ":param arg0: The index of the link inequality constraint you are "
    "removing. Allows negative indexing.\n"
    ":type arg0: int\n"
    ":rtype: void";

const char* const OptimalControlProblem_removeLinkObjective =
    "Discard the specified link objective.\n\n"
    ":param arg0: The index of the link objective you are removing. Allows "
    "negative indexing.\n"
    ":type arg0: int\n"
    ":rtype: void";

const char* const OptimalControlProblem_addPhase =
    "Add a phase to this OCP.\n\n"
    ":param arg0: The phase object you want to add\n"
    ":type arg0: PhaseInterface\n"
    ":returns: The index of the newly added phase\n"
    ":rtype: int";

const char* const OptimalControlProblem_removePhase =
    "Remove a phase from this OCP.\n\n"
    ":param arg0: Index of the phase you want to remove\n"
    ":type arg0: int\n"
    ":rtype: void";

const char* const OptimalControlProblem_Phase =
    "Gets a reference to a specific phase associated with this OCP. Allows "
    "negative indexing.\n\n"
    ":param arg0: Index of the desired phase object.\n"
    ":type arg0: int\n"
    ":returns: A reference to the desired Phase object"
    ":rtype: PhaseInterface";

const char* const OptimalControlProblem_setLinkParams =
    "Set the number and initial values of the OCP's link parameters.\n\n"
    ":param arg0: Numpy array of link parameters\n"
    ":rtype: void";

const char* const OptimalControlProblem_returnLinkParams =
    "Get the link parameters for this OCP.\n\n"
    ":returns: A numpy array of the previously specified link parameters.";

const char* const OptimalControlProblem_solve =
    "Find a solution that satisfies all constraints across and between all "
    "phases. Ignores objective functions. Initial guesses should have already "
    "been set in each phase via  setTraj .\n\n"
    ":returns:  Enumerator value indicating success or failure\n"
    ":rtype: ConvergenceFlags";

const char* const OptimalControlProblem_optimize =
    "Find the optimal solution to the total objective of all phases. Initial "
    "guesses should have already been set in each phase via "
    " setTraj .\n\n"
    ":returns:  Enumerator value indicating success or failure\n"
    ":rtype: ConvergenceFlags";

const char* const OptimalControlProblem_solve_optimize =
    "Two-stage solution: Satisfy constraints, then optimize objective. "
    "Initial guesses should have already been set in each phase via "
    " setTraj .\n\n"
    ":returns:  Enumerator value indicating success or failure\n"
    ":rtype: ConvergenceFlags";

const char* const OptimalControlProblem_transcribe =
    "Given all phases, links, objectives, etc., construct the NonLinearProgram "
    "object that will be optimized.\n\n"
    ":param arg0: Whether to print problem statistics such as the number of "
    "variables, constraints, and phases.\n"
    ":type arg0: bool\n"
    ":param arg1: Whether to print information about the objective and "
    "constraint functions, such as the name and size.\n"
    ":type arg1: bool\n"
    ":rtype: void";

const char* const OptimalControlProblem_Phases =
    "list: A list of references to all phases associated with this OCP.";

const char* const OptimalControlProblem_optimizer =
    "PSIOPT: An optimizer instance attached to the OCP.";

const char* const OptimalControlProblem_Threads =
    "int: Number of threads to use for this OCP.";

}  // namespace doc
}  // namespace ASSET
