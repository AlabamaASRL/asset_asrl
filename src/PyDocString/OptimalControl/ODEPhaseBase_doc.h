#pragma once

namespace ASSET {
namespace doc {

const char* const ODEPhaseBase_setTraj1 =
    "Set a trajectory from an existing mesh with fixed spacings and defect "
    "numbers\n\n"
    ":param arg0: Full trajectory mesh to use as vector\n"
    ":param arg1: Vector of spacings between bins (0 to 1) (vector of "
    "doubles)\n"
    ":param arg2: Vector of size (arg1 - 1) defining number of defects per bin "
    "(vector of ints)\n"
    ":returns: void";

const char* const ODEPhaseBase_setTraj2 =
    "Set a trajectory from an existing mesh\n\n"
    ":param arg0: Full trajectory mesh to use as vector\n"
    ":param arg1: Vector defining number of defects per bin\n"
    ":returns: void";

const char* const ODEPhaseBase_switchTranscriptionMode1 =
    "Change the current transcription mode with fixed spacings and defect "
    "numbers\n\n"
    ":param arg0: Transcription mode to change to (Enumerator)(LGL3, LGL5, "
    "LGL7, Trapezoidal, CentralShooting)\n"
    ":param arg1: Vector of spacings between bins (0 to 1) (vector of "
    "doubles)\n"
    ":param arg2: Vector of size (arg1 - 1) defining number of defects per bin "
    "(vector of ints)\n"
    ":returns: void";

const char* const ODEPhaseBase_switchTranscriptionMode2 =
    "Change the current transcription mode\n\n"
    ":param arg0: Transcription mode to change to (Enumerator options = LGL3, "
    "LGL5, LGL7, Trapezoidal, CentralShooting)\n"
    ":returns: void";

const char* const ODEPhaseBase_transcribe =
    "Force transcription. Note: this is done internally when any problem "
    "definitions are changed and usually shouldn't be called\n\n"
    ":param arg0: Displays number of variables in phase (bool)\n"
    ":param arg1: Displays all functions attached to problem, along with "
    "vindices and cindeces (bool)\n"
    ":returns: void";

const char* const ODEPhaseBase_refineTrajManual1 =
    "Manually refine the trajectory by modifying the number of defects per "
    "bin\n\n"
    ":param arg0: New number of defects per bin (int)\n"
    ":returns: void";

const char* const ODEPhaseBase_refineTrajManual2 =
    "Manually refine the trajectory by modifying the bin spacing and number of "
    "defects per bin\n\n"
    ":param arg0: Vector of new bin spacing (vector of double)\n"
    ":param arg1: Vector of new defects per bin (vector of int)\n"
    ":returns: void";

const char* const ODEPhaseBase_refineTrajEqual =
    "Refine the trajectory to obtain equal error between each segment of the "
    "trajectory\n\n"
    ":param arg0: Number of segments to refine the trajectory (int)\n"
    ":returns: Refined trajectory (vector of states)";

const char* const ODEPhaseBase_setStaticParams =
    "Set the statc paramaters of the transcription\n\n"
    ":param arg0: Vector of paramaters to set (vector of doubles)\n"
    ":returns: void";

const char* const ODEPhaseBase_setControlMode =
    "Set the control mode of the transcription\n\n"
    ":param arg0: Desired control mode (Enumerator options = "
    "HighestOrderSpline, FirstOrderSpline, NoSpline, BlockConstant)\n"
    ":returns: void";

const char* const ODEPhaseBase_setIntegralMode =
    "Set the integral mode of the transcription\n\n"
    ":param arg0: Desired integral mode (Enumerator options = BaseIntegral, "
    "SimpsonIntegral, TrapIntegral)\n"
    ":returns: void";

const char* const ODEPhaseBase_subStaticParams =
    "Change the existing static paramaters to a new input\n\n"
    ":param arg0: Vector of paramaters to set (vector of doubles)\n"
    ":returns: void";

const char* const ODEPhaseBase_subVariables =
    "Switch the existing variables of the transcription to a new input\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Vector of indices in the problem to change (vector of int)\n"
    ":param arg2: Vector of values to corresponding variable indices (vector "
    "of double)\n"
    ":returns: void";

const char* const ODEPhaseBase_subVariable =
    "Switch one (1) existing variables of the transcription to a new input\n\n"
    ":param arg0: Index of variable to replace\n"
    ":param arg1: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg2: Value of new variable\n"
    ":returns: void";

const char* const ODEPhaseBase_returnTraj =
    "Returns the active trajectory of the transcription\n\n"
    "returns: Vector containing states of active trajectory (vector of states)";

const char* const ODEPhaseBase_returnTrajRange =
    "Returns active trajectory states between two times"
    ":param arg0: Number of defects to return (int) (The number of states "
    "returned is arg0*(number of states per defect)\n\n"
    ":param arg1: Starting dimensional time to get states (double)\n"
    ":param arg2: Final dimensional time to return states (double)\n"
    ":returns: Vector of states between times t0->tf (vector of states)";

const char* const ODEPhaseBase_returnTrajRangeND =
    "Returns active trajectory states between two nondimensional times\n\n"
    ":param arg0: Number of defects to return (int) (The number of states "
    "returned is arg0*(number of states per defect)\n"
    ":param arg1: Starting nondimensional time to get states (double)\n"
    ":param arg2: Final nondimensional time to return states (double)\n"
    ":returns: Vector of states between nondimensional times t0->tf (vector of "
    "states)";

const char* const ODEPhaseBase_returnCostateTraj =
    "Returns an approximation of the costates of the trajectory (must be done "
    "after at least one solve or optimize call)\n\n"
    ":returns: Vector of costates and times (size xvars + 1)";

const char* const ODEPhaseBase_returnEqualConLmults =
    "Returns equality constraint lambda multipliers (must be done after at "
    "least one solve or optimize call)\n\n"
    ":param arg0: Index of equality constraint to obtain lambda multipliers\n"
    ":returns: Vector of lamda multipliers for trajectory equality constraints";

const char* const ODEPhaseBase_returnInequalConLmults =
    "Returns inequality constraint lambda multipliers (must be done after at "
    "least one solve or optimize call)\n\n"
    ":param arg0: Index of inequality constraint to obtain lambda multipliers\n"
    ":returns: Vector of lamda multipliers for trajectory inequality "
    "constraints";

const char* const ODEPhaseBase_returnStaticParam =
    "Returns the current active static parameters of the trajectory\n\n"
    ":returns: Vector of static parameters (vector of double)";

const char* const ODEPhaseBase_removeEqualCon =
    "Removes an equality constraint from the transcription\n\n"
    ":param arg0: Transcription index of the equality constraint to remove\n"
    ":returns: void";

const char* const ODEPhaseBase_removeInequalCon =
    "Removes an inequality constraint from the transcription\n\n"
    ":param arg0: Transcription index of the inequality constraint to remove\n"
    ":returns: void";

const char* const ODEPhaseBase_removeStateObjective =
    "Removes a state objective from the transcription\n\n"
    ":param arg0: Transcription index of the state objective to remove\n"
    ":returns: void";

const char* const ODEPhaseBase_removeIntegralObjective =
    "Removes an integral objective from the transcription\n\n"
    ":param arg0: Transcription index of the integral objective to remove\n"
    ":returns: void";

const char* const ODEPhaseBase_removeIntegralParamFunction =
    "Removes an integral parameter function from the transcription\n\n"
    ":param arg0: Transcription index of the integral parameter function to "
    "remove\n"
    ":returns: void";
////////////////////////////////////////////////////////////////////////////////////////////////////////

const char* const ODEPhaseBase_addEqualCon1 =
    "Adds an equality constraint to the transcription\n\n"
    ":param arg0: Set equality constraint with ASSET StateConstraint object\n"
    ":returns: Index of the added equality constraint in the transcription";

const char* const ODEPhaseBase_addEqualCon2 =
    "Adds an equality constraint to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, "
    "Path)\n"
    ":param arg1: Function (passed with arguments) representing the equality "
    "constraint\n"
    ":param arg2: Variables of each state to apply equality constraint to\n"
    ":returns: Index of the added equality constraint in the transcription";

const char* const ODEPhaseBase_addEqualCon3 =
    "Adds an equality constraint to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, "
    "Path)\n"
    ":param arg1: Function (passed with arguments) representing the equality "
    "constraint\n"
    ":param arg2: Variables of each state to apply equality constraint to\n"
    ":param arg3: Vector of ODE  parameter variables (if any)\n"
    ":param arg4: Vector of Static parameters (if any)\n"
    ":returns: Index of the added equality constraint in the transcription";

const char* const ODEPhaseBase_addDeltaVarEqualCon1 =
    "Adds equality constraint of the difference of a variable to the "
    "transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = FrontandBack, "
    "BackandFront)\n"
    ":param arg1: The index of the variable to set the equality constraint\n"
    ":param arg2: Value of the difference to satisfy the constraint\n"
    ":param arg3: Scaling parameter (typically 1.0)\n"
    ":returns: Index of the equality constraint in the transcription";

const char* const ODEPhaseBase_addDeltaVarEqualCon2 =
    "Adds equality constraint of the difference between two variables to the "
    "transcription (Assumes BackFront PhaseRegionFlag)\n\n"
    ":param arg0: The index of the variable to set the equality constraint\n"
    ":param arg1: Value of the difference to satisfy the constraint\n"
    ":param arg2: Scaling parameter (typically 1.0)\n"
    ":returns: Index of the equality constraint in the transcription";

const char* const ODEPhaseBase_addDeltaVarEqualCon3 =
    "Adds equality constraint of the difference between two variables to the "
    "transcription (Assumes BackFront PhaseRegionFlag and scale 1)\n\n"
    ":param arg0: The index of the variable to set the equality constraint\n"
    ":param arg1: Value of the difference to satisfy the constraint\n"
    ":returns: Index of the equality constraint in the transcription";

const char* const ODEPhaseBase_addDeltaTimeEqualCon1 =
    "Adds equality constraint for the difference between first and final time "
    "in the transcription\n\n"
    ":param arg0: Value of time difference (time of flight) to satisfy this "
    "constraint\n"
    ":param arg1: Scaling parameter (typically 1.0)\n"
    ":returns: Index of the equality constraint in the transcription";

const char* const ODEPhaseBase_addDeltaTimeEqualCon2 =
    "Adds equality constraint for the difference between first and final time "
    "in the transcription (scale assumed to be 1.0)\n\n"
    ":param arg0: Value of time difference (time of flight) to satisfy this "
    "constraint\n"
    ":returns: Index of the equality constraint in the transcription";

const char* const ODEPhaseBase_addBoundaryValue =
    "Adds a boundary constraint to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Vector of indices to apply the inequality constraint to "
    "(vector of int)\n"
    ":param arg2: Vector of values to set bounds on satisfying the constraint "
    "(vector of double)\n"
    ":returns: Index of the boundary constraint in the transcription";

const char* const ODEPhaseBase_addValueLock =
    "Locks value of variable for the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n\n"
    ":param arg1: Vector of indices to lock (vector of int)\n"
    ":returns: Index of locked variables in the transcription";

const char* const ODEPhaseBase_addBoundaryValues =
    "Adds boundary constraints to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n\n"
    ":param arg1: Vector of indices to apply the inequality constraint to "
    "(vector of int)\n"
    ":param arg2: Vector of values for each variable to set bounds on "
    "satisfying the constraint (vector of double)\n"
    ":returns: Index of the boundary constraint in the transcription";

///////////////////////////////////////////////////////////////////////////////

const char* const ODEPhaseBase_addInequalCon1 =
    "Adds an inequality constraint to the transcription\n\n"
    ":param arg0: ASSET StateConstraint object representing the ineqality "
    "constraint\n"
    ":returns: Index of inequality constraint in the transcription";

const char* const ODEPhaseBase_addInequalCon2 =
    "Adds an inequality constraint to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Function (passed with arguments) representing the inequality "
    "constraint\n"
    ":param arg2: Vector of variable indices to apply the inequality "
    "constraint\n"
    ":returns: Index of inequality constraint in the transcription";

const char* const ODEPhaseBase_addInequalCon3 =
    "Adds an inequality constraint to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Function (passed with arguments) representing the inequality "
    "constraint\n"
    ":param arg2: Variables of each state to apply inequality constraint to\n"
    ":param arg3: Vector of ODE  parameter variables (if any)\n"
    ":param arg4: Vector of Static parameters (if any)\n"
    ":returns: Index of the added inequality constraint in the transcription";

const char* const ODEPhaseBase_addLUVarBound1 =
    "Adds a lower and upper bound constraint to the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Index of the variable to apply the bounds to (int)\n"
    ":param arg2: Value of constraint lower bound (double)\n"
    ":param arg3: Value of constraint upper bound (double)\n"
    ":returns: Index of the upper and lower bound constraint in the "
    "transcription";

const char* const ODEPhaseBase_addLUVarBound2 =
    "Adds a lower and upper bound constraint to the transcription with a "
    "scaling term\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Index of the variable to apply the bounds to (int)\n"
    ":param arg2: Value of constraint lower bound (double)\n"
    ":param arg3: Value of constraint upper bound (double)\n"
    ":param arg4: Scale value for constraint (usually 1.0)\n"
    ":returns: Index of the upper and lower bound constraint in the "
    "transcription";

const char* const ODEPhaseBase_addLUVarBound3 =
    "Adds a lower and upper bound constraint to the transcription with "
    "seperate scaling terms for the upper and lower bound\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Index of the variable to apply the bounds to (int)\n"
    ":param arg2: Value of constraint lower bound (double)\n"
    ":param arg3: Value of constraint upper bound (double)\n"
    ":param arg4: Scale value lower bound (usually 1.0)\n"
    ":param arg5: Scale value upper bound (usually 1.0)\n"
    ":returns: Index of the upper and lower bound constraint in the "
    "transcription";

const char* const ODEPhaseBase_addLUVarBounds =
    "Adds a lower and upper bound constraint to several variables within the "
    "transcription with a scaling term\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Vector of Indices of the variable to apply the bounds to "
    "(vector of int)\n"
    ":param arg2: Value of constraint lower bound for all variables(double)\n"
    ":param arg3: Value of constraint upper bound for all variables (double)\n"
    ":param arg4: Scale value for constraint (usually 1.0)\n"
    ":returns: Vector of Indices of the upper and lower bound constraints in "
    "the transcription (vector of int)";

const char* const ODEPhaseBase_addLowerVarBound1 =
    "Adds a lower bound to a variable within the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n\n"
    ":param arg1: Index of variable to add the lower bound constraint (int)\n"
    ":param arg2: Value of the lower bound constraint (double)\n"
    ":param arg3: Scaling term (usually 1.0)\n"
    ":returns: Index of the lower bound constraint in the transcription";

const char* const ODEPhaseBase_addLowerVarBound2 =
    "Adds a lower bound to a variable within the transcription, assuming scale "
    "factor of 1.0\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Index of variable to add the lower bound constraint (int)\n"
    ":param arg2: Value of the lower bound constraint (double)\n"
    ":returns: Index of the lower bound constraint in the transcription";

const char* const ODEPhaseBase_addUpperVarBound1 =
    "Adds a upper bound to a variable within the transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Index of variable to add the upper bound constraint (int)\n"
    ":param arg2: Value of the upper bound constraint (double)\n"
    ":param arg3: Scaling term (usually 1.0)\n"
    ":returns: Index of the upper bound constraint in the transcription";

const char* const ODEPhaseBase_addUpperVarBound2 =
    "Adds a upper bound to a variable within the transcription, assuming scale "
    "factor of 1.0\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Index of variable to add the upper bound constraint (int)\n"
    ":param arg2: Value of the upper bound constraint (double)\n"
    ":returns: Index of the upper bound constraint in the transcription";

const char* const ODEPhaseBase_addLUNormBound1 =
    "Adds a constraint to the norm of the upper and lower bounds within the "
    "transcription\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Vector of indices for the variables to apply the norm bound "
    "on (vector of int)\n"
    ":param arg2: Value of the lower bound for the norm of the variables "
    "(double)\n"
    ":param arg3: Value of the upper bound for the norm of the variables "
    "(double)\n"
    ":param arg4: Scale value for the lower bound of the norm of the variables "
    "(usually 1.0) (double)\n"
    ":param arg5: Scale value for the upper bound of the norm of the variables "
    "(usually 1.0) (double)\n"
    ":returns: Index of the LUnorm bound constraint in the transcription";

const char* const ODEPhaseBase_addLUNormBound2 =
    "Adds a constraint to the norm of the upper and lower bounds within the "
    "transcription, with the same scale value for both upper and lower "
    "bounds\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Vector of indices for the variables to apply the norm bound "
    "on (vector of int)\n"
    ":param arg2: Value of the lower bound for the norm of the variables "
    "(double)\n"
    ":param arg3: Value of the upper bound for the norm of the variables "
    "(double)\n"
    ":param arg4: Scale value for the upper and lower bound of the norm of the "
    "variables (usually 1.0) (double)\n"
    ":returns: Index of the LUnorm bound constraint in the transcription";

const char* const ODEPhaseBase_addLUNormBound3 =
    "Adds a constraint to the norm of the upper and lower bounds within the "
    "transcription, with the same scale value of 1.0 for both upper and lower "
    "bounds\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: Vector of indices for the variables to apply the norm bound "
    "on (vector of int)\n"
    ":param arg2: Value of the lower bound for the norm of the variables "
    "(double)\n"
    ":param arg3: Value of the upper bound for the norm of the variables "
    "(double)\n"
    ":returns: Index of the LUnorm bound constraint in the transcription";

const char* const ODEPhaseBase_addLowerFuncBound1 =
    "Adds a constraint on the lower bound of a function"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n\n"
    ":param arg1: The function to add the lower bound constraint to\n"
    ":param arg2: Vector of indices for variables to pass to the function in "
    "arg1 (vector of int)\n"
    ":param arg3: Value of the constraint on the lower bound of the function "
    "(double)\n"
    ":param arg4: Scale factor for the lower bound constraint (usually 1.0)\n"
    ":returns: Index of the Lower function bound in the transcription";

const char* const ODEPhaseBase_addUpperFuncBound2 =
    "Adds a constraint on the upper bound of a function\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = Front, Back, Path, "
    "ODEParams, StaticParams)\n"
    ":param arg1: The function to add the upper bound constraint to\n"
    ":param arg2: Vector of indices for variables to pass to the function in "
    "arg1 (vector of int)\n"
    ":param arg3: Value of the constraint on the upper bound of the function "
    "(double)\n"
    ":param arg4: Scale factor for the upper bound constraint (usually 1.0)\n"
    ":returns: Index of the upper function bound in the transcription";

const char* const ODEPhaseBase_addLowerDeltaVarBound1 =
    "Adds Lower bound constraint for the difference between a variable along "
    "the phase\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = FrontandBack, "
    "BackandFront)\n"
    ":param arg1: Index of the variable to add the Lower Delta variable bound "
    "to (int)\n"
    ":param arg2: Value of the lower bound constraint (double)\n"
    ":param arg3: Scale factor for the lower bound constraint (usually 1.0)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addLowerDeltaVarBound2 =
    "Adds Lower bound constraint for the difference between a variable along "
    "the phase, assuming PhaseRegionFlag = FrontandBack\n\n"
    ":param arg0: Index of the variable to add the Lower Delta variable bound "
    "to (int)\n"
    ":param arg1: Value of the lower bound constraint (double)\n"
    ":param arg2: Scale factor for the lower bound constraint (usually 1.0)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addLowerDeltaVarBound3 =
    "Adds Lower bound constraint for the difference between a variable along "
    "the phase, assuming PhaseRegionFlag = FrontandBack and scale factor = "
    "1.0\n\n"
    ":param arg0: Index of the variable to add the Lower Delta variable bound "
    "to (int)\n"
    ":param arg1: Value of the lower bound constraint (double)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addLowerDeltaTimeBound1 =
    "Adds Lower bound constraint for the difference between in time (time of "
    "flight) along the phase\n\n"
    ":param arg0: Value of the lower bound constraint (double)\n"
    ":param arg1: Scale factor for the lower bound constraint (usually 1.0)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addLowerDeltaTimeBound2 =
    "Adds Lower bound constraint for the difference between in time (time of "
    "flight) along the phase, assuming scale factor = 1.0\n\n"
    ":param arg0: Value of the lower bound constraint (double)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addUpperDeltaVarBound1 =
    "Adds upper bound constraint for the difference between a variable along "
    "the phase\n\n"
    ":param arg0: The PhaseRegionFlag (Enumerator options = FrontandBack, "
    "BackandFront)\n"
    ":param arg1: Index of the variable to add the upper Delta variable bound "
    "to (int)\n"
    ":param arg2: Value of the upper bound constraint (double)\n"
    ":param arg3: Scale factor for the upper bound constraint (usually 1.0)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addUpperDeltaVarBound2 =
    "Adds upper bound constraint for the difference between a variable along "
    "the phase, assuming PhaseRegionFlag = FrontandBack\n\n"
    ":param arg0: Index of the variable to add the upper Delta variable bound "
    "to (int)\n"
    ":param arg1: Value of the upper bound constraint (double)\n"
    ":param arg2: Scale factor for the upper bound constraint (usually 1.0)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addUpperDeltaVarBound3 =
    "Adds upper bound constraint for the difference between a variable along "
    "the phase, assuming PhaseRegionFlag = FrontandBack and scale factor = "
    "1.0\n\n"
    ":param arg0: Index of the variable to add the upper Delta variable bound "
    "to (int)\n"
    ":param arg1: Value of the upper bound constraint (double)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addUpperDeltaTimeBound1 =
    "Adds upper bound constraint for the difference between in time (time of "
    "flight) along the phase\n\n"
    ":param arg0: Value of the upper bound constraint (double)\n"
    ":param arg1: Scale factor for the upper bound constraint (usually 1.0)\n"
    ":returns: Index of the constraint in the transcription";

const char* const ODEPhaseBase_addUpperDeltaTimeBound2 =
    "Adds upper bound constraint for the difference between in time (time of "
    "flight) along the phase, assuming scale factor = 1.0\n\n"
    ":param arg0: Value of the upper bound constraint (double)\n"
    ":returns: Index of the constraint in the transcription";

////////////////////////////////////////////////////////////////////////////

const char* const ODEPhaseBase_addStateObjective =
    "Adds an objective function computable at the state/states indicated\n\n"
    ":param arg0: ASSET StateObjective type indicating the desired states and "
    "objective function\n"
    ":returns: Index of the state objective constraint";

const char* const ODEPhaseBase_addValueObjective =
    "Adds an objective function that is the value of the specified state "
    "variable at the phase region\n\n"
    ":param arg0: The PhaseRegionFlag (Enumurator options = Front, Back, "
    "ODEParams, StaticParams)\n"
    ":param arg1: The Index of the variable to evaluate for the objective "
    "function (int)\n"
    ":param arg2: Scale factor for the objective function (usually 1.0) "
    "(double)\n"
    ":returns: Index of the value objective constraint";

const char* const ODEPhaseBase_addDeltaVarObjective =
    "Adds an objective function that is the change between the value of the "
    "specified state variable at the phase region\n\n"
    ":param arg1: The Index of the variable to evaluate for the objective "
    "function (int)\n"
    ":param arg2: Scale factor for the objective function (usually 1.0) "
    "(double)\n"
    ":returns: Index of the delta variable objective";

const char* const ODEPhaseBase_addDeltaTimeObjective =
    "Adds an objective that is the change between the times at the phase "
    "region (Always tf - t0)\n\n"
    ":param arg0: Scale factor for the objective function (usually 1.0) "
    "(double)\n"
    ":returns: Index of the delta time objective";

const char* const ODEPhaseBase_addIntegralObjective1 =
    "Adds an objective that evaluates the integral of the given function over "
    "the phase\n\n"
    ":param arg0: ASSET StateObjective object\n"
    ":returns: Index of the integral objective";

const char* const ODEPhaseBase_addIntegralObjective2 =
    "Adds an objective that evaluates the integral of the given function over "
    "the phase\n\n"
    ":param arg0: Function to integrate over the phase\n"
    ":param arg1: Variables neccessary to invoke function in arg0 (Vector of "
    "indices)\n"
    ":returns: Index of the integral objective";

///////////////////////////////////////////////////////////////////////////////

const char* const ODEPhaseBase_addIntegralParamFunction1 =
    "Adds an integral parameter constraint such that the static parameter with "
    "index 'pv' is equal to the integral of the given function\n\n"
    ":param arg0: ASSET StateObjective object (not neccessarily an objective, "
    "but must be Scalar)\n"
    ":param arg1: Index of the parameter of interest\n"
    ":returns: Index of the integral parameter constraint";

const char* const ODEPhaseBase_addIntegralParamFunction2 =
    "Adds an integral parameter constraint such that the static parameter with "
    "index 'pv' is equal to the integral of the given function\n\n"
    ":param arg0: Function to integrate over the phase\n"
    ":param arg1: Variables neccessary to invoke function in arg0 (Vector of "
    "indices)\n"
    ":param arg2: Index of the parameter of interest\n"
    ":returns: Index of the integral parameter constraint";

const char* const ODEPhaseBase_optimizer =
    "Returns a reference to the optimizer PSIOPT instance\n\n"
    ":type: PSIOT";

const char* const ODEPhaseBase_Threads =
    "The number of threads to parallelize the phase functions\n\n"
    ":type: int";

const char* const ODEPhaseBase_solve =
    "Solves the phase subject to the provided constraints. (Note: This does "
    "NOT optimize)\n\n"
    ":returns: The solver convergence flag (bool, succeeded or failed)";

const char* const ODEPhaseBase_optimize =
    "Optimizes the phase subject to the provided constraints and objectives. "
    "(Note: This does optimize)\n\n"
    ":returns: The optimizer convergence flag (bool, succeeded or failed)";

const char* const ODEPhaseBase_solve_optimize =
    "First, solves the phase subject to the provided constraints, then "
    "optimizes the phase subject to the provided constraints and objectives. "
    "(Note: This does optimize)\n\n"
    ":returns: The optimizer convergence flag (bool, succeeded or failed)";

}  // namespace doc
}  // namespace ASSET
