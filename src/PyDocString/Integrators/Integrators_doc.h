#pragma once

namespace ASSET {
namespace doc {

const char* const IntegratorBase_default =
    "Instanstiate an Integrator object\n\n"
    ":param arg0: Object of ODE type with compute function to evaluate\n"
    ":param arg1: Object of Stepper type determining which integrator method "
    "to use\n"
    ":param arg2: Stepsize of integration (double)\n"
    ":returns: Integrator object";

const char* const IntegratorBase_init =
    "Method to set parameters for Integrator this->object\n\n"
    ":param arg0: Object of ODE type with compute function to evaluate\n"
    ":param arg1: Object of Stepper type determining which integrator method "
    "to use\n"
    ":param arg2: Stepsize of integration (double)\n"
    ":returns: void";

const char* const IntegratorBase_setAbsTol =
    "Set absolute integration tolerance for Integrator object\n\n"
    ":param arg0: Absolute integration tolerance (double)\n"
    ":returns: void";

const char* const IntegratorBase_compute_constant =
    "Evaluates the integral over the range t0->tf with a constant stepsize.\n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Output vector after integration step\n"
    ":returns: void";

const char* const IntegratorBase_compute_adaptivec =
    "Evaluates the integral over the range t0->tf with adaptive stepsize\n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Output vector after integration step\n"
    ":param arg2: (Bool) store adaptive step outputs\n"
    ":returns: vector of states at each step of integration";

const char* const IntegratorBase_compute_impl =
    "Performs either adaptive or constant integration, dependent on \n\n"
    "this->Adaptive member\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Output vector after integration step\n"
    ":returns: void";

const char* const IntegratorBase_integrate =
    "Integrates state variables from t0->tf\n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Final integration time\n"
    ":returns: Final state after integration step";

const char* const IntegratorBase_integrate_cpv =
    "Integrates state variables from t0->tf with controls\n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Final integration time\n"
    ":param arg2: Vector of controller parameters\n"
    ":returns: Final state after integration step";

const char* const IntegratorBase_integrate_stm =
    "Integrates state transition matrix (STM) of ODE from t0->tf\n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Final integration time\n"
    ":returns: STM at final integration time";

const char* const IntegratorBase_integrate_stm_cpv =
    "Integrates state transition matrix (STM) of ODE from t0->tf\n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Final integration time\n"
    ":param arg2: Vector of controller parameters\n"
    ":returns: STM at final integration time";

const char* const IntegratorBase_integrate_stm_parallel =
    "Integrates vector of state transition matrices (STM) of ODE from t0->tf"
    "in parallel \n"
    ":param arg0: Vector of state variables to integrate\n\n"
    ":param arg1: Vector of final integration times\n"
    ":param arg2: Number to threads to parallelize\n"
    ":returns: Vector of STMs at final integration times\n";

const char* const IntegratorBase_integrate_stm_parallel_cpv =
    "Integrates vector of state transition matrices (STM) of ODE from t0->tf"
    "in parallel \n\n"
    ":param arg0: Vector of state variables to integrate\n"
    ":param arg1: Vector of final integration times\n"
    ":param arg2: Vector of controller parameters for each state\n"
    ":param arg3: Number to threads to parallelize\n"
    ":returns: Vector of STMs at final integration times";

const char* const IntegratorBase_integrate_stm_parallel_single =
    "Integrates a single state transition matrix (STM) of ODE from t0->tf in "
    "parallel \n\n"
    ":param arg0: State variables to integrate\n"
    ":param arg1: Final integration times\n"
    ":param arg2: Number to threads to parallelize\n"
    ":returns: Vector of STMs at final integration times";

const char* const IntegratorBase_integrate_stm_parallel_single_cpv =
    "Integrates a single state transition matrix (STM) of ODE from t0->tf in "
    "parallel \n\n"
    ":param arg0: State variables to integrate\n"
    ":param arg1: Final integration times\n"
    ":param arg2: Controller parameters for each state\n"
    ":param arg3: Number to threads to parallelize\n"
    ":returns: Vector of STMs at final integration times";

const char* const IntegratorBase_integrate_parallel =
    "Integrate vector of states in parallel from vector of times t0->tf in "
    "parallel\n\n"
    ":param arg0: Vector of states to integrate\n"
    ":param arg1: Vector of final integration times\n"
    ":param arg2: Number to threads to parallelize\n"
    ":returns: Vector of final states for each input state vector in x0";

const char* const IntegratorBase_integrate_parallel_cpv =
    "Integrate vector of states in parallel from vector of times t0->tf in "
    "parallel\n\n"
    ":param arg0: Vector of states to integrate\n"
    ":param arg1: Vector of final integration times\n"
    ":param arg2: Number to threads to parallelize\n"
    ":param arg3: Vector of controller parameters\n"
    ":returns: Vector of final states for each input state vector in x0";

const char* const IntegratorBase_integrate_dense =
    "Integrate with dense output from time t0->tf\n\n"
    ":param arg0: State vector to integrate\n"
    ":param arg1: Final integration time\n"
    ":param arg2: Number of states to return in dense output vector\n"
    ":returns: Vector of states with length equal to NumStates for full \n"
    "integration time";

const char* const IntegratorBase_integrate_dense_cpv =
    "Integrate with dense output from time t0->tf\n\n"
    ":param arg0: State vector to integrate\n"
    ":param arg1: Final integration time\n"
    ":param arg2: Number of states to return in dense output vector\n"
    ":param arg3: Vector of controller parameters\n"
    ":returns: Vector of states with length equal to NumStates for full \n"
    "integration time";

const char* const IntegratorBase_integrate_dense_exit =
    "Integrate with dense output from time t0->tf with exit condition\n\n"
    ":param arg0: State vector to integrate\n"
    ":param arg1: Final integration time\n"
    ":param arg2: Number of states to return in dense output vector\n"
    ":param arg3: Function representing exit condition for state vector\n"
    ":returns: Vector of states with length equal to NumStates. ";

const char* const IntegratorBase_DefStepSize =
    "Step size of constant step integration\n\n"
    ":type : double\n";

const char* const IntegratorBase_MaxStepSize =
    "Maximum step length for adaptive integration\n\n"
    ":type : double\n";

const char* const IntegratorBase_MinStepSize =
    "Minimum step length for adaptive integration\n\n"
    ":type : double\n";
const char* const IntegratorBase_MaxStepChange =
    "Maximum allowable step size change between steps of adaptive integration\n\n"
    ":type : double\n";

const char* const IntegratorBase_Adaptive =
    "Set adaptive integration\n\n"
    ":type : bool\n";

const char* const IntegratorBase_AbsTols =
    "Absolute tolerance for adaptive integration\n\n"
    ":type : double\n";

const char* const IntegratorBase_FastAdaptiveSTM =
    "Set adaptive integration for state transition matrix integration\n\n"
    ":type : bool\n";

const char* const IntegratorBase_get_stepper =
    "Returns stepper for Integrator object\n\n"
    ":returns: Stepper object\n";

}  // namespace doc
}  // namespace ASSET
