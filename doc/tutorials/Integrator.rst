Integrators
===========

Almost all use cases for ASSET will involve numerical integration of some type.
To that end, the integration module includes support for a range of common integrator methods.
In this module we will describe some typical examples for basic integration use, as well as more advanced scenarios involving stopping criteria and integration control laws.

Basic Use
#########

If you haven't covered the :ref:`Python: Vector Functions` tutorial, we would recommend you do so now before proceeding.
Every ASSET :code:`Integrator` object requires an :code:`ODE` instance to be constructed. 
We will start with the same Two-Body Solar Sail ODE that we used in :ref:`Python: Vector Functions`.


.. code-block:: python

    import asset as ast

    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments
    Tmodes = oc.TranscriptionModes


    def SolarSail_Acc(r, n, scale):
        ndr2 = vf.dot(r, n).squared()
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc


    def Full_TwoBody_SolarSail_Model(mu, beta):
        args = Args(10)
        r = args.head_3()
        v = args.segment_3(3)
        n = args.tail_3()
        acc = -mu * r.normalized_power3() + SolarSail_Acc(r, n, beta * mu)
        return vf.Stack([v, acc])


    Two_Body_SolarSail_ODE = oc.ode_x_u.ode(Full_TwoBody_SolarSail_Model(1, 0.02), 6, 3)

    phase = Two_Body_SolarSail_ODE.phase(Two_Body_SolarSail_ODE, Tmodes.LGL3)


Once we have defined our ODE, the next step is to create the Integrator object that will perform the integration.
After defining our ODE we can create the integrator by calling the :code:`integrator` method attached to each ASSET ODE.
In the most basic case the code::`integrator` method takes just the desired step size for the integration.
If an adaptive integration step is preferred (as it often is more applications where speed is desirable), each :code:`integrator` object has an attached :code:`adaptive` method.
By default the flag is set to :code:`False`, but can be set to true as shown below.

.. code-block:: python

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)
    integrator.Adaptive = True

Now with our integrator setup we can integrate the ODE attached to the integrator through its :code:`integrate_dense` method.
As always we will first need an initial state to start our integration with.
Here we will use the same initial problem formulation that can be found in the :ref:`Phase Tutorial`.


.. code-block:: python
    
    R0 = 1
    V0 = 1
    RF = 1.1

    IState = np.zeros((10))
    IState[0] = R0
    IState[4] = V0
    IState[7] = 1

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)
    integrator.Adaptive = True

    FinalTime = 2.5 * np.pi
    NumStates = 500
    IG = integ.integrate_dense(IState,FinalTime , NumStates)

After initializing our :code:`IState` telling our integrator where to start, we can call the :code:`integrate_dense` method, which takes the desired initial state, as well as the final time to stop the integration.
The last argument in the function call is the desired number of states to return in the output.
With that, :code:`IG` will be populated with the result of the integration.
But one limitation of the above code is that it is assumed that the control variables are constant throughout the integration, which may not be desirable for highly chaotic dynamics where a good initial guess may be required.
In support of this ASSET features integrator constructors for applying a control law directly to an integration.

Control Laws
############

Continuing with the above example, a few simple changes allows one to add a control law to an :code:`integrator` object.
If, for example, we wished to generate an initial guess for a solar sailing transfer trajectory, a control law that orients the control vector of the sail in the prograde direction may be useful.
This control law will orient the sail throughout the integration to direct the normal vector of the sail in the prograde direction, in this case providing a better initial guess for the optimal control problem, as opposed to flying with a fixed sail orientation.
The constructor for the integrator with a control law is very similar to the typical integrator formation, however now the function and relevant state variables must be passed as arguments to the constructor.
:code:`ProgradeFunc` is passed to the constructor, along with the integers 0-5, which tells the integrator to use the first 6 state variables from each state during integration.
Note that the number of output variables from the control law **must** match the number of controls defined in the ODE, and the order of return variables must be the same as the order of those in the state.
In this case we defined that our ODE, :code:`Two_Body_SolarSail_ODE`, has 3 control variables and our integrator control law returns 3 as well.

.. code-block:: python

    def ProgradeFunc():
        args = Args(6)
        rhat = args.head3().normalized()
        vhat = args.tail3().normalized()
        return (rhat + vhat).normalized()


    integ = ode.integrator(0.01, ProgradeFunc(), range(0, 6))
    IG = integ.integrate_dense(IState, 2.5 * np.pi, 500)

Performing the integration after defining our control law is the same as previously shown, by calling :code:`integrate_dense` on the :code:`integrator` object and passing the initial state, final time, and the number of states to return.

State Transition Matrices
#########################

As well as having the capbility to integrate states, the ASSET integrators are also capable of integrating state transition matrices (STMs).
Integrating STMs can be done in one at a time or in parallel.
Integrating an STM requires no additional parameters to be done, as long as the relevant ODE has been defined.
For completeness' sake we will continue to use the previously defined ODE :code:`Full_TwoBody_SolarSail_Model` and the same integrator instance :code:`integ`.
The STM integration will return the STM of the full ODE, including the time, final time, and control variables as well as the final integrated state in the form of a tuple.
This results in an integrated STM that is of the size of the ODE by ODE + 1.
If the integrator that is being used to generate the STM has been defined with a control law, that control law will also be used during the integration.
Keep in mind, however, that the derivatives with respect to the controls will be 0, as the initial controls will be modified throughout the course of the integration.
If the integrator does not have a control law, then the controls will be propagated as constants throughout the integration.
The :code:`integrate_stm` function is used practically the same way as the standard integration method, as shown below.
Additionally, we can also integrate many STMs in parallel, to decrease the amount of time required to evaluate large numbers of states.
The parallel evaluation simply takes a list of initial states and times, and the number of cores that should be used for the integration.

.. code-block:: python

    #single STM and state
    #The state and STM are returned as a tuple, so we can unpack both of them at the same time
    finalState, stmSingle = integ.integrate_stm(IState, 2.5*np.pi)
    
    #We will now do many STMs in parallel
    stateList = []
    timeList = []
    #For this example we will initialize our input vectors with the same states and times, but this is not a requirement
    #All of the initial states be different and have their own final times to integrate to
    for i in range(0, 100):
        stateList.append(IState)
        timeList.append(2.5*np.pi)

    #The only difference in arguments is that integrate_stm_parallel takes a list of states and times,
    #as well as the number of cores to use
    #The method also returns the final list of states and STMs
    fileStateVec, stmVec = integ.integrate_stm_parallel(stateList, timeList, 8)

Parallel Integration
####################
Similar to the parallel integration of the STMS, :code:`integrate_dense` features a similar method, :code:`integrate_dense_parallel`.
Using the list of states and times produced in our parallel STM example we can write:

.. code-block:: python

    #number of threads to use with the parallel integration
    numthreads = 8
    #Returns a list of lists of the vector outputs of the integration, one for each initial state and time.
    IG_Parallel = integ.integrate_dense_parallel(stateList, timeList, numthreads)

Likewise with the STM parallel integration, the return value is a list of the vector results for each initial state and time.
As a word of caution, the input size for the initial states and times **must** be the same size.
    


Full Source Listing
-------------------

.. code-block:: python

    import asset as ast

    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments
    Tmodes = oc.TranscriptionModes


    def SolarSail_Acc(r, n, scale):
        ndr2 = vf.dot(r, n).squared()
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc


    def Full_TwoBody_SolarSail_Model(mu, beta):
        args = Args(10)
        r = args.head_3()
        v = args.segment_3(3)
        n = args.tail_3()
        acc = -mu * r.normalized_power3() + SolarSail_Acc(r, n, beta * mu)
        return vf.Stack([v, acc])


    Two_Body_SolarSail_ODE = oc.ode_x_u.ode(Full_TwoBody_SolarSail_Model(1, 0.02), 6, 3)

    phase = Two_Body_SolarSail_ODE.phase(Two_Body_SolarSail_ODE, Tmodes.LGL3)

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)
    integrator.Adaptive = True

    R0 = 1
    V0 = 1
    RF = 1.1

    IState = np.zeros((10))
    IState[0] = R0
    IState[4] = V0
    IState[7] = 1

    integrator = Two_Body_SolarSail_ODE.integrator(0.01)
    integrator.Adaptive = True

    FinalTime = 2.5 * np.pi
    NumStates = 500
    IG = integ.integrate_dense(IState,FinalTime , NumStates)

    def ProgradeFunc():
        args = Args(6)
        rhat = args.head3().normalized()
        vhat = args.tail3().normalized()
        return (rhat + vhat).normalized()


    integ = ode.integrator(0.01, ProgradeFunc(), range(0, 6))
    IG = integ.integrate_dense(IState, 2.5 * np.pi, 500)

    #single STM and state
    #The state and STM are returned as a tuple, so we can unpack both of them at the same time
    finalState, stmSingle = integ.integrate_stm(IState, 2.5*np.pi)
    
    #We will now do many STMs in parallel
    stateList = []
    timeList = []
    #For this example we will initialize our input vectors with the same states and times, but this is not a requirement
    #All of the initial states be different and have their own final times to integrate to
    for i in range(0, 100):
        stateList.append(IState)
        timeList.append(2.5*np.pi)

    #number of threads to use with the parallel integration
    numthreads = 8
    #The only difference in arguments is that integrate_stm_parallel takes a list of states and times,
    #as well as the number of cores to use
    #The method also returns the final list of states and STMs
    fileStateVec, stmVec = integ.integrate_stm_parallel(stateList, timeList, 8)

    #Integrating a list of initial states and times with integrate_dense_parallel
    #Returns a list of lists of the vector outputs of the integration, one for each initial state and time.
    IG_Parallel = integ.integrate_dense_parallel(stateList, timeList, numthreads)


