.. _scale-guide

=================================
Auto Scaling and Component Naming
=================================
Beginning in version 0.5.0, we have made significant improvements to the ODE, phase and OCP interfaces
in order to simplify the definition of more complex optimal control problems. Note that these changes are backwards
compatible and the previous tutorials and any existing code will continue to work as is. We will be updating all existing
tutorials to reflect the new changes, at a later date, but for now we will provide a brief overview of the new features. A subset of
the existing examples have been updated to reflect the new interface and can be found in the examples/NewInterface folder on the repo.


Component Naming
================

The first major improvement is the ability to give string name aliases to the input arguments of an ODE in the constructor.
These names may then be used in place of integer indexes when applying constraints and objectives to phases
and optimal control problems. As an example, lets take a look at how this works for the ODE defined for the Delta-III
launch example.

Here, we define the ODE exactly as we normally would in terms of ODEArguments. We can then add the different components of
the arguments to a dictionary indexed by the name or names that we want to use as aliases. Multiple aliases for the same variable
grouping can be used by indexing the dictionary with a tuple of string names. The variables of the alias can be specified with
the list integer indices,the actual segment or element, or a list of segments and/or elements. After populating the dictionary,we pass it 
to the Vgroups (Variable groups) argument of ODEBase's constructor.

.. code:: python
    
    class RocketODE(oc.ODEBase):
        def __init__(self,T,mdot):
            ####################################################
            XtU  = oc.ODEArguments(7,3)
        
            R = XtU.XVec().head(3)
            V = XtU.XVec().segment(3,3)
            m = XtU.XVar(6)
       
            U = XtU.UVec()
        
            h       = R.norm() - Re
            rho     = RhoAir * vf.exp(-h / h_scale)
            Vr      = V + R.cross(np.array([0,0,We]))
        
            D       = (-0.5*CD*S)*rho*(Vr*Vr.norm())
        
            Rdot    =  V
            Vdot    =  (-mu)*R.normalized_power3() + (T*U.normalized() + D)/m
        
            ode = vf.stack(Rdot,Vdot,-mdot)
        
            Vgroups = {}


            Vgroups[("R","Position")]=R
            #Vgroups[("R","Position")]=[0,1,2] # same as above

            Vgroups[("V","Velocity")]=V
            Vgroups[("U","ThrustVec")]=U

            Vgroups["RV"] = [R,V]
            #Vgroups["RV"] = [0,1,2,3,4,5] # same as above

            Vgroups["Xt"] = XtU.XtVec()

            Vgroups[("t","time")]=XtU.TVar()
            Vgroups[("m","mass")]=m

            ####################################################
            super().__init__(ode,7,3,Vgroups = Vgroups)


Having defined the variable groups for our ODE we can now the string names themselves or lists of string names
anywhere where we could previously use integer indices or lists of indices. As an example, lets take a look at how this would modify the
definition of the delta-III launch vehicle optimal control problem below. For example, we can now simply use the alias "U" which we have
defined to be our thrust vector when placing an LUNorm bound on our control. Similarly, we can replace the range(0,8) in the boundary value specification,
with a list of string names specifying the different components of the states or with the combined alias "Xt" of those same variables. Note that when specifying multiple names,
the combined index vector is just the concatenation of sub index vectors, so the order matters.

.. code:: python

    #########################################
    phase1 = ode1.phase(tmode,IG1,nsegs1)
    phase1.setControlMode(cmode)

    phase1.addLUNormBound("Path","U",.5,1.5)
    ## Same as above
    ## phase1.addLUNormBound("Path",[8,9,10],.5,1.5)

    phase1.addBoundaryValue("Front",["R","V","m","t"],IG1[0][0:8])
    ## Same as above
    ##phase1.addBoundaryValue("Front","Xt",IG1[0][0:8])
    ##phase1.addBoundaryValue("Front",range(0,8),IG1[0][0:8])


    phase1.addLowerNormBound("Path","R",Re*.999999)

    phase1.addBoundaryValue("Back","time",tf_phase1) ## Can drop brackets if scalar
    ## Same as above
    ## phase1.addBoundaryValue("Back",[7],[tf_phase1])


    #########################################
    phase2 = ode2.phase(tmode,IG2,nsegs2)
    phase2.setControlMode(cmode)

    phase2.addLowerNormBound("Path","R",Re)
    phase2.addLUNormBound("Path","U",.5,1.5)

    phase2.addBoundaryValue("Front","mass", m0_phase2)
    phase2.addBoundaryValue("Back", "time" ,tf_phase2)

    #########################################
    phase3 = ode3.phase(tmode,IG3,nsegs3)
    phase3.setControlMode(cmode)

    phase3.addLowerNormBound("Path","R",Re)
    phase3.addLUNormBound("Path","U",.5,1.5)
    phase3.addBoundaryValue("Front","mass", m0_phase3)
    phase3.addBoundaryValue("Back", "time" ,tf_phase3)

    #########################################
    phase4 = ode4.phase(tmode,IG4,nsegs4)
    phase4.setControlMode(cmode)

    phase4.addLowerNormBound("Path","R",Re)
    phase4.addLUNormBound("Path","U",.5,1.5)
    phase4.addBoundaryValue("Front","mass", m0_phase4)
    phase4.addUpperVarBound("Back","time",tf_phase4)

    phase4.addEqualCon("Back",TargetOrbit(at,et,istart,Ot,Wt),["R","V"])
    ## Same as above
    ## phase4.addEqualCon("Back",TargetOrbit(at,et,istart,Ot,Wt),"RV")

    # Maximize final mass
    phase4.addValueObjective("Back","mass",-1.0)

    #########################################

We can also now use these same string names when applying any link constraints and objectives to OptimalControlProblem objects as well.
So for the delta 3 example, we can modifiy the ForwardLinkEqualCon as shown below. Note that you will need to define the string aliases in the ODE
associated with each phase. Note however that the indices specified by a string name do not have to be the same in every phase/ODE linked
(though they are in this case). This makes it much easier to enforce continuity between variables in each phase/ODE even if they
have different indices. For example, in the old interface, models with different numbers of state variables would have time with a different index.
Now, so long as the user names time say "t" in both model definitions, then the call below will enforce continuity correctly. 

.. code:: python

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)



    ## All phases continuous in everything but mass (var 6)
    ocp.addForwardLinkEqualCon(phase1,phase4,["R","V","t","U"])

    ## Same as above
    ##ocp.addForwardLinkEqualCon(phase1,phase4,[0,1,2,3,4,5, 7,8,9,10])
    

Finally, the new string names or lists of names can also be used when applying specifying the inputs to control law for an ODE's integrator as shown below.

.. code:: python

	    ode = RocketODE(T_phase1,mdot_phase1)

        integ = ode.integrator(1.0,Args(3).normalized(),"V")


    


Auto-Scaling
============

The second major addition to the interface is automatic problem scaling from user defined canonical units. In the
Delta-III and Shuttle tutorials we emphasized the importance of defining problems in non-dimensional units. This is typically
done by defining a set of base length, mass, and time units and then redefining all constants and boundary conditions in this new 
unit system. This is easy enough for simple problems like the Delta-III, but quickly becomes cumbersome for more complex scenarios.
For that we reason, we have added interfaces to phase and optimal control problem that will handle this non-dimensionalization behind the scenes
and allow users to specify their problem in traditional units. To use this interface, the user must enable auto-scaling and declare the canonical units
associated with each ODE input variables for a phase. As shown below for the Delta-III example, we can specify the units using phase.setUnits 
by passing a single vector with the same dimensions as the ODE's input vector, or if we defined names for our components, we can assign them by name.

.. code:: python

    phase1 = ode1.phase(tmode,IG1,nsegs1)
    phase1.setControlMode(cmode)
    
    ## Enable AutoScaling, off by default
    phase1.setAutoScaling(True)
    
    units = np.ones((11))
    units[0:3]=Lstar
    units[3:6]=Vstar
    units[6]=Mstar
    units[7]=Tstar
    ## All others are one,i.e no auto-scaling
    
    phase1.setUnits(units)  # As a single vector
    # Or
    phase1.setUnits(R=Lstar,V=Vstar,t=Tstar,m=Mstar) 

    
    phase1.addLUNormBound("Path","U",.5,1.5)
    phase1.addBoundaryValue("Front",["R","V","m","t"],IG1[0][0:8])

    #. Continue definition
    #.
    #


The specified canonical units will be used under the hood to non-dimensionalizes any trajectory passed into the phase and any variables sent to the optimizer.
From the units for the states and times, we can uniquely determine a set of output scales for the transcription defect constraints that will make them equivalent
to a problem that was non-dimensionalized by hand. 

..  note:: 

    When auto-scaling is enabled the mesh tolerance for adaptive mesh refinement refers to the scaled ODE system.


However, since we don't track the physical units of functions, this is not possible for all other constraints and objectives added to phase.
By default for all non-dynamics constraints and objectives, we compute a set of output scales that normalizes each row of
the functions Jacobian at the initial guess for the problem. Alternatively, the user can override these scales manually. 
All of this is controlled an optional "AutoScale" argument that has been,
added to all phase and ocp .add### methods. As an example, lets take a look at a portion of the definition of the Delta-III problem again below.


.. code:: python

    ## AutoScale = "auto" if not specified
    phase4.addBoundaryValue("Front","mass", m0_phase4)
    phase4.addUpperVarBound("Back","time",tf_phase4)
    # AutoScale=None, will turn it off for this constraint
    phase4.addLUNormBound("Path","U",.5,1.5,AutoScale=None)

    # Override the scale for this constraint
    phase4.addLowerNormBound("Path","R",Re,AutoScale=1/Lstar)
    phase4.addEqualCon("Back",TargetOrbit(at,et,istart,Ot,Wt),["R","V"],AutoScale = [1/Lstar,1.0,1.0,1.0,1.0])
    
    # Maximize final mass
    phase4.addValueObjective("Back","mass",-1.0)



By default AutoScale is set to "auto" for all constraints and objectives. This will work well in most cases, but can be overridden when the user can specify a better scale factor.
Manual scales specified by a assigning a scalar or vector of scales to the AutoScaling parameter. These will multiply the output of the function whenever AutoScaling is enabled.
For example, for the bound on "R", we know that the units of the output will have dimensions of length, so it is reasonable to set the AutoScale variable 1/Lstar.
Similarly, for the TargetOrbit constraint, we know that the first component of the output(semi-major axis) has dimensions of length and all others already non-dimensional.
In that case, we can manually specify the output of the first component and then leaves the others set to 1.0.


When adding multiple phases to an OCP we should also enable AutoScaling for the OCP object as well. This will enable auto-scaling
on all linked constraints and objectives between phases. It should also be noted that units do not have to be the same for all phases in an OCP.
As with phases, the optional AutoScale parameter on all link constraints and objectives can be overridden
with custom scales if necessary.  

.. code:: python

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    
    # Enable AutoScaling for the OCP,and all constitiuent phases currently in ocp
    ocp.setAutoScaling(True,True)
    ocp.setAdaptiveMesh(True)  
    
    
    for phase in ocp.Phases:
        phase.setUnits(R=Lstar,V=Vstar,t=Tstar,m=Mstar)
        phase.setMeshTol(1.0e-6)
        phase.setMeshErrorCriteria('max')
        phase.setMeshErrorEstimator('integrator')  


    ## Each Phase does not have to have the same AutoScale units even if its the same ODE
    phase4.setUnits(R=2*Lstar,V=Vstar,t=.8*Tstar,m=Mstar)

    ## Can override the AutoScale for any link constraints and objectives as well
    ocp.addForwardLinkEqualCon(phase1,phase4,["R","V","t","U"],AutoScale="auto")

	