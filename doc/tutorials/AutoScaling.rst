.. _scale-guide

=================================
Auto Scaling and Component Naming
=================================
Beginning in version 0.4.0, we have made significant improvements to the phase and ODE interfaces
in order to simplify the definition of more complex optimal control problems. Note that these changes are backwards
compatible and the previous tutorials and any existing code will continue to work as is.


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

            Vgroups[("t","time")]=XtU.TVar()
            Vgroups[("m","mass")]=m

            ####################################################
            super().__init__(ode,7,3,Vgroups = Vgroups)


Auto-Scaling
============