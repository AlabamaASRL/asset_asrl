import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes

###############################################################################    

class LTModel(oc.ODEBase):
    
    def __init__(self, amax):
        Xvars = 4
        Uvars = 2
        
        ############################################################
        XtU  = oc.ODEArguments(Xvars,Uvars)
        
        r,theta,vr,vt = XtU.XVec().tolist()
        
        u,alpha = XtU.UVec().tolist()
       
        
       
        rdot = vr
        thetadot =  vt/r
        vrdot    =  (vt**2)/r - 1/(r**2) + amax*u*vf.sin(alpha)
        vtdot    = -(vt*vr)/r            + amax*u*vf.cos(alpha)
        
       
        ode = vf.stack([rdot,thetadot,vrdot,vtdot])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)



############################################################################### 

def Plot(TimeOptimal,MassOptimal):
    fig = plt.figure()

    ax0 = plt.subplot(521)
    ax1 = plt.subplot(523)
    ax2 = plt.subplot(525)
    ax3 = plt.subplot(527)
    ax4 = plt.subplot(529)

    ax5 =plt.subplot(122)


    TO = np.array(TimeOptimal).T
    MO = np.array(MassOptimal).T

    axs =[ax0,ax1,ax2,ax3,ax4]
    labs =[r'$r$',r'$v_r$',r'$v_t$',r'$u$',r'$\alpha$']
    idxs = [0,2,3,5]
    for i in range(0,4):
        

        axs[i].plot(TO[4],TO[idxs[i]],color='r')
        axs[i].plot(MO[4],MO[idxs[i]],color='b')
        axs[i].set_ylabel(labs[i])
        axs[i].grid(True)

    axs[4].plot(TO[4],TO[6],color='r')
    axs[4].plot(MO[4],MO[6],color='b')
    axs[4].set_ylabel(labs[4])
    axs[4].grid(True)

    ax4.set_xlabel(r"$t$")

    ax5.plot(np.cos(TO[1])*TO[0],np.sin(TO[1])*TO[0],
             color='r',label='Time Optimal')
    ax5.plot(np.cos(MO[1])*MO[0],np.sin(MO[1])*MO[0],
             color='b',label='Mass Optimal')


    ax5.scatter(1,0,color='k')
    ax5.scatter(np.cos(TO[1][-1])*TO[0][-1],
                np.sin(TO[1][-1])*TO[0][-1],color='r',marker = '*')
    ax5.scatter(np.cos(MO[1][-1])*MO[0][-1],
                np.sin(MO[1][-1])*MO[0][-1],color='b',marker = '*')


    angs = np.linspace(0,2*np.pi,1000)

    ax5.plot(np.cos(angs),np.sin(angs),label=r'$r_0$',linestyle ='dotted',color='k')
    ax5.plot(RF*np.cos(angs),RF*np.sin(angs),label=r'$r_f$',linestyle ='dashed',color='k')
    ax5.grid(True)

    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.axis("Equal")
    ax5.legend()

    fig.set_size_inches(15.0, 7.5, forward=True)
    fig.tight_layout()


    plt.show()
    

###############################################################################    

if __name__ == "__main__":
   
    amax = .01
    ode = LTModel(amax)
    integ = ode.integrator(.01)
    
    
    
    RF = 4.0
    VF = np.sqrt(1/RF)
    
    
    IState = np.zeros((7))
    IState[0]=1
    IState[3]=1
    IState[5]=.99
    IState[6]=0
    
    def RFunc(x):return (x[0]>RF)
    
    ToptIG = integ.integrate_dense(IState,130,1000,RFunc)
    IState[5]=.5
    MoptIG = integ.integrate_dense(IState,130,1000,RFunc)
    
    
    phase = ode.phase("LGL3",ToptIG,400)
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front",range(0,5),IState[0:5])
    phase.addLUVarBound("Path",5, 0.0001, 1, 1.0)
    phase.addLUVarBound("Path",6, -1*np.pi, 1*np.pi, 1.0)

    phase.addBoundaryValue("Back",[0,2,3],[RF,0,VF])
    
    phase.optimizer.PrintLevel = 0
    phase.optimizer.MaxAccIters = 100
    phase.optimizer.BoundFraction = .998
    phase.optimizer.deltaH = 1.0e-6
   
    
    
    # Scale to be order 1 based on initial guess
    dtscale = 1/ToptIG[-1][4]
    phase.addDeltaTimeObjective(dtscale)
    phase.solve_optimize_solve()
    
    TimeOptimal = phase.returnTraj()
    
    phase.removeStateObjective(0)
    
    phase.setTraj(MoptIG,400)
    
    # Scale to be order 1 based on initial guess
    integscale = 1/MoptIG[-1][4]
    print(integscale)
    phase.addIntegralObjective(Args(1)[0]*integscale,[5])
    
    #phase.addUpperDeltaTimeBound(MoptIG[-1][4]*1.00)
    
    phase.optimize_solve()
    phase.refineTrajManual(800)
    phase.optimize_solve()
    MassOptimal = phase.returnTraj()
    
    Plot(TimeOptimal,MassOptimal)


##############################################################################







    