import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes

###############################################################################    

class LTModel(oc.ode_x_u.ode):
    
    def __init__(self, amax):
        ############################################################
        args  = oc.ODEArguments(4,2)
        
        r     = args.XVar(0)
        theta = args.XVar(1)
        vr    = args.XVar(2)
        vt    = args.XVar(3)
        
        u = args.UVar(0)
        alpha = args.UVar(1)
        
        rdot = vr
        thetadot =  vt/r
        vrdot    =  (vt**2)/r - 1/(r**2) + amax*u*vf.sin(alpha)
        vtdot    = -(vt*vr)/r            + amax*u*vf.cos(alpha)
        
       
        ode = vf.stack([rdot,thetadot,vrdot,vtdot])
        ##############################################################
        super().__init__(ode,4,2)



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
    phase.addBoundaryValue("Front",range(0,5),IState[0:5])
    phase.addLUVarBound("Path",5, 0.001, 1, 1.0)
    phase.addLUVarBound("Path",6, -2*np.pi, 2*np.pi, 1.0)

    phase.addBoundaryValue("Back",[0,2,3],[RF,0,VF])
    
    phase.optimizer.PrintLevel = 0
    phase.optimizer.MaxAccIters = 200
    phase.optimizer.BoundFraction = .993
    phase.optimizer.deltaH = 1.0e-8
    phase.optimizer.QPThreads = 8
    phase.Threads=8
    
    #phase.optimizer.set_OptLSMode("L1")
    #phase.optimizer.set_OptLSMode("AUGLANG")
    phase.optimizer.MaxLSIters = 2
    

    phase.addDeltaTimeObjective(0.1)
    phase.solve_optimize()
    TimeOptimal = phase.returnTraj()
    
    phase.removeStateObjective(-1)
    
    phase.setTraj(MoptIG,400)
    phase.addIntegralObjective(Args(1)[0]/10,[5])
    
    phase.solve_optimize()
    #phase.refineTrajManual(600)
    phase.optimizer.KKTtol = 1.0e-8
    #phase.optimize()
    
    MassOptimal = phase.returnTraj()
    
    Plot(TimeOptimal,MassOptimal)


##############################################################################







    