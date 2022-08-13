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
        
        ur = args.UVar(0)
        ut = args.UVar(1)
        
        rdot = vr
        thetadot =  vt/r
        vrdot    =  (vt**2)/r - 1/(r**2) + amax*ur
        vtdot    = -(vt*vr)/r            + amax*ut
        
       
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
    idxs = [0,2,3,5,6]
    for i in range(0,5):
        

        axs[i].plot(TO[4],TO[idxs[i]],color='r')
        axs[i].plot(MO[4],MO[idxs[i]],color='b')
        axs[i].set_ylabel(labs[i])
        axs[i].grid(True)

   

    ax4.set_xlabel(r"$t$")

    ax5.plot(np.cos(TO[1])*TO[0],np.sin(TO[1])*TO[0],color='r',label='Time Optimal')
    ax5.plot(np.cos(MO[1])*MO[0],np.sin(MO[1])*MO[0],color='b',label='Mass Optimal')


    ax5.scatter(1,0,color='k')
    ax5.scatter(np.cos(TO[1][-1])*TO[0][-1],np.sin(TO[1][-1])*TO[0][-1],color='r',marker = '*')
    ax5.scatter(np.cos(MO[1][-1])*MO[0][-1],np.sin(MO[1][-1])*MO[0][-1],color='b',marker = '*')


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
    
    RF = 1.75
    VF = np.sqrt(1/RF)
    
    
    IG = np.zeros((7))
    IG[0]=1
    IG[3]=1
    IG[5]=0.01
    IG[6]=.6
    
    def RFunc(x):return (x[0]>RF)
    
    
    MoptIG = integ.integrate_dense(IG,130,1000,RFunc)
    
    tf = MoptIG[-1][4]
    
    phase = ode.phase("LGL3",MoptIG,400)
    #phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front",range(0,5),IG[0:5])
    #phase.addLUSquaredNormBound("Path",[5,6], 0.00001, 1, 1.0)
    
    
    phase.addUpperNormBound("Path",[5,6],  1, 1.0)
    phase.addLowerSquaredNormBound("Path",[5,6], 0.000001, 1.0)

    
    #phase.addLUNormBound("Path",[5,6], 0.00001, 1, 1.0)

    phase.addBoundaryValue("Back",[0,2,3,4],[RF,0,VF,tf])
    
    
    #phase.optimizer.set_QPOrderingMode("MINDEG")

    phase.optimizer.PrintLevel = 0
    phase.optimizer.MaxAccIters = 200
    phase.optimizer.BoundFraction = .99
    phase.optimizer.deltaH = 1.0e-6
    phase.optimizer.decrH = .333

    phase.optimizer.KKTtol = 1.0e-6
    phase.optimizer.QPThreads = 8
    phase.Threads=8

    phase.addIntegralObjective(Args(2).norm()/10,[5,6])
    phase.solve_optimize()
    MassOptimal = phase.returnTraj()
    

    
    
    
    
    
    Plot(MoptIG,MassOptimal)


##############################################################################

