import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

'''
Source for problem formulation
https://www.hindawi.com/journals/aaa/2014/851720/

'''
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
    MoptIG = integ.integrate_dense(IState,160,1000,RFunc)
    
    
    phase = ode.phase("LGL3",ToptIG,400)
    phase.integrator.setAbsTol(1.0e-14)
    
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("Front",range(0,5),IState[0:5])
    phase.addLUVarBound("Path",5, 0.01/2, 1, 100.0)
    phase.addLUVarBound("Path",6, -np.pi, np.pi, 1.0)

    phase.addBoundaryValue("Back",[0,2,3],[RF,0,VF])
    
    phase.optimizer.set_PrintLevel(1)
    phase.optimizer.set_MaxAccIters(350)
    phase.optimizer.set_MaxIters(1000)
    phase.optimizer.set_BoundFraction(.995)
    phase.optimizer.deltaH = 1.0e-6
    phase.optimizer.set_EContol(1.0e-10)
    phase.AdaptiveMesh = False
    phase.MeshTol =1.0e-6
    phase.MeshErrorEstimator='integrator'
    phase.NeverDecrease = False
    # Scale to be order 1 based on initial guess
    phase.addDeltaTimeObjective(1/100)
    phase.solve_optimize_solve()
    
    
    
    ts1,merr1,mdist1 = phase.getMeshInfo(False,100)
    ts2,merr2,mdist2 = phase.getMeshInfo(True,100)

    me1 = np.array(mdist1).T
    me2 = np.array(mdist2).T

    plt.plot(ts1,me1,color='r')
    plt.plot(ts2,abs(me2),color='b')
    plt.yscale("log")
    plt.show()
    
    
    
    
    
    
    TimeOptimal = phase.returnTraj()
    
    phase.removeStateObjective(0)
    
    phase.setTraj(MoptIG,600)
    
    phase.addIntegralObjective(Args(1)[0]/100,[5])
    
    #phase.addUpperDeltaTimeBound(MoptIG[-1][4]*1.3)
    #phase.AdaptiveMesh = False
    ## This problem likes to grind, could probabably
    # be improved by making integral a state variable
    phase.optimize_solve()
    
    ts1,merr1,mdist1 = phase.getMeshInfo(False,100)
    ts2,merr2,mdist2 = phase.getMeshInfo(True,100)

    me1 = np.array(mdist1).T
    me2 = np.array(mdist2).T

    plt.plot(ts1,me1,color='r')
    plt.plot(ts2,abs(me2),color='b')
    plt.yscale("log")
    plt.show()
    
    MassOptimal = phase.returnTraj()
    
    Plot(TimeOptimal,MassOptimal)


##############################################################################







    