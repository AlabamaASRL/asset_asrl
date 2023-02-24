import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast
from asset_asrl.Astro.Extensions.ThrusterModels import CSIThruster
from asset_asrl.Astro.AstroModels import MEETwoBody_CSI
from asset_asrl.Astro.FramePlot import TBPlot,colpal
import asset_asrl.Astro.Constants as c
from asset_asrl.OptimalControl.MeshErrorPlots import PhaseMeshErrorPlot


##############################################################################
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

'''
This is a widely used low-thrust optimization test problem of computing
a mass optimal transfer from Earth to Dionysus.

Im basing the initial and terminal conditons and non-dimensional scaling
off of the Junkins Paper below.

https://arc.aiaa.org/doi/abs/10.2514/1.G003686


'''
def RadFunc():
    args = Args(6)
    p = args[0]
    f = args[1]
    g = args[2]
    h = args[3]
    k = args[4]
    L = args[5]
    
    sinL = vf.sin(L)
    cosL = vf.cos(L)
    
    w = 1.+f*cosL +g*sinL
    r = (p/w)
    
    return r
def GTO():
    Isp_dim  = 3100        #S
    Tmag_dim = .5         #N
    
    nd = 0
    
    tf_dim   = (6+nd)*c.day  #s 
    mass_dim = 100        #kg
   
    ast.SoftwareInfo()
    thruster = CSIThruster(Tmag_dim, Isp_dim, mass_dim)
    ode = MEETwoBody_CSI(c.MuEarth,c.RadiusEarth,thruster)
    
    
    tf = tf_dim/ode.tstar
    integ = ode.integrator(.1)
    
    ## Already Non Dimesonalized in the correct units from junkins paper
    X0 = np.array([1.8226, 0.725, 0.0, 0.06116, 0.0, 0.])
    XF = np.array([6.611, 0.0, 0.0, 0.0, 0.0, 53.4071 + 0*2*np.pi])
    
    
    print(53.4071/(2*np.pi))
    Istate = np.zeros((11))
    Istate[0:6]=X0
    Istate[6]=1     # Full mass is non-dimensionalized to one
    Istate[9]=.5
    
    
    ts = np.linspace(0,tf,500)
    
    TrajIG = []
    
    # Lerp initial guess
    for t in ts:
        State = np.zeros((11))
        Xi = X0 + (XF-X0)*t/tf
        State[0:6]=Xi
        State[6]=1
        State[7]=t
        State[9]=.1
        TrajIG.append(State)
        
    n =300
    #TrajIG = integ.integrate_dense(TrajIG[0],tf,1000)
    phase = ode.phase("LGL3",TrajIG,n)
    #phase.setControlMode("NoSpline")  # This problem only likes this
    phase.setControlMode("BlockConstant")  # This problem only likes this

    phase.integrator.setAbsTol(1.0e-11)
    phase.addBoundaryValue("Front",range(0,8),Istate[0:8])
    phase.addLowerFuncBound("Path",RadFunc(),range(0,6),.99,1.0)

    phase.addLUNormBound("Path",range(8,11),.01/2,1,1) # Lowbound is leakmass
    phase.addBoundaryValue("Back",[7],[tf])
    phase.addBoundaryValue("Back",range(0,6),XF[0:6])
    phase.addValueObjective("Back",6,-.1)
    #phase.addIntegralObjective(Args(3).squared_norm(),[8,9,10])
    #phase.enable_vectorization(False)
    
    
    phase.setThreads(8,8)
    phase.optimizer.set_OptLSMode("AUGLANG")
    phase.optimizer.set_SoeLSMode("L1")

    phase.optimizer.set_MaxLSIters(3)
    phase.optimizer.set_MaxAccIters(100)
    phase.optimizer.set_BoundFraction(.997)
    phase.optimizer.set_PrintLevel(1)
    #phase.optimizer.set_deltaH(1.0e-6)
    #phase.optimizer.set_decrH(.333)
    phase.optimizer.deltaH=1.0e-6
    phase.MeshTol = 1.0e-6
    phase.RelSwitchTol = .1
    phase.ForceOneMeshIter=True
    phase.DetectControlSwitches = True
    phase.optimizer.MaxAccIters=350
    phase.optimizer.MaxIters=700

    phase.optimizer.set_KKTtol(1.0e-5)

    phase.optimizer.set_EContol(1.0e-7)
    #phase.optimizer.set_QPOrderingMode("MINDEG")
    import time

    phase.MeshErrorEstimator = 'integrator'
    #phase.MeshErrorDistributor='geometric'
    #phase.AdaptiveMesh = True
    phase.MeshIncFactor = 5
    t00 = time.perf_counter()

    #phase.refineTrajEqual(n)
    #phase.solve()
    phase.AdaptiveMesh = True
   
    phase.optimize()

    
    PhaseMeshErrorPlot(phase,show=True)
    #phase.addBoundaryValue("Back",range(5,6),XF[5:6])
    #phase.optimize()
    

    
    ConvTraj = phase.returnTraj()
    
    FinalMass = ConvTraj[-1][6]*mass_dim
    
    print("Final Mass   :",FinalMass," kg")
    print("Mass Expended:",mass_dim - FinalMass," kg")

    
    
    tff = time.perf_counter()
    
    TT = np.array(ConvTraj).T

    plt.plot(TT[7],TT[1])
    plt.show()
    
    
    fig,axs = plt.subplots(1,2)
    
    TT = np.array(ConvTraj).T
    axs[0].plot(TT[7],TT[8],label=r'$U_r$')
    axs[0].plot(TT[7],TT[9],label=r'$U_t$')
    axs[0].plot(TT[7],TT[10],label=r'$U_n$')
    axs[0].plot(TT[7],(TT[8]**2+TT[9]**2+TT[10]**2)**.5,label=r'$|U|$')
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel(r"$t$")

    Earth = []
    Dion = []
    for ang in np.linspace(0,2*np.pi,1000):
        XE = np.zeros((6))
        XE[0:5]=X0[0:5]
        XE[5] =ang
        Earth.append(ast.Astro.modified_to_cartesian(XE,1))
        
        XD = np.zeros((6))
        XD[0:5]=XF[0:5]
        XD[5] =ang
        Dion.append(ast.Astro.modified_to_cartesian(XD,1))

    

    

    TrajCart   = ode.MEEToCartesian(ConvTraj)
    TrajIGCart = ode.MEEToCartesian(TrajIG)
    
    plot = TBPlot(ode)
    plot.addTraj(TrajCart, "Transfer",color='r')
    #plot.addTraj(TrajIGCart, "Lerped Initial Guess",color='b')
    
    plot.addTraj(Earth, "Earth",color='g',linestyle='--')
    plot.addTraj(Dion, "Dionysus",color='b',linestyle='--')

    plot.addPoint(TrajCart[-1], "Arrival",color='k',marker='*',markersize=80)
    plot.addPoint(TrajCart[0], "Departure",color='k',marker='o',markersize=30)

    plot.Plot2dAx(axs[1],legend=True,view=[1,2])
    #axs[1].axis("Equal")
    axs[1].grid(True)
    
    plt.show()
    
    


def Other():
    Isp_dim  = 3000        #S
    Tmag_dim = .6          #N
    tf_dim   = 420*c.day   #s 
    mass_dim = 1000        #kg
   
    ast.SoftwareInfo()
    thruster = CSIThruster(Tmag_dim, Isp_dim, mass_dim)
    ode = MEETwoBody_CSI(c.MuSun,c.AU,thruster)
    
    
    tf = tf_dim/ode.tstar
    integ = ode.integrator(.1)
    
    ## Already Non Dimesonalized in the correct units from junkins paper
    X0 = np.array([1.000064,-0.003764, 0.015791,-1.211e-5,-4.514e-6,5.51356])
    XF = np.array([2.328616,-0.191235,-0.472341,0.033222,0.085426,4.96395 + 2.0*np.pi])
    
    print(ast.Astro.modified_to_classic(XF,1))
    Istate = np.zeros((11))
    Istate[0:6]=X0
    Istate[6]=1     # Full mass is non-dimensionalized to one
    Istate[9]=.5
    
    
    ts = np.linspace(0,tf,500)
    
    TrajIG = []
    
    # Lerp initial guess
    for t in ts:
        State = np.zeros((11))
        Xi = X0 + (XF-X0)*t/tf
        State[0:6]=Xi
        State[6]=1
        State[7]=t
        State[9]=.9
        TrajIG.append(State)
        
        
    #TrajIG = integ.integrate_dense(TrajIG[0],tf,1000)
    phase = ode.phase("LGL7",TrajIG,15)
    #phase.setControlMode("NoSpline")  # This problem only likes this
    phase.setControlMode("HighestOrderSpline")  # This problem only likes this

    phase.integrator.setAbsTol(1.0e-11)
    phase.addBoundaryValue("Front",range(0,8),Istate[0:8])
    phase.addLUNormBound("Path",range(8,11),.001,1,1) # Lowbound is leakmass
    phase.addBoundaryValue("Back",[7],[tf])
    phase.addBoundaryValue("Back",range(0,6),XF[0:6])
    #phase.addValueObjective("Back",6,-1.0)
    phase.addIntegralObjective(Args(3).squared_norm(),[8,9,10])
    #phase.enable_vectorization(False)
    phase.RelSwitchTol = .1
    phase.ForceOneMeshIter=True
    phase.DetectControlSwitches = True
    
    phase.setThreads(8,8)
    phase.optimizer.set_OptLSMode("AUGLANG")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_MaxAccIters(300)
    phase.optimizer.set_BoundFraction(.996)
    phase.optimizer.set_PrintLevel(1)
    #phase.optimizer.set_deltaH(1.0e-6)
    phase.optimizer.set_decrH(.333)
    phase.MeshTol = 1.0e-8
    phase.MeshErrFactor=10.0
    phase.optimizer.set_EContol(1.0e-8)
    #phase.optimizer.set_QPOrderingMode("MINDEG")
    import time

    #phase.MeshErrorEstimator = 'integrator'
   
    #phase.AdaptiveMesh = True
    phase.MeshIncFactor = 5
    t00 = time.perf_counter()

    phase.solve()
    
    
    PhaseMeshErrorPlot(phase,show=True)
    #phase.addBoundaryValue("Back",range(5,6),XF[5:6])
    #phase.optimize()
    

    
    ConvTraj = phase.returnTraj()
    
    FinalMass = ConvTraj[-1][6]*mass_dim
    
    print("Final Mass   :",FinalMass," kg")
    print("Mass Expended:",mass_dim - FinalMass," kg")

    
    
    tff = time.perf_counter()
    
    TT = np.array(ConvTraj).T

    plt.plot(TT[7],TT[5])
    plt.show()
    
    
    fig,axs = plt.subplots(1,2)
    
    TT = np.array(ConvTraj).T
    TT[7]=TT[7]/TT[7][-1]
    axs[0].plot(TT[7],TT[8],label=r'$U_r$')
    axs[0].plot(TT[7],TT[9],label=r'$U_t$')
    axs[0].plot(TT[7],TT[10],label=r'$U_n$')
    axs[0].plot(TT[7],(TT[8]**2+TT[9]**2+TT[10]**2)**.5,label=r'$|U|$')
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel(r"$t$")

    Earth = []
    Dion = []
    for ang in np.linspace(0,2*np.pi,1000):
        XE = np.zeros((6))
        XE[0:5]=X0[0:5]
        XE[5] =ang
        Earth.append(ast.Astro.modified_to_cartesian(XE,1))
        
        XD = np.zeros((6))
        XD[0:5]=XF[0:5]
        XD[5] =ang
        Dion.append(ast.Astro.modified_to_cartesian(XD,1))

    

    

    TrajCart   = ode.MEEToCartesian(ConvTraj)
    TrajIGCart = ode.MEEToCartesian(TrajIG)
    
    plot = TBPlot(ode)
    plot.addTraj(TrajCart, "Transfer",color='r')
    #plot.addTraj(TrajIGCart, "Lerped Initial Guess",color='r')
    
    plot.addTraj(Earth, "Earth",color='g',linestyle='--')
    plot.addTraj(Dion, "Dionysus",color='b',linestyle='--')

    plot.addPoint(TrajCart[-1], "Arrival",color='k',marker='*',markersize=80)
    plot.addPoint(TrajCart[0], "Departure",color='k',marker='o',markersize=30)

    plot.Plot2dAx(axs[1],legend=True)
    axs[1].axis("Equal")
    axs[1].grid(True)
    
    plt.show()


if __name__ == "__main__":
   
    GTO()
    Other()
    
    Isp_dim  = 3000        #S
    Tmag_dim = .32         #N
    tf_dim   = 3534*c.day  #s 
    mass_dim = 4000        #kg
   
    ast.SoftwareInfo()
    thruster = CSIThruster(Tmag_dim, Isp_dim, mass_dim)
    ode = MEETwoBody_CSI(c.MuSun,c.AU,thruster)
    
    
    tf = tf_dim/ode.tstar
    integ = ode.integrator(.1)
    
    ## Already Non Dimesonalized in the correct units from junkins paper
    X0 = np.array([0.99969,-0.00376, 0.01628,-7.702e-6, 6.188e-7, 14.161])
    XF = np.array([1.5536, 0.15303,-0.51994, 0.01618, 0.11814, 46.3302])
    
    Istate = np.zeros((11))
    Istate[0:6]=X0
    Istate[6]=1     # Full mass is non-dimensionalized to one
    Istate[9]=.5
    
    
    ts = np.linspace(0,tf,500)
    
    TrajIG = []
    
    # Lerp initial guess
    for t in ts:
        State = np.zeros((11))
        Xi = X0 + (XF-X0)*t/tf
        State[0:6]=Xi
        State[6]=1
        State[7]=t
        State[9]=.5
        TrajIG.append(State)
        
        
    phase = ode.phase("LGL7",TrajIG,100)
    phase.setControlMode("BlockConstant")  # This problem only likes this
    phase.integrator.setAbsTol(1.0e-11)
    phase.addBoundaryValue("Front",range(0,8),Istate[0:8])
    phase.addLUNormBound("Path",range(8,11),.001,1,1) # Lowbound is leakmass
    phase.addBoundaryValue("Back",[7],[tf])
    phase.addBoundaryValue("Back",range(0,6),XF[0:6])
    phase.addValueObjective("Back",6,-1.0)
    #phase.addIntegralObjective(Args(3).squared_norm(),[8,9,10])
    #phase.enable_vectorization(False)
    phase.RelSwitchTol = .1
    phase.ForceOneMeshIter=True
    
    phase.setThreads(8,8)
    phase.optimizer.set_OptLSMode("AUGLANG")
    phase.optimizer.set_MaxLSIters(2)
    phase.optimizer.set_MaxAccIters(300)
    phase.optimizer.set_BoundFraction(.996)
    phase.optimizer.set_PrintLevel(1)
    phase.optimizer.set_deltaH(1.0e-6)
    phase.optimizer.set_decrH(.333)
    phase.MeshTol = 1.0e-10
    phase.MeshErrFactor=2000.0

    #phase.DetectControlSwitches = True
    phase.optimizer.set_EContol(1.0e-11)
    phase.optimizer.set_KKTtol(1.0e-8)

    phase.optimizer.set_QPOrderingMode("MINDEG")
    import time

    phase.MeshErrorEstimator = 'integrator'
    
    phase.AdaptiveMesh = True
    phase.MeshIncFactor = 5
    t00 = time.perf_counter()

    phase.optimize()
    #phase.optimize()
    #phase.optimize()

    tff = time.perf_counter()
    PhaseMeshErrorPlot(phase,show=True)

    print(1000*(tff-t00))

    
    ts2,merr2,mdist2 = phase.getMeshInfo(True,100)

    
    ConvTraj = phase.returnTraj()
    
    
    ConvTrajI = phase.returnTraj()

    Tab  = phase.returnTrajTable()
    
   

    
       
       
        
    
    #phase.refineTrajEqual(500)
    #phase.optimize()
    Tab2 = phase.returnTrajTable()
    
    
    ConvTraj = phase.returnTraj()

    FinalMass = ConvTraj[-1][6]*mass_dim
    
    print("Final Mass   :",FinalMass," kg")
    print("Mass Expended:",mass_dim - FinalMass," kg")

    
    
    integ = ode.integrator(.1,Tab2)
    integ.setAbsTol(1.0e-13)
   
    ## Do this for non-blockconstant control or if you dont care about exact accuracy
    
    t00 = time.perf_counter()

    ReintTraj1 = integ.integrate_dense(ConvTraj[0],ConvTraj[-1][7])
    tff = time.perf_counter()
    
    print(1000*(tff-t00))
    
    t00 = time.perf_counter()

    ## This is to be preffered if control is blockconstant
    ReintTraj2 = [ConvTraj[0]]    
    for i in range(0,len(ConvTraj)-1):
        Next = integ.integrate_dense(ReintTraj2[-1],ConvTraj[i+1][7])[1::]
        ReintTraj2+=Next
    
    
    
    print(ReintTraj1[-1]-ConvTraj[-1]) # Less Accurate but still fine
    print(ReintTraj2[-1]-ConvTraj[-1]) # More Accurate
    t00 = time.perf_counter()
    phase.calc_global_error()
    
    tff = time.perf_counter()
    print(1000*(tff-t00))
    print(phase.calc_global_error())

    #########################################################################
    TT = np.array(ConvTraj).T
    plt.plot(TT[7]/TT[7][-1],(TT[8]**2+TT[9]**2+TT[10]**2)**.5,label=r'$|U|$',marker='o')

  
    plt.show()
    '''
    ts, errs,errint = Tab.NewErrorIntegral()
    plt.plot(ts,errs)
    #plt.plot(ts,errint)

    ts, errs,errint = Tab2.NewErrorIntegral()
    plt.plot(ts,errs,color='red')
    '''
    ts2,merr2,mdist2 = phase.getMeshInfo(True,200)
    plt.plot(ts2,mdist2,color='r')
    
    ts2,merr2,mdist2 = phase.getMeshInfo(False,200)
    plt.plot(ts2,mdist2,color='b')
    #plt.plot(ts,errint)
    plt.yscale("log")
    plt.show()
    
    #######################################################################
    
   
    
    fig,axs = plt.subplots(1,2)
    
    TT = np.array(ConvTraj).T
    axs[0].plot(TT[7],TT[8],label=r'$U_r$')
    axs[0].plot(TT[7],TT[9],label=r'$U_t$')
    axs[0].plot(TT[7],TT[10],label=r'$U_n$')
    axs[0].plot(TT[7],(TT[8]**2+TT[9]**2+TT[10]**2)**.5,label=r'$|U|$')
    axs[0].grid()
    axs[0].legend()
    axs[0].set_xlabel(r"$t$")

    Earth = []
    Dion = []
    for ang in np.linspace(0,2*np.pi,1000):
        XE = np.zeros((6))
        XE[0:5]=X0[0:5]
        XE[5] =ang
        Earth.append(ast.Astro.modified_to_cartesian(XE,1))
        
        XD = np.zeros((6))
        XD[0:5]=XF[0:5]
        XD[5] =ang
        Dion.append(ast.Astro.modified_to_cartesian(XD,1))

    
    plt.show()

    fig,ax = plt.subplots(1,1)
    
    axs =[ax]

    TrajCart   = ode.MEEToCartesian(ConvTraj)
    TrajIGCart = ode.MEEToCartesian(TrajIG)
    
    plot = TBPlot(ode)
    plot.addTraj(TrajCart, "Transfer",color='r')
    #plot.addTraj(TrajIGCart, "Lerped Initial Guess",color='r')
    
    plot.addTraj(Earth, "Earth",color='g',linestyle='--')
    plot.addTraj(Dion, "Dionysus",color='b',linestyle='--')

    plot.addPoint(TrajCart[-1], "Arrival",color='k',marker='*',markersize=80)
    plot.addPoint(TrajCart[0], "Departure",color='k',marker='o',markersize=30)

    plot.Plot2dAx(axs[0],legend=True)
    axs[0].axis("Equal")
    axs[0].grid(True)
    
    plt.show()
            
        
        
        
    
    
    
    
  
    
    

    

        

