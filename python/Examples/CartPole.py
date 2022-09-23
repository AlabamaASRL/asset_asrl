import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

vf = ast.VectorFunctions
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Args = vf.Arguments

class CartPole(oc.ode_x_u.ode):
    
    def __init__(self,l,m1,m2,g):
        
        args = oc.ODEArguments(4,1)
        
        q1  = args.XVar(0)
        q2  = args.XVar(1)
        q1d = args.XVar(2)
        q2d = args.XVar(3)
        
        q1,q2,q1d,q2d = args.XVec().tolist()
        
        u = args.UVar(0)
        
        q1dd = (l*m2*vf.sin(q2)*(q2d**2) + u + m2*g*vf.cos(q2)*vf.sin(q2))/( m1 + m2*((1-vf.cos(q2)**2)))
        q2dd = -1*(l*m2*vf.cos(q2)*vf.sin(q2)*(q2d**2) +u*vf.cos(q2) +(m1*g+m2*g)*vf.sin(q2))/( l*m1 + l*m2*((1-vf.cos(q2)**2)))
        
        ode = vf.stack([q1d,q2d,q1dd,q2dd])
        super().__init__(ode,4,1)
        
###############################################################################

def Plot(Traj):
    
    T = np.array(Traj).T
    
    
    P0X = T[0]
    P0Y = np.zeros_like(T[0])
    P1X  = T[0] + l*np.sin(T[1])
    P1Y  = -l*np.cos(T[1])
    
    n = len(P0X)
    
    cols=sns.color_palette("viridis",n)
    
    fig = plt.figure()
    ax0 = plt.subplot(321)
    ax1 = plt.subplot(323)
    ax2 = plt.subplot(325)
    ax3 = plt.subplot(122)
    
    axs = [ax0,ax1,ax2]
    
    for i, var in enumerate([0,1,5]):
        for j in range(0,len(T[0])-1):
            axs[i].plot(T[4][j:j+2],T[var][j:j+2],color=cols[j])
    
    
    
    ax3.plot(P0X,P0Y,color='k')
    ax3.plot(P1X,P1Y,color='k')
    
    for i in range(0,n):
        xs = [P0X[i],P1X[i]]
        ys = [P0Y[i],P1Y[i]]
        ax3.plot(xs,ys,color=cols[i],marker='o')
      
    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    
    ax0.set_ylabel(r'$X$')
    ax1.set_ylabel(r'$\theta$')
    ax2.set_ylabel(r'$U$')
    ax2.set_xlabel(r't')
    
    ax3.set_xlabel(r'$X$')
    ax3.set_ylabel(r'$Y$')
    fig.set_size_inches(10.5, 5.5, forward=True)
    
    fig.set_tight_layout(True)
    plt.show()

def Animate(Traj):
    T = np.array(Traj).T
    
    
    P0X = T[0]
    P0Y = np.zeros_like(T[0])
    P1X  = T[0] + l*np.sin(T[1])
    P1Y  = -l*np.cos(T[1])
    
    n = len(P0X)
    fig = plt.figure()
    
    ax0 = plt.subplot(321,xlim=(-.05, max(T[4])*1.05), ylim=(min(T[0])*1.2, max(T[0])*1.1))
    ax1 = plt.subplot(323,xlim=(-.05, max(T[4])*1.05), ylim=(min(T[1])*1.2, max(T[1])*1.1))
    ax2 = plt.subplot(325,xlim=(-.05, max(T[4])*1.05), ylim=(min(T[5])*1.2, max(T[5])*1.1))    
    ax3 = fig.add_subplot(122, aspect='equal',xlim=(-.4, 1.6), ylim=(-1.0, 1.0))

    ax3.grid(True)
    
    ax3.plot([-2,2],[0,0],color='k')

    pole, = ax3.plot([],[],marker="o")
    X, =ax0.plot([],[])
    theta, =ax1.plot([],[])
    U, =ax2.plot([],[])

    def init():
        pole.set_data([],[])
        U.set_data([],[])
        X.set_data([],[])
        theta.set_data([],[])
        return pole,X,theta,U

    def animate(i):
        xs = [P0X[i],P1X[i]]
        ys = [P0Y[i],P1Y[i]]
        pole.set_data(xs,ys)
        X.set_data(T[4][0:i+1],T[0][0:i+1])
        theta.set_data(T[4][0:i+1],T[1][0:i+1])
        U.set_data(T[4][0:i+1],T[5][0:i+1])
        
        return pole,X,theta,U

    ani = animation.FuncAnimation(fig, animate, frames=len(P0X),
                                  interval=60, blit=True, init_func=init,
                                  repeat_delay=5000)



    ax0.grid(True)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    
    ax0.set_ylabel(r'$X$')
    ax1.set_ylabel(r'$\theta$')
    ax2.set_ylabel(r'$U$')
    ax2.set_xlabel(r't')
    ax3.set_xlabel(r'$X$')
    ax3.set_ylabel(r'$Y$')
    fig.set_size_inches(15.5, 7.5, forward=True)

    plt.show()
    
    
##############################################################################        
        
if __name__ == "__main__":

    ast.SoftwareInfo()        
            
    m1 = 1
    m2 =.3
    l=.5
    g = 9.81
    
    umax = 20
    dmax = 2
    
    tf = 2
    d = 1
    
    ts = np.linspace(0,tf,100)
    IG = [[d*t/tf,np.pi*t/tf,0,0,t,.00] for t in ts]
    ode = CartPole(l,m1,m2,g)
    
    
    
    phase = ode.phase("LGL5",IG,64)
    phase.addBoundaryValue("Front",range(0,5),[0,0,0,0,0])
    phase.addBoundaryValue("Back",range(0,5),[d,np.pi,0,0,tf])
    phase.addLUVarBound("Path",5,-umax,umax,1.0)
    phase.addLUVarBound("Path",0,-dmax,dmax,1.0)
    phase.addIntegralObjective(Args(1)[0]**2,[5])
    phase.Threads=8
    
    #phase.optimizer.QPThreads = 8
    phase.optimizer.PrintLevel= 0
    phase.optimize()
    
    
    print("Total Time (Sum of all below)             :",phase.optimizer.LastTotalTime," s")
    print("Function/Derivative Eval Time             :",phase.optimizer.LastFuncTime," s")
    print("KKT Matrix Factor/Solve Time              :",phase.optimizer.LastKKTTime," s")
    print("KKT Matrix Pre-Analysis/Mem Alloc Time    :",phase.optimizer.LastPreTime," s")
    print("Miscellaneous (Mostly Console Print) Time :",phase.optimizer.LastMiscTime," s")
    
    
    Traj = phase.returnTraj()
   
    Plot(Traj)
    Animate(Traj)

    ###########################################################################


