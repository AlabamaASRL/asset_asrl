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
##############################################################################
class MountainCar(oc.ode_x_u.ode):
    
    def __init__(self):
        
        args = oc.ODEArguments(2,1)
        x = args.XVar(0)
        v = args.XVar(1)
        u = args.UVar(0)
        
        xdot = v
        vdot = .001*u -.0025*vf.cos(3*x)
        
        ode = vf.stack([xdot,vdot])
        super().__init__(ode,2,1)

def Plot(Traj):
    TT = np.array(Traj).T
    fig = plt.figure()

    ax0 = plt.subplot(321)
    ax1 = plt.subplot(323)
    ax2 = plt.subplot(325)
    ax3 =plt.subplot(122)

    xs = np.linspace(-1.7,0.7,1000)
    cols = sns.color_palette("viridis",len(TT[0]))

    f = lambda x: np.sin(3*x)/3.0 

    ax3.plot(xs,f(xs),color='k')
    ax3.scatter(TT[0],f(TT[0]),color = cols,alpha=.7)
    ax3.grid(True)
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$h$')

    axs =[ax0,ax1,ax2]
    labs = [r'$x$', r'$v$', r'$u$']
    for i,var in enumerate([0,1,3]):
        
        for j in range(0,len(TT[0])-1):
            axs[i].plot(TT[2][j:j+2],TT[var][j:j+2],color=cols[j])
            
        axs[i].set_ylabel(labs[i])
        axs[i].grid(True)


    axs[2].set_xlabel(r'$t$')

    fig.set_size_inches(15.0, 7.5, forward=True)
    plt.show()
    
def Animate(Traj):
    
    T= np.array(Traj).T
    xs = np.linspace(-1.7,0.7,1000)
    
    fig = plt.figure()
    f = lambda x: np.sin(3*x)/3.0 
    ax0 = plt.subplot(321,xlim=(-2, max(T[2])*1.05), ylim=(min(T[0])*1.2, max(T[0])*1.1))
    ax1 = plt.subplot(323,xlim=(-2, max(T[2])*1.05), ylim=(min(T[1])*1.2, max(T[1])*1.1))
    ax2 = plt.subplot(325,xlim=(-2, max(T[2])*1.05), ylim=(min(T[3])*1.2, max(T[3])*1.1))    
    ax3 = fig.add_subplot(122,xlim=(-1.7,0.7), ylim=(-.4, .4))
    ax3.plot(xs,f(xs),color='k')
    ax3.grid(True)
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$h$')

    axs =[ax0,ax1,ax2]
    labs = [r'$x$', r'$v$', r'$u$']
    
    objs = []
    for i,var in enumerate([0,1,3]):
        objs.append(axs[i].plot([],[])[0])
        axs[i].set_ylabel(labs[i])
        axs[i].grid(True)
        
    objs.append(ax3.plot([],[],marker='o')[0])
    def init():
        for i,var in enumerate([0,1,3]):
            objs[i].set_data([],[])
            
        objs[3].set_data([],[])
        return objs
    
    def animate(j):
        for i,var in enumerate([0,1,3]):
            objs[i].set_data(T[2][0:j],T[var][0:j])
            
        objs[3].set_data([T[0][j],T[0][j]],[f(T[0][j]),f(T[0][j])])
        return objs
            
    ani = animation.FuncAnimation(fig, animate, frames=len(T[0]),
                                  interval=60, blit=True, init_func=init,
                                  repeat_delay=5000)
    fig.set_size_inches(15.5, 7.5, forward=True)
    plt.show()
            



##############################################################################
ode = MountainCar()

x0 = -.5
v0 = 0
xf = .52
tf = 500

IG = [[x0 + (xf-x0)*t/tf, 
       t/tf,
       t,
       np.sin(t/tf)] for t in np.linspace(0,tf,100)]

phase = ode.phase(Tmodes.LGL3,IG,128)
phase.addBoundaryValue("First",[0,1,2],[x0,v0,0])
phase.addBoundaryValue("Last",[0],[xf])
phase.addLowerVarBound(PhaseRegs.Back,1,0.0,1.0)
phase.addLUVarBound(PhaseRegs.Path,0,-1.2,.55,1.0)
phase.addLUVarBound(PhaseRegs.Path,1,-0.07,.07,100.0)  # Scale to Be order 1
phase.addLUVarBound(PhaseRegs.Path,3,-1,1,1.0)
phase.addDeltaTimeObjective(0.01) # Scale to Be order 1

phase.optimizer.set_OptLSMode("L1")
phase.optimizer.MaxAccIters = 200
phase.optimizer.PrintLevel = 1
phase.optimizer.KKTtol = 1.0e-9

phase.solve_optimize()

Traj = phase.returnTraj()

Plot(Traj)
Animate(Traj)

################################################################



#############################################################################



