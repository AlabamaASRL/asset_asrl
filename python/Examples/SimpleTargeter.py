import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import seaborn as sns

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
solvs = ast.Solvers


def CalcPoints(mu):
        P1 = np.array([-mu,0,0])
        P2 = np.array([1.0-mu,0,0])
        L4 = np.array([.5 - mu, np.sqrt(3.0) / 2.0, 0.0,0,0,0])
        L5 = np.array([.5 - mu, -np.sqrt(3.0) / 2.0, 0.0,0,0,0])
        
        ###L1
        gamma0 = pow((mu*(1.0 - mu)) / 3.0, 1.0 / 3.0)
        guess = gamma0 + 1.0
        while (abs(guess - gamma0) > 10e-15):
            gamma0 = guess
            guess = pow((mu*(gamma0 - 1)*(gamma0 - 1.0)) / (3.0 
                        - 2.0 * mu - gamma0 * (3.0 - mu - gamma0)), 1.0 / 3.0);
        L1 = np.array([1 - mu - guess, 0, 0,0,0,0]);

		###L2
        gamma0 = pow((mu*(1.0 - mu)) / 3.0, 1.0 / 3.0);
        guess = gamma0 + 1.0;
        while (abs(guess - gamma0) > 10e-15):
            gamma0 = guess;
            guess = pow((mu*(gamma0 + 1)*(gamma0 + 1.0)) / (3.0 
                        - 2.0 * mu + gamma0 * (3.0 - mu + gamma0)), 1.0 / 3.0);
        L2 =  np.array([1 - mu + guess, 0, 0,0,0,0]);

		#### L3
        gamma0 = pow((mu*(1.0 - mu)) / 3.0, 1.0 / 3.0);
        guess = gamma0 + 1.0;
        while (abs(guess - gamma0) > 10e-15): 
            gamma0 = guess;
            guess = pow(((1.0 - mu)*(gamma0 + 1)*(gamma0 
                         + 1.0)) / (1.0 + 2.0 * mu + gamma0 * (2.0 
                               + mu + gamma0)), 1.0 / 3.0);  
        L3 =  np.array([-mu - guess, 0, 0,0,0,0])
        
        return L1,L2,L3,L4,L5

def CR3BP(mu):
    irows = 7    
    args = vf.Arguments(irows)
    r = args.head3()
    v = args.segment3(3)
    
    x    = args[0]
    y    = args[1]
    xdot = args[3]
    ydot = args[4]
    
    rterms = vf.stack([2*ydot + x,
                       -2.0*xdot +y]).padded_lower(1)
    
    p1loc = np.array([-mu,0,0])
    p2loc = np.array([1.0-mu,0,0])
    
    g1 = (r-p1loc).normalized_power3()*(mu-1.0)
    g2 = (r-p2loc).normalized_power3()*(-mu)
    
   
    acc = vf.sum([g1,g2,rterms])
    return oc.ode_6.ode(vf.stack(v,acc))


def Plot(T,ax = plt,color='k',label='Spacecraft',linestyle='solid'):
    Traj = np.array(T).T
    ax.plot(Traj[0],Traj[1],color=color,label=label,linestyle = linestyle)
    

def Targeter(integ,x0t,xft):
    
    Xdtmp = Args(7)
    iargs= vf.stack(Xdtmp.head(6),0,Xdtmp[6])
    Xfreal = integ.vf().eval(iargs).head(6)
    
    XdtXf = Args(13)
    Xdt = XdtXf.head(7)
    Xf  = XdtXf.tail(6)
    eq1 = Xdt.head(3)-x0t
    eq2= Xf.head3()-xft
    return vf.stack([eq1,eq2]).eval(vf.stack([Xdtmp,Xfreal]))
    


def SimpleTargeter():
    mu = 0.0121505856
    lstar = 384400
    tstar = 375190
    vstar = lstar/tstar
    
    ode = CR3BP(mu)
    
    atol = 1.000e-11
    integ = ode.integrator(np.pi/1000)
    integ.setAbsTol(atol)
    integ.Adaptive=True
    integ.MinStepSize = integ.DefStepSize/30000
    LiS = CalcPoints(mu)
    
    IG = np.zeros(7)
    IG[0] = 300000/lstar
    IG[4] = .5/vstar
    IG[5] = .5/vstar
    
    Rf = np.array([500000,-90000,200000])/lstar
    
    
    dt = 10*3600*24/tstar
    
    
    Traj = integ.integrate_dense(IG,dt)
    
    
    OTc = Targeter(integ,Traj[0][0:3],Rf)
    
    X0 = np.zeros((7))
    X0[0:6]=Traj[0][0:6]
    X0[6]=Traj[-1][6]
    
    
    prob = ast.Solvers.OptimizationProblem()
    prob.setVars(X0)
    prob.addEqualCon(OTc,range(0,7))
    prob.addObjective( (Args(3).head3()-X0[3:6]).norm() , [3,4,5])
    prob.optimizer.OptLSMode = solvs.LineSearchModes.L1
    prob.optimizer.PrintLevel = 0
    prob.solve()
    Xc = prob.returnVars()
    
    IGI = np.zeros((7))
    IGI[0:6] = Xc[0:6]
    dt = Xc[6]
    Trajc = integ.integrate_dense(IGI,dt)
    
    
    prob.optimize()
    Xc = prob.returnVars()

    IGI = np.zeros((7))
    IGI[0:6] = Xc[0:6]
    dt = Xc[6]
    Trajd = integ.integrate_dense(IGI,dt)
    
    ##########################################################
    plt.scatter(1-mu,0,color='grey',label='Moon',zorder=10)
    
    plt.scatter(LiS[0][0],LiS[0][1],label='L'+str(1),zorder=10)
    plt.scatter(Rf[0],Rf[1],label='Target')


    Plot(Traj,color='k',label='Initial')
    Plot(Trajc,color='b',label='Least Norm')
    Plot(Trajd,color='g',label='Optimal')

    plt.xlabel("X(ND)")
    plt.ylabel("Y(ND)")
    plt.grid(True)

    plt.legend()
    plt.show()
    ##########################################################
    
if __name__ == "__main__":
   
    SimpleTargeter()
    
