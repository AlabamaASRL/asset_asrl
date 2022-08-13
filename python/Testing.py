import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes

PhaseRegs = oc.PhaseRegionFlags



##############################################
X = Args(3)

x1 = X[0]
x2 = X[1]
x3 = X[2]

f = vf.ifelse(x1<x2,
              x3,
              x3*5.0)

print(f.compute([1.0,2.0,1.0])) # prints 1
print(f.compute([3.0,2.0,1.0])) # prints 5
########################################


#input("s")

def Plot(Traj,name,col,ax=plt,linestyle='-'):
    TT = np.array(Traj).T
    ax.plot(TT[0],TT[1],label=name,color=col,linestyle=linestyle)
def Scatter(State,name,col,ax=plt):
    ax.scatter(State[0],State[1],label=name,c=col)
def ThrottlePlot(Traj,name,col,ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[6],(TT[7]**2 + TT[8]**2+ TT[9]**2)**.5,label=name,color=col)    
def ThrottlePlot2(Traj,name,col,ax=plt):
    TT = np.array(Traj).T
    ax.plot(TT[6],TT[7],label=name + " x",color=col)   
    ax.plot(TT[6],TT[8],label=name + " y",color=col)
    ax.plot(TT[6],TT[9],label=name + " z",color=col)    
    ax.plot(TT[6],(TT[7]**2 + TT[8]**2+ TT[9]**2)**.5,label=name +" |mag|",color=col)    


class TestModel(oc.ode_x_x_x.ode):
    def __init__(self,mu,ltacc):
        
        Xvars = 6
        Uvars = 3
        ############################################################
        args = oc.ODEArguments(Xvars,1,Uvars)
        r = args.head3()
        v = args.segment3(3)
        throttle = args[7]
        u = args.tail3()
        g = r.normalized_power3()*(-mu)
        thrust = u.normalized()*ltacc*throttle
        acc = g + thrust
        ode =  vf.Stack([v,acc])
        #############################################################
        super().__init__(ode,Xvars,1,Uvars)


ode = TestModel(1,.02)


integ = ode.integrator(.01)

X0 = np.zeros((11))

X0[0]=1
X0[4]=1
X0[7]=.99
X0[9]=.99

TOF = 2.0

TrajIG = integ.integrate_dense(X0,TOF,100)

###########################################
Plot(TrajIG,"Initial Guess",'blue')
Scatter(X0,"X0",'black')
plt.grid(True)
plt.axis("Equal")
plt.show()
###########################################
#ode.vf().SuperTest(X0,1000000)

phase= ode.phase(Tmodes.LGL3, TrajIG, 350)
#phase.setControlMode(Cmodes.BlockConstant)

phase.addBoundaryValue(PhaseRegs.Front,range(0,7),X0[0:7])
phase.addLUVarBound(PhaseRegs.Path,7,.01,1.0,1.0)

phase.addLUNormBound(PhaseRegs.ODEParams,[0,1,2],.9,1.1,1.0)
phase.addBoundaryValue(PhaseRegs.Back,[6],[TOF])

def C3():
    x = Args(6)
    r = x.head3()
    v = x.tail3()
    return v.dot(v) - 2/r.norm()
phase.addStateObjective(PhaseRegs.Back,-C3(),range(0,6))


#phase.solve()

#phase.optimizer.QPOrderingMode = ast.QPOrderingModes.MINDEG

phase.optimize()

TrajConv = phase.returnTraj()

TT = np.array(TrajConv).T

plt.plot(TT[6],TT[7])
plt.show()





###########################################
Plot(TrajIG,"Initial Guess",'blue')
Plot(TrajConv,"Optimal",'red')

Scatter(X0,"X0",'black')
plt.grid(True)
plt.axis("Equal")
plt.show()
###########################################


#phase.transcribe(True,True)











