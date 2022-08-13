import asset as ast
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time 


norm = np.linalg.norm
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
solvs = ast.Solvers



def RosenBrockObj(xy = Args(2)):
    x = xy[0]
    y = xy[1]
    return (1-x)**2 + 100*((y-x**2))**2
def RosenBrockCon1():
    return Args(2).squared_norm()-2.0
def RosenBrockCon2(xy = Args(2)):
    x = xy[0]
    y = xy[1]
    c1 = (x -1)**3 -y + 1
    c2 = x+y-2
    return vf.stack(c1,c2)
def RosenBrockObjPlot(x,y):
    return (1-x)**2 + 100*((y-x**2))**2

def MishraObj(xy = Args(2)):
    x = xy[0]
    y = xy[1]
    return vf.sin(y)*vf.exp((1-vf.cos(x))**2) + vf.cos(x)*vf.exp((1-vf.sin(y))**2) + (x-y)**2
def MishraCon(xy = Args(2)):
    x = xy[0]
    y = xy[1]
    return (xy+[5,5]).squared_norm()-25.0
def MishraObjPlot(x,y):
    return np.sin(y)*np.exp((1-np.cos(x))**2) + np.cos(x)*np.exp((1-np.sin(y))**2) + (x-y)**2
    

def Mishra():
    
    Ipoint = [-0,-0]
    
    
    prob = solvs.OptimizationProblem()
    
    prob.setVars(Ipoint)
    prob.addObjective(MishraObj(),[0,1])
    prob.addInequalCon(MishraCon(), [0,1])
    prob.optimizer.OptLSMode = solvs.LineSearchModes.L1
    prob.optimizer.PrintLevel = 0
    prob.JetJobMode= solvs.JetJobModes.Optimize
    prob.optimize()
    
    Fpoint = prob.returnVars()
    x = np.linspace(-10.1,.1, 1000)
    y = np.linspace(-10.1, .1, 1000)
    
    X, Y = np.meshgrid(x, y)
    Z = MishraObjPlot(X, Y)
    
    plt.contourf(X, Y, Z, 10, cmap='viridis')
    plt.colorbar();
    plt.scatter(Ipoint[0],Ipoint[1],color='k',marker = "o")
    plt.scatter(Fpoint[0],Fpoint[1],color='r',marker = "*")
    
    thetas = np.linspace(0,6.283,100)
    
    plt.plot(np.cos(thetas)*5-5,np.sin(thetas)*5-5,color='k',linestyle='--')
    
    plt.show()
    
def Rosen():
    
    Ipoint = [-1,1]
    
    
    prob = solvs.OptimizationProblem()
    prob.setVars(Ipoint)
    prob.addObjective(RosenBrockObj(),[0,1])
    prob.addInequalCon(RosenBrockCon2(), [0,1])
    #prob.optimizer.OptLSMode = solvs.LineSearchModes.L1
    prob.optimizer.PrintLevel = 0
    prob.optimize()
    
    Fpoint = prob.returnVars()
    x = np.linspace(-1.5,1.5, 1000)
    y = np.linspace(-1.5,1.5, 1000)
    
    X, Y = np.meshgrid(x, y)
    Z = RosenBrockObjPlot(X, Y)
    
    plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar();
    plt.scatter(Ipoint[0],Ipoint[1],color='k',marker = "o")
    plt.scatter(Fpoint[0],Fpoint[1],color='r',marker = "*")
    
    thetas = np.linspace(0,2*np.pi,500)
    
    plt.plot(np.cos(thetas)*np.sqrt(2),np.sin(thetas)*np.sqrt(2),color='k',linestyle='--')
    
    plt.show()    
    
#############
def RastigrinObj(x= Args(1)[0], A = 10):
    return A + x**2 - A*vf.cos(2*np.pi*x)

def RastigrinObjPlot(x,y,A=10):
    return A + x**2 - A*np.cos(2*np.pi*x) + A + y**2 - A*np.cos(2*np.pi*y)



def Rastigrin():
    
    x = np.linspace(-5,5, 1000)
    y = np.linspace(-5,5, 1000)
    
    X, Y = np.meshgrid(x, y)
    Z = RastigrinObjPlot(X, Y)
    n = 1000
    Ipoints = []
    
    probs = []
    for i in range(0,n):
        Ipoint =[rand.uniform(-5, 5),rand.uniform(-5, 5)]
        prob = solvs.OptimizationProblem()
        prob.setVars(Ipoint)
        prob.addObjective(RastigrinObj(),[[0],[1]])
        prob.addInequalCon(Args(1)[0]-5,[[0],[1]])
        prob.addInequalCon(-Args(1)[0]-5,[[0],[1]])
        
        prob.optimizer.OptLSMode = solvs.LineSearchModes.L1
        prob.optimizer.PrintLevel = 0
        prob.JetJobMode= solvs.JetJobModes.Optimize
        probs.append(prob)
        
    t0 = time.perf_counter()
    probs = solvs.Jet.map(probs,2,True)
    tf = time.perf_counter()
    
    print((tf-t0)*1000)
    
    Fpoints = [p.returnVars() for p in probs]
    
    Fp = np.array(Fpoints).T
    
    plt.scatter(Fp[0],Fp[1],color='k',zorder=30)
    
    plt.contourf(X, Y, Z, 20, cmap='viridis')
    plt.colorbar();
    #plt.scatter(Ipoint[0],Ipoint[1],color='k',marker = "o")
    #plt.scatter(Fpoint[0],Fpoint[1],color='r',marker = "*")
    
    thetas = np.linspace(0,7,100)
    
    
    
    
    plt.show()    
    

Rosen()
Mishra()  
Rastigrin()
