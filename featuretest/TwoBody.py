import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast
from asset_asrl.Astro.Extensions.ThrusterModels import CSIThruster
from asset_asrl.Astro.AstroModels import MEETwoBody_CSI,TwoBody
from asset_asrl.Astro.FramePlot import TBPlot,colpal
import asset_asrl.Astro.Constants as c
from MeshErrorPlots import PhaseMeshErrorPlot


##############################################################################
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

print(1/6480)

def GetError(Traj):
    tsnd = []
    errs = []
    
    
    ode = TwoBody(1.0,1.0)
    integ = ode.integrator(.1)
    integ.setAbsTol(1.0e-13)
    for i in range(0,len(Traj)-1):
        Next = integ.integrate(Traj[i],Traj[i+1][6])
        
        err = abs(Next[0:6]-Traj[i+1][0:6]).max()
        
        errs.append(err)
        tsnd.append(Traj[i][6]/Traj[-1][6])
    return tsnd,errs
    


ode = TwoBody(1.0,1.0)


integ = ode.integrator(.1)

X0 = np.zeros((7))
X0[0]=1
X0[4]=1.1

Traj = integ.integrate_dense(X0,74,1000)

phase = ode.phase("LGL7",Traj,64)
phase.integrator.setAbsTol(1.0e-14)
phase.addBoundaryValue("Front",range(0,7),Traj[0][0:7])
phase.optimizer.EContol=1.0e-12
phase.addDeltaTimeEqualCon(Traj[-1][6])
ts1,merr1,mdist1 = phase.getMeshInfo(False,100)

phase.MeshTol =1.0e-7
Traj1 = phase.returnTraj()
phase.MeshErrorEstimator = 'integrator'
phase.MeshIncFactor = 9.01
phase.AdaptiveMesh = True
phase.optimizer.PrintLevel =2
phase.solve()

PhaseMeshErrorPlot(phase,show=True)


Times  = []
Errors = []



import time


n = 100
errs = np.linspace(0,1,n+1)


ts1,merr1,mdist1 = phase.getMeshInfo(False,100)
ts2,merr2,mdist2 = phase.getMeshInfo(True,100)

me1 = np.array(mdist1).T
me2 = np.array(mdist2).T

es=[]
for i in range(0,int((len(me2)-1)/3)):
    start = 3*i
    es.append(np.mean(me2[start:start+3]))
    
es.append(es[-1])

plt.plot(ts1,me1,color='r')
plt.plot(ts2,abs(me2),color='b')
plt.plot(ts1,es,color='k')


plt.yscale("log")
plt.show()

plt.plot(phase.MeshTimes,phase.MeshError)
plt.show()



Tab = phase.returnTrajTable()
ts, errs,errint = Tab.NewErrorIntegral()




Tab = phase.returnTrajTable()
Traj = phase.returnTraj()


#########################################
t1,e1 = GetError(Traj1)
t2,e2 = GetError(Traj)

plt.plot(t1,e1)
plt.plot(t2,e2)

plt.yscale("log")
plt.show()

print("SD")

############################################

print(len(ts))
fig,axs = plt.subplots(1,2)





for i in range(0,len(Times)):
    
    axs[0].plot(Times[i],Errors[i],label='Mesh {}'.format(0))
    
axs[0].set_yscale("log")

plot = TBPlot(ode)

plot.addTraj(Traj, "Converged",color='b')



plot.Plot2dAx(axs[1],legend=True)
axs[1].axis("Equal")
axs[1].grid(True)
plt.show()