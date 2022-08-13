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

# Create a point at which to optimize
X0 = np.array([3,7])
# Number of segments
nseg = 10000
# Create a reference trajectory
t = np.linspace(0, 10, nseg)
x = np.linspace(-4, 4, nseg)
y = x**2
traj = np.stack([x, y, t], axis=1)
# Create an interpolation table from this data
traj_tab = oc.LGLInterpTable(2, traj, traj.shape[0])
print(traj_tab)
# Create an interpfunction VectorFunction type
fun = oc.InterpFunction(traj_tab, [0,1]).vf()
# print(fun.compute([4]))
# # Create an ode
# args = vf.Arguments(3)
# ode_xy = args.head2()
# ode_t = args[2]
# ode_xy_dot = args.Constant([0,0])
# xdot = ode_xy_dot
# ode = oc.ode_x_u.ode(xdot, 2, 0)
# Create an obj scalar function
args = vf.Arguments(1) # 2 X , 1 T, 0 U, 1 S
vf_t = args[0]
square_norm_vf = ( (fun.eval(vf_t) - X0).squared_norm() )**.5
print(square_norm_vf)
out = []
for tt in t:
    out.append(square_norm_vf.compute([tt]))



prob = solvs.OptimizationProblem()
prob.setVars([5.0])
#prob.addInequalCon(Args(1)[0]-10,[0])
#prob.addInequalCon(-Args(1)[0],[0])
prob.addObjective(square_norm_vf,[0])
prob.optimizer.OptLSMode = ast.Solvers.LineSearchModes.LANG
prob.optimize()
tf = prob.returnVars()[0]

print(square_norm_vf.compute([tf]))
print(square_norm_vf.jacobian([tf]))
print(square_norm_vf.adjointhessian([tf],[1]))

vf2 = vf.PyScalarFunction(1,square_norm_vf.compute)

ocp = oc.OptimalControlProblem()
fig = plt.figure()
ax = fig.gca()

oug = [ square_norm_vf.jacobian([tf])[0] for tf in t]
oug2 = [ vf2.jacobian([tf])[0] for tf in t]

outh = [ square_norm_vf.adjointhessian([tf],[1])[0] for tf in t]
outh2 = [ vf2.adjointhessian([tf],[1])[0] for tf in t]

ax.plot(t, out)

ax.grid(True)
plt.scatter(tf,square_norm_vf.compute([tf])[0])

#ax.plot(x, y)
#ax.scatter(X0[0], X0[1])
plt.show()




