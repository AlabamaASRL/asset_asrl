import numpy as np
import asset as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

vf = ast.VectorFunctions
Args = vf.Arguments

nx = 60
ny = 60

lim = np.pi

xs = np.linspace(-lim,lim,nx)
ys = np.linspace(-lim,lim,ny)

xs2 = np.linspace(-lim,lim,nx*3)
ys2 = np.linspace(-lim,lim,ny*3)

xis = np.linspace(-lim,lim,nx*82)

def f(x,y):
    return np.sin(x)*np.cos(y) 
def df(x,y):
    return np.array([ np.cos(x)*np.cos(y) ,-np.sin(x)*np.sin(y)   ])
def d2f(x,y):
    return np.array([[ -np.sin(x)*np.cos(y) ,-np.cos(x)*np.sin(y)   ],
                     [ -np.cos(x)*np.sin(y) ,-np.sin(x)*np.cos(y)   ]])


xs = list(xs)

ys = list(ys)

X, Y = np.meshgrid(xs, ys)

Z    = f(X,Y)

tabi = ast.VectorFunctions.InterpTable2D(xs,ys,Z,'cubic')

X, Y = np.meshgrid(xs2, ys2)

prob = ast.Solvers.OptimizationProblem()
prob.setVars([0,.1])
prob.addInequalCon(Args(1)[0]-lim,[[0],[1]])
prob.addInequalCon(-lim -Args(1)[0],[[0],[1]])

prob.addObjective(tabi.sf(),[0,1])
prob.optimize()


Z = tabi.interp(X,Y)

plt.contourf(X,Y,Z,levels=100)
plt.show()

#Z = f(X,Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

tab   = ast.VectorFunctions.InterpTable2D(xs2,ys2,Z,'cubic')
tab2  = ast.VectorFunctions.InterpTable2D(xs2,ys2,Z,'linear')


vals = np.linspace(0,1,11)
v = .000


func1 = tabi.vf()
func2 = tab2.vf()

print(func1.jacobian([1,1])-df(1,1))
print(func1.adjointhessian([1,1],[1])-d2f(1,1))


input("S")
func1_fd = vf.PyScalarFunction(2,lambda x: [tab.interp(x[0],x[1])]).vf()
func2_fd = vf.PyScalarFunction(2,lambda x: [tab2.interp(x[0],x[1])]).vf()



XX = [1,2]
L = [0]

#func1.rpt(XX,1000000)
print(func2.jacobian(XX))

print(func2.adjointhessian(XX,[1]))
print(func2_fd.adjointhessian(XX,[1]))

print(tab2.interp_deriv2(XX[0],XX[1])[2])

print(func1(XX)-func1_fd(XX))
print(func1.jacobian(XX)-func1_fd.jacobian(XX))
print(func1.adjointhessian(XX,L)-func1_fd.adjointhessian(XX,L))

print(func2.compute(XX)-func2_fd.compute(XX))
print(func2.jacobian(XX)-func2_fd.jacobian(XX))
print(func2.adjointhessian(XX,L)-func2_fd.adjointhessian(XX,L))



plt.show()


func = vf.PyScalarFunction(2,lambda x: [tab2.interp(x[0],x[1])]).vf()


x = 1.9
y = 1.1

#print(func.adjointhessian([x,y],[1]))
#fs2 = [tab2.interp_deriv1(x,y)[1] for y in xis]
#fsa = [df(x,y) for  y in xis]

fs2 = [tab.interp_deriv1(x,y)[1] for y in xis]
fs2a = [func.jacobian([x,y])[0] for y in xis]
fsa = [df(x,y) for  y in xis]

fs2 = np.array(fs2).T
fs2a = np.array(fs2a).T
fsa = np.array(fsa).T

plt.plot(xis,fs2[0])
plt.plot(xis,fs2a[0])
plt.plot(xis,fsa[0])
plt.show()






print(tab.interp(x,y)-f(x,y))
print(tab2.interp(x,y)-f(x,y))

print("S")
xs = np.linspace(-lim,lim,nx+3)
ys = np.linspace(-lim,lim,ny+3)

errs1 = []
errs2 = []
for i in range(0,nx):
    for j in range(0,ny):
        errs1.append(tab.interp(xs[i],ys[j])-f(xs[i],ys[j]))
        errs2.append(tab2.interp(xs[i],ys[j])-f(xs[i],ys[j]))
        
        
print(sum(abs(np.array(errs1)))/len(errs1))
print(sum(abs(np.array(errs2)))/len(errs2))




