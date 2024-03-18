import numpy as np
import asset_asrl as ast



from asset_asrl.VectorFunctions.Extensions.DerivChecker import FDDerivChecker



def f(x,y,z,w):
    return np.cos(x)*np.cos(y)*np.cos(z)*np.cos(w) 



np.set_printoptions(linewidth=120)
vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments


def f(x,y,z,w):return np.cos(x)*np.cos(y)*np.cos(z)*np.cos(w)


nx = 50
ny = 50
nz = 50
nw = 50

xlim = np.pi
ylim = np.pi
zlim = np.pi
wlim = np.pi

xs = np.linspace(-xlim,xlim,nx)
ys = np.linspace(-ylim,ylim,ny)
zs = np.linspace(-zlim,zlim,nz)
ws = np.linspace(-zlim,zlim,nw)


X,Y,Z,W = np.meshgrid(xs,ys,zs,ws,indexing = 'ij')
Fs    = f(X,Y,Z,W)    #Scalar data defined on 4-D meshgrid in ij format!!!

kind = 'cubic' # or 'linear', defaults to 'cubic'
cache = False # defaults to False
#cache = True # will precalculate and cache all interpolation coeffs

Tab4D = vf.InterpTable4D(xs,ys,zs,ws,Fs,kind=kind,cache=cache)

print(Tab4D(0,0,0,0))  #prints 1.0 

Tab4D.WarnOutOfBounds=True   # By default
print(Tab4D(-10,0,0,0))        # prints a warning
print(Tab4D(0,-10,0,0))        # prints a warning
print(Tab4D(0,0,-10,0))        # prints a warning
print(Tab4D(0,0,0,-10))        # prints a warning


xyzw,c= Args(5).tolist([(0,4),(4,1)])
x,y,z,w = xyzw.tolist()

# Use it as scalar function inside a statement
Tab4sf = Tab4D(xyzw)
Tab4sf = Tab4D(x,y,z,w)             # Or
Tab4sf = Tab4D(vf.stack([x,y,z,w])) # Or


Tab4D.ThrowOutOfBounds=True
#print(Tab4D(-10,0,0,0))       # throws exception

Func = Tab4sf + c

print(Func([0,0,0,0,1]))  # prints [2.0]
    

