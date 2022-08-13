import asset as ast
import numpy as np

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments



def RendezvousConstraint(tab,Vars):
    n = len(Vars)
    args = Args(n+1)
    x = args.head(n)
    t = args[n]
    fun = oc.InterpFunction(tab,Vars).vf()
    return (fun.eval(t) - x)

def VinfMatchCon(tab):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]
    
    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    vInfPlus=(BodyV.eval(t0)-v0).norm()
    vInfMinus=(BodyV.eval(t1)-v1).norm()
    return (vInfPlus-vInfMinus)
    
def RendCon(tab):
    XT = Args(7)
    x = XT.head(6)
    t = XT[6]
    fun = oc.InterpFunction(tab,range(0,6)).vf()
    return fun.eval(t) - x

def PosCon(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab,range(0,3)).vf()
    return fun.eval(t) - x
def VinfFunc(tab):
    XT = Args(4)
    x = XT.head(3)
    t = XT[3]
    fun = oc.InterpFunction(tab,range(3,6)).vf()
    return fun.eval(t) - x

def FlybyAngleBound(tab,mubod,minrad):
    X = Args(8)
    v0 = X.head3()
    t0 = X[3]
    
    v1 = X.tail(4).head3()
    t1 = X.tail(4)[3]
    
    BodyV =  oc.InterpFunction(tab,range(3,6)).vf()
    v0dir =(BodyV.eval(t0)-v0).normalized()
    v1dir =(BodyV.eval(t1)-v1).normalized()
    
    vInf2  =(BodyV.eval(t0)-v0).squared_norm()
    
    delta      = vf.arccos(vf.dot(v0dir,v1dir))
    deltaMax   =  2*vf.arcsin(mubod/(mubod + minrad*vInf2)) 
    return   delta - deltaMax

def VinfFunction(tab):
    args = Args(4)
    v = args.head(3)
    t = args[3]
    fun = oc.InterpFunction(tab,[3,4,5]).vf()
    dV = (fun.eval(t) - v).norm()
    return (dV)
def VinfSquaredFunction(tab):
    args = Args(4)
    v = args.head(3)
    t = args[3]
    fun = oc.InterpFunction(tab,[3,4,5]).vf()
    dV = (fun.eval(t) - v).norm_squared()
    return (dV)
def CosAlpha():
    args = Args(6)
    rhat = args.head3().normalized()
    nhat = args.tail3().normalized()
    return vf.dot(rhat,nhat)