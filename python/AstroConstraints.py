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