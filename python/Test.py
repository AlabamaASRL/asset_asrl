import numpy as np
import asset as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags





    
def SolarSail1(r,n,scale):
    ndr2 = vf.dot(r,n).squared()
    acc = scale*ndr2*r.inverse_four_norm()*n.normalized_power3()
    return acc

def SolarSail2(scale):
    args = Args(6)
    r = args.head_3()
    n = args.tail_3()
    return SolarSail1(r,n,scale)

def Model( mu, beta):
    args = Args(10)
    r = args.head_3() 
    v = args.segment_3(3)
    n = args.tail_3()
    acc = -mu*r.normalized_power3() + SolarSail1(r,n,beta*mu)
    acc = -mu*r.normalized_power3() + SolarSail2(beta*mu).eval(vf.Stack([r,n]))
    return vf.Stack([v,acc])
    

ode=oc.ode_x_x.ode(Model(1,.01),6,3)
oc.ode_x_x.phase(ode,Tmodes.LGL3)
ode.integrator(.01)



    
    
    
    
    
    
    
    
    
    
    
    
    