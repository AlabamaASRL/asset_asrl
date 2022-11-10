import numpy as np
import asset_asrl as ast
import unittest
import asset as astt
import matplotlib.pyplot as plt
import time

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags



np.set_printoptions(precision  = 3, linewidth = 200)



class TwoBody(oc.ODEBase):
    def __init__(self):
        Xt = oc.ODEArguments(6)
        R = Xt.head3()
        V =Xt.segment3(3)
        G = -R.normalized_power3()
        ode = vf.stack([V,G])
        super().__init__(ode,6)
        
class TwoBodyLTRTN(oc.ODEBase):
    def __init__(self,acc=.01,RTN = True):
        Xt = oc.ODEArguments(6,3)
        R = Xt.head3()
        V =Xt.segment3(3)
        U =Xt.tail3()
        
        Rhat = R.normalized()
        Nhat = R.cross(V).normalized()
        That = Nhat.cross(R).normalized()
        
        M = vf.ColMatrix([Rhat,That,Nhat])
        
        if(RTN):
            U = M*U
            
        
        
        G = -R.normalized_power3()
        Acc = U*acc
        ode = vf.stack([V,G+Acc])
        super().__init__(ode,6,3)
        
        

def Test_integrate(X0,tf, OldInteg,NewInteg):
    

    
    FStateOld = OldInteg.integrate(X0,tf)
    FStateNew = NewInteg.integrate(X0,tf)
    
    print(" integrate:", abs(FStateOld-FStateNew).max())
    print(" final state:",FStateNew)
    
    ntrajs = 100
    X0s = [X0]*ntrajs
    tfs = [tf]*ntrajs
    
    FStatesOld = OldInteg.integrate_parallel(X0s,tfs,6)
    FStatesNew = NewInteg.integrate_parallel(X0s,tfs,6)
    
    Errs = []
    
    for i in range(0,ntrajs):
        FStateOld = FStatesOld[i]
        FStateNew = FStatesNew[i]
        
        Err = abs(FStateOld-FStateNew).max()
        Errs.append(Err)
        
    
    print(" integrate_parallel:", max(Errs))
    
def Test_integrate_dense(X0,tf, OldInteg,NewInteg):
    
    
    TrajOld = OldInteg.integrate_dense(X0,tf)
    TrajNew = NewInteg.integrate_dense(X0,tf)
    
    FStateOld = TrajOld[-1]
    FStateNew = TrajNew[-1]
     
    print(" integrate_dense:", abs(FStateOld-FStateNew).max())
    
    
    TrajOld = OldInteg.integrate_dense(X0,tf,100)
    TrajNew = NewInteg.integrate_dense(X0,tf,100)
    
    m = int(len(TrajOld)/2)
    FStateOld = TrajOld[-1]
    FStateNew = TrajNew[-1]
     
    print(" integrate_dense(n)[-1]:", abs(TrajOld[-1]-TrajNew[-1]).max())
    

    print(" integrate_dense(n)[0]:", abs(TrajOld[0]-TrajNew[0]).max())
    print(" integrate_dense(n)[m]:", abs(TrajOld[m]-TrajNew[m]).max())

    
    ntrajs = 100
    X0s = [X0]*ntrajs
    tfs = [tf]*ntrajs
    
    t1 = time.perf_counter()
    TrajsOld = OldInteg.integrate_dense_parallel(X0s,tfs,6)
    t2 = time.perf_counter()
    TrajsNew = NewInteg.integrate_dense_parallel(X0s,tfs,6)
    t3 = time.perf_counter()
    Errs = []
    
    for i in range(0,ntrajs):
        FStateOld = TrajsOld[i][-1]
        FStateNew = TrajsNew[i][-1]
        
        Err = abs(FStateOld-FStateNew).max()
        
        Errs.append(Err)
        
    
    print(" integrate_dense_parallel:", max(Errs),(t2-t1)/(t3-t2))    
    

def Test_derivs(X0,tf,OldInteg,NewInteg):

    XIN = np.zeros((len(X0)+1))
    XIN[0:len(X0)]=X0
    XIN[len(X0)]=tf
    
    L = range(1,len(X0)+1)
    t1 = time.perf_counter()
    fxOld,jxOld,gxOld,hxOld = OldInteg.vf().computeall(XIN,L)
    t2 = time.perf_counter()
    fxNew,jxNew,gxNew,hxNew = NewInteg.vf().computeall(XIN,L)
    t3 = time.perf_counter()
    
    print(" fx:",abs(fxOld-fxNew).max())
    print(" jx:",abs(jxOld-jxNew).max())
    print(" gx:",abs(gxOld-gxNew).max())
    print(" hx:",abs(hxOld-hxNew).max(),(t2-t1)/(t3-t2))
    
    
   
    
    
def TestTwoBody(X0,tf,atol = 1.0e-12,meth = "DOPRI87"):
    
    print("X0:",X0," tf:",tf)

    
    ode = TwoBody()
    ode_new = oc.ode_x.ode(ode.vf(),6)
    ode_new2 = oc.ode_x.ode(ode.vf(),6)

    
    
    dstep = .1
    
    OldInteg = ode_new.integrator(dstep)
    OldInteg.Adaptive=True
    OldInteg.setAbsTol(atol)
    
    NewInteg = astt.OptimalControl.TestIntegratorX(ode_new,meth,dstep)
    NewInteg.setAbsTol(atol)
    
    Test_integrate(X0,tf, OldInteg,NewInteg)
    Test_integrate_dense(X0,tf, OldInteg,NewInteg)
    Test_derivs(X0,tf, OldInteg,NewInteg)
    
def TestTwoBodyLTRTN(X0,tf,atol = 1.0e-13,meth = "DOPRI87"):
    
    print("X0:",X0," tf:",tf)

    
    ode = TwoBodyLTRTN(.01,True)
    ode_new = oc.ode_x_u.ode(ode.vf(),6,3)
    
    
    meth = "DOPRI87"
    dstep = .9
    
    OldInteg = ode.integrator(dstep)
    OldInteg.Adaptive=True
    OldInteg.setAbsTol(atol)
    
    NewInteg = astt.OptimalControl.TestIntegratorXU(ode_new,meth,dstep)
    NewInteg.setAbsTol(atol)
    
    Test_integrate(X0,tf, OldInteg,NewInteg)
    Test_integrate_dense(X0,tf, OldInteg,NewInteg)
    Test_derivs(X0,tf, OldInteg,NewInteg)  
    
def TestTwoBodyLTRTNP(X0,tf,atol = 1.0e-13,meth = "DOPRI87"):
    
    print("X0:",X0," tf:",tf)

    
    ode = TwoBodyLTRTN(.01,True)
    ode_new = oc.ode_x_u_p.ode(ode.vf(),6,0,3)
    
    
    meth = "DOPRI87"
    dstep = .001
    
    OldInteg = ode_new.integrator(dstep)
    OldInteg.Adaptive=True
    OldInteg.setAbsTol(atol)
    
    NewInteg = astt.OptimalControl.TestIntegratorXUP(ode_new,meth,dstep)
    NewInteg.setAbsTol(atol)
    
    Test_integrate(X0,tf, OldInteg,NewInteg)
    Test_integrate_dense(X0,tf, OldInteg,NewInteg)
    Test_derivs(X0,tf, OldInteg,NewInteg)    
    
    
def TestTwoBodyLT(X0,tf,atol = 1.0e-14,meth = "DOPRI87"):
    
    print("X0:",X0," tf:",tf)

    
    ode = TwoBodyLTRTN(.01,False)
    ode_new = oc.ode_x_u.ode(ode.vf(),6,3)
    
    
    meth = "DOPRI87"
    dstep = .001
    
    OldInteg = ode.integrator(dstep,Args(3).normalized(),[3,4,5])
    OldInteg.Adaptive=True
    OldInteg.setAbsTol(atol)
    
    NewInteg = astt.OptimalControl.TestIntegratorXU(ode_new,meth,dstep,Args(3).normalized(),[3,4,5])
    NewInteg.setAbsTol(atol)
    
    Test_integrate(X0,tf, OldInteg,NewInteg)
    Test_integrate_dense(X0,tf, OldInteg,NewInteg)
    Test_derivs(X0,tf, OldInteg,NewInteg)
    
    
if __name__ == "__main__":
    
    V0 = 1.1
    
    eps=(V0**2)/2-1
    
    a = -.5/eps
    P = 2*np.pi*a**1.5
    print(a,P)
    
    tf = P*2
    tf=1.5
    
    
    X0tb = np.zeros((7))
    X0tb[0]=1
    X0tb[4]=V0
    
    TestTwoBody(X0tb,tf)
    
    X0rtn = np.zeros((10))
    X0rtn[0]=1
    X0rtn[4]=V0
    X0rtn[8]=1
    
    #TestTwoBodyLTRTN(X0rtn,tf)
    #TestTwoBodyLTRTNP(X0rtn,tf)
    #TestTwoBodyLT(X0rtn,tf)

    
    
    
    
    

        
        

        
            
            

