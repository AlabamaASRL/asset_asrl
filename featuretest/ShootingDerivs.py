import asset_asrl as ast
import numpy as np
import time
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
astro = ast.Astro



class ODE1(oc.ODEBase):
    
    def __init__(self):
        
        Xt = oc.ODEArguments(6)
        
        R,V = Xt.XVec().tolist([(0,3),(3,3)])
        
        t = Xt.TVar()
        
        
        G = - R.normalized_power3()*(1 + .1*vf.sin(t))
        
        ode = vf.stack(V,G)
        
        super().__init__(ode,6)
        
        
        
def ShootingCons(ode,integ):
    
    irows = ode.XtUVars()*2 + ode.PVars()
    
    X1X2P = vf.Arguments(irows)
    
    XtU1 = X1X2P.head(ode.XtUVars())
    
    XtU2 = X1X2P.segment(ode.XtUVars(),ode.XtUVars())
    
    P    = X1X2P.tail(ode.PVars())
    
    tm = (XtU1[ode.TVar()] + XtU2[ode.TVar()])/2
    
    I1 = [XtU1,tm] if ode.PVars()==0 else [XtU1,P,tm]
    I2 = [XtU2,tm] if ode.PVars()==0 else [XtU2,P,tm]
    
    Ivf = integ.vf()
    
    defect = Ivf(I1).head(ode.XVars()) - Ivf(I2).head(ode.XVars())
    
    return defect


def Test1():
    
    
    ode = ODE1()
    integ = ode.integrator(.1)
    
    
    X1 = np.zeros((7))
    X1[0]=1
    X1[4]=1.001
    X1[5]=.001
    X1[6]=.03
    
    
    X2 = np.zeros((7))
    X2[1]=1
    X2[3]=-1
    X2[5]=.001
    X2[6]=1.57
    
    XS = np.zeros((14))
    XS[0:7]=X1
    XS[7:14]=X2
    
    
    
    Truth = ShootingCons(ode,integ)
    Test = ode.ode.shooting_defect(integ)
    
    LS = range(1,7)
    
    print(abs(Truth(XS)-Test(XS)).max())
    print(abs(Truth.jacobian(XS)-Test.jacobian(XS)).max())
    print(abs(Truth.adjointhessian(XS,LS)-Test.adjointhessian(XS,LS)).max())


    Test.SpeedTest(XS,100)
    
    
    
Test1()



    
    

    
    
    