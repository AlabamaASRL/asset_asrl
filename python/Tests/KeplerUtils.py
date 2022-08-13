import numpy as np
import asset as ast
import matplotlib.pyplot as plt
import random as rand
import time

vf    = ast.VectorFunctions
oc    = ast.OptimalControl
sol   = ast.Solvers
astro = ast.Astro


Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
Imodes = oc.IntegralModes
PhaseRegs = oc.PhaseRegionFlags

np.set_printoptions(precision=3, linewidth=120)


def classic_cartesian(OE,mu):
    
    X = astro.classic_to_cartesian(OE,mu)

    OEBack = astro.cartesian_to_classic(X,mu)
    
    Err = OE - OEBack
    Err[3:6] = np.arctan2(np.sin(Err[3:6]), np.cos(Err[3:6]))
    merr = max(Err)
    if(merr>1.0e-10 or True):
        print("Input Elements  :\n",OE)
        print("State Vector    :\n",X)
        print("Output Elements : \n",OEBack)
        print("Error:\n",Err)
        
    print("Max Error:\n",merr)
    return merr
    
def classic_cartesian_test(n = 10):
    Errors = []
    for k in range(0,n):
        print("#####################################")
        
        a = rand.uniform(-2,2)
        e = rand.uniform(0, 1)
        if(a<0):e+=1
        
        i = rand.uniform(0, np.pi/2)
        O = rand.uniform(0, 2*np.pi)
        w = rand.uniform(0, 2*np.pi)
        M = rand.uniform(0, 2*np.pi)
        
        OE1 = np.array([a,e,i,O,w,M])
        err = classic_cartesian(OE1,1)
        Errors.append(err)
    

def propagate_test(n = 10):
    
    mu = 1.0
    
    ode = astro.Kepler.ode(mu)
    integ = ode.integrator(.01)
    integ.Adaptive = True
    print("######################################")
    print("######## Proagator Test ##############")
    print("######################################")
    
    for k in range(0,n):
        
        a = rand.uniform(.5,6)
        e = rand.uniform(0, 1)
        if(a<0):e+=1
        
        i = rand.uniform(0, np.pi/2)
        O = rand.uniform(0, 2*np.pi)
        w = rand.uniform(0, 2*np.pi)
        M = rand.uniform(0, 2*np.pi)
        
        dt =  rand.uniform(0, 2*np.pi)
        
        OE = np.array([a,e,i,O,w,M])

        
        X0 = np.zeros((7))
        
        X0[0:6]=astro.classic_to_cartesian(OE,mu)
        
        tt = time.perf_counter()
        
        XFTrue = integ.integrate(X0,dt)
        XFCart = astro.propagate_cartesian(X0[0:6], dt,mu)
        OEF    = astro.propagate_classic(OE, dt,mu)
        XFOE   = astro.classic_to_cartesian(OEF,mu)
        
        tt1 = time.perf_counter()
        
        
        ErrCart = XFTrue[0:6]-XFCart
        ErrOE   = XFTrue[0:6]-XFOE
        
        print("#####################################")
        
        #print((tt1-tt)*1000)
        print("Input Elements  :\n",OE)
        print("State Vector    :\n",X0[0:6])
        print("Propagation Time:", dt)

        print("Cartesian Error \n",ErrCart)
        print("Classic   Error \n",ErrOE)
        
        
        
        
        

    

if __name__ == "__main__":
    print("Kepler Utils Test")
    
    classic_cartesian_test(n = 10)
    propagate_test(n = 10)
    
    
            
            
    
            
            
    
    
    
    
