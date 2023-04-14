import asset_asrl as ast
import numpy as np
import time
import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
astro = ast.Astro





class CR3BP(oc.ODEBase):
    def __init__(self,mu, cpp = False):
        
            self.mu = mu
            self.P1 = np.array([-self.mu,0,0])
            self.P2 = np.array([1-self.mu,0,0])
            ##########################################  
            Xt =  oc.ODEArguments(6)
            r,v = Xt.tolist([(0,3),(3,3)])
            
            x    = r[0]
            y    = r[1]
            xdot = v[0]
            ydot = v[1]
            
           
            q = vf.stack([2*ydot+x,-2*xdot+y]).padded_lower(1)
            g1 = (r-self.P1).normalized_power3()*(self.mu-1.0)
            g2 = (r-self.P2).normalized_power3()*(-self.mu)
    
            rdot = v
            vdot = vf.sum(g1,g2,q)
            
            ode = vf.stack(rdot,vdot)
            if(cpp):ode = astro.cr3bp(mu)
            super().__init__(ode,6)
            ##########################################
            

def TimeIt(msg,func,*args):
    t0 = time.perf_counter()
    Val = func(*args)
    tf = time.perf_counter()
    
    print(msg,f": {1000*(tf-t0)} ms")
    return Val
    


def Perf2():
    np.set_printoptions(linewidth=180)
    
    ode = ast.Astro.Kepler.ode(1.0)
    
    integ = ode.integrator(.1)
    integ.EnableVectorization = True
    
    kp  = ast.Astro.Kepler.KeplerPropagator(1.0).vf()
    
    DT = 3
    Istate = [1,
              0.,
              0,
              0,
              1.3,
              .01,
              0]
    
    Kin = np.copy(Istate)
    Kin[6] = DT
    
    
    nevals = 8
    
    L = np.ones((7))
    L[6]=0
    
    Istates = [Istate]*nevals
    Lstates = [L]*nevals
    
    DTs = [1.0*DT]*nevals
    DTs = np.linspace(1.5*DT,0.5*DT,nevals)
    
    D1  =TimeIt("CPP   -No Vectorization",integ.integrate_stm_v,Istates,DTs,False)
    D2  =TimeIt("CPP  -   Vectorization",integ.integrate_stm_v,Istates,DTs,True)
    
    JH1  =TimeIt("CPP   -No Vectorization",integ.integrate_stm2_v,Istates,DTs,Lstates,False)
    JH2  =TimeIt("CPP  -   Vectorization",integ.integrate_stm2_v,Istates,DTs ,Lstates,True)
    
    
    for i in range(0,len(D1)):
       print("################################################################")
       Kin[6] = DTs[i]

       JK = kp.jacobian(Kin)[0:6,0:6]
       HK = kp.adjointhessian(Kin,L[0:6])[0:6,0:6]



       print((D1[i][1][0:6,0:6] - JK)/JK)
       
       print("-----------------------------------------------------------------")

       print((D2[i][1][0:6,0:6] - JK)/JK)
       
       print("-----------------------------------------------------------------")

       print((JH1[i][1][0:6,0:6] - JK)/JK)
       
       print("-----------------------------------------------------------------")

       print((JH1[i][2][0:6,0:6] - HK)/HK)
       
       print("-----------------------------------------------------------------")

       print((JH2[i][2][0:6,0:6] - HK)/HK)




            
def Perf():
    
    np.set_printoptions(linewidth=180)
    odePy = CR3BP(.01,False)
    odeCPP= CR3BP(.01,True)
    
    integPy = odePy.integrator("DP87",0.01)
    integPy.setAbsTol(1.0e-12)
    integPy.EnableVectorization = True

    
    integCPP = odeCPP.integrator("DP87",.01)
    integCPP.setAbsTol(1.0e-12)
    integCPP.EnableVectorization = True
    

    Istate = [0.80, 0.0, 0.01, 0.0, 0.6276410653920693-.8, 0.,0]
    DT = 0.5
    
    
    nevals = 300
    
    L = np.ones((7))
    L[6]=0
    
    Istates = [Istate]*nevals
    DTs     = [1.0*DT]*nevals
    
    Lstates = [L]*nevals
    
    #DTs = [1.5*DT,1.1*DT,1.7*DT,.8*DT]*2
    #DTs = np.linspace(1.5*DT,0.5*DT,nevals)
    
    
    Traj = integPy.integrate_dense(Istate,DT)
    
    TT = np.array(Traj).T

    plt.plot(TT[0],TT[1],color='k')
    plt.xlabel("X(ND)")
    plt.ylabel("Y(ND)")
    plt.grid(True)
    plt.axis("Equal")
    plt.scatter(-.01,0,label='P1')
    plt.scatter( .99, 0,label='P2',zorder=100,s=3)
    
    plt.legend()
    
    
    plt.show()
    
    t0 = time.perf_counter()
    for i in range(0,30):
        print("################################################################")


        TimeIt("Python-No Vectorization",integPy.integrate_v,Istates,DTs,False)
        TimeIt("Python-   Vectorization",integPy.integrate_v,Istates,DTs,True)
        TimeIt("CPP   -No Vectorization",integCPP.integrate_v,Istates,DTs,False)
        TimeIt("CPP   -   Vectorization",integCPP.integrate_v,Istates,DTs,True)
        print("-----------------------------------------------------------------")

        NV1 =TimeIt("Python-No Vectorization",integPy.integrate_stm_v,Istates,DTs,False)
        V1 =TimeIt("Python-   Vectorization",integPy.integrate_stm_v,Istates,DTs,True)
        
        NV2 =TimeIt("CPP   -No Vectorization",integCPP.integrate_stm_v,Istates,DTs,False)
        V2  =TimeIt("CPP   -   Vectorization",integCPP.integrate_stm_v,Istates,DTs,True)
        
        print("-----------------------------------------------------------------")

        JH1  =TimeIt("Python-No Vectorization",integPy.integrate_stm2_v,Istates,DTs,Lstates,False)
        JH2  =TimeIt("Python-   Vectorization",integPy.integrate_stm2_v,Istates,DTs ,Lstates,True)

        JH1  =TimeIt("CPP   -No Vectorization",integCPP.integrate_stm2_v,Istates,DTs,Lstates,False)
        JH2  =TimeIt("CPP  -   Vectorization",integCPP.integrate_stm2_v,Istates,DTs ,Lstates,True)
        
    tf = time.perf_counter()
    
    
    print((tf-t0))
    
    '''
    for i in range(0,len(V1)):
        print("################################################################")
        print((NV1[i][1]-V1[i][1])[0:6,0:8]/V1[i][1][0:6,0:8])
        print("-----------------------------------------------------------------")
        print((NV2[i][1]-V2[i][1])[0:6,0:8]/V2[i][1][0:6,0:8])
        #print(NV1[i][1])
    '''
    
    

    
    
if __name__ == "__main__":
    Perf()