import numpy as np
import asset as ast
import unittest

import matplotlib.pyplot as plt
import time

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

class LorenzODE(oc.ode_x.ode):
    def __init__(self,sigma,rho,beta):
        
        x,y,z = oc.ODEArguments(3).XVec().tolist()
        
        ode = vf.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        
        super().__init__(ode,3)
        
class CauchyEulerODE(oc.ode_x.ode):
    def __init__(self,a,b):
       
       self.a = a
       self.b = b
       
       args =  oc.ODEArguments(2)
       x    = args.XVar(0)
       xdot = args.XVar(1)
       t    = args.TVar() 
       
       xddot = -a*xdot/t - b*x/t**2
       
       ode = vf.stack([xdot,xddot])
       
       super().__init__(ode,2)
       
       
    def analytic(self,x0,xdot0,t0,tf,n=1000):
        
        r1,r2 = np.roots([1.0,self.a-1,self.b])
        
        ts = np.linspace(t0,tf,n)
        
        if(np.isreal(r1) and np.isreal(r2)):
            
            M = np.array([[t0**r1 , t0**r2],
                          [r1*t0**(r1-1) , r2*t0**(r2-1)]])
            
            c1,c2 = np.dot(np.linalg.inv(M),np.array([x0,xdot0]))
            
            xs    = c1*ts**r1 + c2*ts**r2
            xdots = r1*c1*ts**(r1-1) + c2*r2*ts**(r2-1)
                           
            
        else:
            alpha = np.real(r1)
            beta  = np.imag(r1)
            
            M = np.array([[(t0**alpha)*np.cos(beta*np.log(t0)) 
                           , (t0**alpha)*np.sin(beta*np.log(t0))],
                          
                          [(t0**(alpha-1))*( alpha*np.cos(beta*np.log(t0)) - beta*np.sin(beta*np.log(t0)) ) ,
                           (t0**(alpha-1))*( alpha*np.sin(beta*np.log(t0)) + beta*np.cos(beta*np.log(t0)) )]
                          ])
            
            c1,c2 = np.dot(np.linalg.inv(M),np.array([x0,xdot0]))
            
            xs = c1*(ts**alpha)*np.cos(beta*np.log(ts)) + c2*(ts**alpha)*np.sin(beta*np.log(ts))
            
            xdots = c1*(ts**(alpha-1))*( alpha*np.cos(beta*np.log(ts)) - beta*np.sin(beta*np.log(ts)) ) \
                  + c2*(ts**(alpha-1))*( alpha*np.sin(beta*np.log(ts)) + beta*np.cos(beta*np.log(ts)) )
            
            
        return xs,xdots,ts
    
class QuatModel(oc.ode_7_3.ode):
    def __init__(self,I):
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars
        
        self.Ivec = I
        ############################################################
        args = vf.Arguments(Ivars)
        
        q    = args.head(4)
        w    = args.segment3(4)
        T    = args.tail3()
        
        qdot  = vf.quatProduct(q,w.padded_lower(1))/2.0

        L     = w.cwiseProduct(I)
        wdot  = (L.cross(w) + T).cwiseQuotient(I)
        ode = vf.stack(qdot,wdot )        
        
        ##############################################################
        super().__init__(ode,Xvars,Uvars)
    def DetumbleLaw(self):
            w=Args(3).head3()
            Lhat = w.cwiseProduct(self.Ivec).normalized()
            return -Lhat
        
        
class test_Integrators(unittest.TestCase):
    
    def test_Lorenz(self):
        '''
        Tests: Adaptive Integration of Autonomous ODEs
        '''
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        
        abstol = 1.0e-13
        defstepsize = .001
        minstepsize = .000000001
        errtol = 1.0e-6
        n = 2000
        
        tf = 20
        X0 = np.array([1,1,1,0])
        ode = LorenzODE(sigma, rho, beta)
        
        integ = ode.integrator(defstepsize)
        integ.setAbsTol(abstol)
        integ.MinStepSize = minstepsize
        integ.Adaptive=True
        
        Traj =integ.integrate_dense(X0,tf,n)
        
        XF = np.array([13.79319963, 12.95180398, 34.90160871,20])
        
        
        Err = np.linalg.norm(XF-Traj[-1])
        
        self.assertLess(Err, errtol,
                         "Integration Error exceeds expected maximum")
        
        if __name__ == "__main__":
            Traj = np.array(Traj)
            fig = plt.figure()
            ax = plt.subplot(projection="3d")
            ax.plot(Traj[:, 0], Traj[:, 1], Traj[:, 2])
            plt.show()

        
    def test_CauchyEuler(self):
        '''
        Tests: Adaptive Integration of non-autonomous ODEs
        '''
        a = -.5
        b = 16
        
        x0 = 1
        xdot0 =.25
        t0 = .1
        tf = 10
        
        abstol = 1.0e-13
        
        defstepsize = .01
        minstepsize = .0000000001
        
        errtol = 1.0e-11
        
        ode = CauchyEulerODE(a,b)
        
        integ = ode.integrator("DOPRI54",defstepsize)
        integ.setAbsTol(abstol)
        integ.MinStepSize = minstepsize
        integ.Adaptive=True
        
        n = 100
        
        xs,xdots,ts = ode.analytic(x0, xdot0, t0, tf,n)
    

        Traj = integ.integrate_dense([x0,xdot0,t0],tf,n)
        T = np.array(Traj).T

        xerr = max(abs(xs-T[0]))
        xdoterr = max(abs(xdots-T[1]))
        
        maxerr = max(xerr,xdoterr)
        
        self.assertLess(maxerr, errtol,
                         "Integration Error exceeds expected maximum")
        
        if __name__ == "__main__":

            plt.plot(ts,abs(xs-T[0]))
            plt.plot(ts,abs(xdots-T[1]))
            plt.yscale("log")
            plt.show()
            
            plt.plot(T[2],T[0])
            plt.plot(T[2],T[1])
            
            plt.show()
        

    def test_Detumble(self):
        '''
        Tests: Adaptive Integration of state dependent control law, Quaternions, Cross Products
        '''
        
        QErrtol =1.0e-10
        WErrtol =1.0e-6
        
        Ivec = np.array([1.,2.,3.])
        W0   = np.array([3.,9.,3.])
        
        X0 = np.zeros((11))
        X0[3]=1
        X0[4:7]=W0
        
        ode = QuatModel(Ivec)
        integ = ode.integrator(.01,ode.DetumbleLaw(),range(4,7))
        integ.Adaptive = True
        integ.setAbsTol(1.0e-14)
        tf = np.linalg.norm(W0*Ivec)
        
        n  = 1000
        
        Traj = integ.integrate_dense(X0,tf)
        
        QF = np.array([-0.893804752502,0.125508984388,0.070813040593,0.424671723248])
        
        QErr = np.linalg.norm(Traj[-1][0:4]-QF)
        WErr = np.linalg.norm(Traj[-1][4:7])
        
        self.assertLess(QErr, QErrtol,
                         "Quaternion Integration Error exceeds expected maximum")
        self.assertLess(WErr, WErrtol,
                         "Angular Velocity Integration Error exceeds expected maximum")
        
        if __name__ == "__main__":
        
            T = np.array(Traj).T
            
            plt.plot(T[7],T[4])
            plt.plot(T[7],T[5])
            plt.plot(T[7],T[6])
            plt.show()
            
    def test_TwoBodySTM(self):
        '''
        Tests: Serial and parallel STM computation
        '''
        ode = ast.Astro.Kepler.ode(1)
        kprop = ast.Astro.Kepler.KeplerPropagator(1.0) # Ground Truth
        
        integ = ode.integrator(.01)
        integ.setAbsTol(1.0e-13)
        integ.Adaptive=True
        integ.FastAdaptiveSTM = False
        
        Xtol = 1.0e-10
        Jtol = 1.0e-9
        
        x0  = 1
        vy0 = 1.35
        vz0 = .1
        tf  = 10
        
        X0 = np.zeros((7))
        X0[0]=x0
        X0[4]=vy0
        X0[5]=vz0
        
        KX = np.copy(X0)
        KX[6]=tf
        
        fx,jx = kprop.vf().computeall(KX,np.ones((6)))[0:2]
        
        Xf,STM = integ.integrate_stm(X0,tf)
        
        Xerr = np.linalg.norm(Xf[0:6]-fx)
        Jerr = abs(STM[0:6,0:6]-jx[0:6,0:6]).max()
        
        with self.subTest('Serial'):
            self.assertLess(Xerr, Xtol,
                             "State Integration Error exceeds expected maximum")
            self.assertLess(Jerr, Jtol,
                             "STM Integration Error exceeds expected maximum")
            
        Xf,STM = integ.integrate_stm_parallel(X0,tf,8)
        
        Xerr = np.linalg.norm(Xf[0:6]-fx)
        Jerr = abs(STM[0:6,0:6]-jx[0:6,0:6]).max()
        
        with self.subTest('Parallel'):
            self.assertLess(Xerr, Xtol,
                             "State Integration Error exceeds expected maximum")
            self.assertLess(Jerr, Jtol,
                             "STM Integration Error exceeds expected maximum")
            
    def test_EventDetection(self):
        
        r  = 1.0
        v  = 1.1
        t0 = 0.0
        tf = 200.0
        
        
        X0t0 = np.zeros((7))
        X0t0[0]=r
        X0t0[4]=v
        X0t0[6]=t0
        
        def ApseFunc():
            R,V = Args(7).tolist([(0,3),(3,3)])
            return R.dot(V)
        
        direction = -1
        stopcode = False
        ApoApseEvent  = (ApseFunc(),direction,stopcode)
        
        direction = 1
        stopcode = False
        PeriApseEvent  = (ApseFunc(),direction,stopcode)
        
        direction = 0
        stopcode  = 10  
        AllApseEvent  = (ApseFunc(),direction,stopcode)
        
        
        Events = [ApoApseEvent,PeriApseEvent,AllApseEvent]
        
        
        
        ode = ast.Astro.Kepler.ode(1)
        
        integ = ode.integrator(.01)
        integ.setAbsTol(1.0e-13)
        integ.Adaptive=True
        integ.FastAdaptiveSTM = False
        
        
        integ.EventTol =1.0e-10
        integ.MaxEventIters =12
        
        Xf, EventLocs1 = integ.integrate(X0t0,tf,Events)
        Xf, EventLocs2 = integ.integrate(X0t0,tf,Events)

        self.assertTrue(len(EventLocs1)==len(EventLocs2))
        
        self.assertTrue(len(EventLocs1[0])==5)
        self.assertTrue(len(EventLocs1[1])==5)
        self.assertTrue(len(EventLocs1[2])==10)
        
        
        afunc = ApseFunc()

        for i in range(0,len(EventLocs1)):
            self.assertTrue(len(EventLocs1[i])==len(EventLocs2[i]))

            for j in range(0,len(EventLocs1[i])):
                Xerr = np.linalg.norm(EventLocs1[i][j][0:6]-EventLocs2[i][j][0:6])
                Fxerr = abs(afunc(EventLocs1[i][j])[0])
                
                self.assertLess(Xerr, 1.0e-10,
                                 "Forward time and backward time event states are different")
                
                self.assertLess(Fxerr, integ.EventTol,
                                 "Event root error exceeds tolerance")
                
                
        
        
    def test_BatchCalls1(self):
        a = -.5
        b = 16
        
        x0 = 1
        xdot0 =.25
        t0 = .1
        tf = 10
        
        abstol = 1.0e-13
        
        defstepsize = .01
        minstepsize = .0000000001
        
        errtol = 1.0e-11
        
        ode = CauchyEulerODE(a,b)
        
        integ = ode.integrator("DOPRI87",defstepsize)
        integ.setAbsTol(abstol)
        integ.setStepSizes(defstepsize,minstepsize,10)
        integ.VectorizeBatchCalls = True
        
        batchsizes = [1,3,4,15,100,1003]
        X0 = [x0,xdot0,t0]
        
        for batchsize in batchsizes:
            
            tfs = np.linspace(tf,tf*2,batchsize)
            
            X0s = [X0]*batchsize
            
            Xfs = integ.integrate(X0s,tfs)
            
            for tfi,Xf in zip(tfs,Xfs):
                
                
                xs,xdots,ts = ode.analytic(x0, xdot0, t0, tfi,5)

                xerr = abs(xs[-1]-Xf[0])
                xdoterr = abs(xdots[-1]-Xf[1])
                
                maxerr = max(xerr,xdoterr)
                
                self.assertLess(maxerr, errtol,
                                 "Integration Error exceeds expected maximum")
            
            
    def test_BatchCalls2(self):
            
            kprop = ast.Astro.Kepler.KeplerPropagator(1.0) # Ground Truth
            ode1 = ast.Astro.Kepler.ode(1)
            ode2 = ast.OptimalControl.ode_6.ode(ode1.vf(),6)
            ode3 = ast.OptimalControl.ode_x.ode(ode1.vf(),6)
            ode4 = ast.OptimalControl.ode_x_u.ode(ode1.vf(),6,0)
            ode5 = ast.OptimalControl.ode_x_u_p.ode(ode1.vf(),6,0,0)
            
            for ode in [ode1,ode2,ode3,ode4,ode5]:

                integ = ode.integrator(.001)
                integ.setAbsTol(1.0e-13)
                integ.VectorizeBatchCalls = True
    
                Xtol = 1.0e-11
                Jtol = 1.0e-10
                Htol = 1.0e-9
                
                
                x0  = 1
                vy0 = 1.35
                vz0 = .1
                tf  = 10
                
                X0 = np.zeros((7))
                X0[0]=x0
                X0[4]=vy0
                X0[5]=vz0
                
                batchsizes = [1,3,4,5,15,42,69,103]
    
                for batchsize in batchsizes:
                    
                    tfs = np.linspace(tf*.5,tf*1.5,batchsize)
                    
                    X0s = [X0]*batchsize
                    
                    LF = np.ones((7))
                    LF[6]=0
                    LFs = [LF]*batchsize
                    
                    
                    Result1 = integ.integrate(X0s,tfs)
                    Result2 = integ.integrate_stm(X0s,tfs)
                    Result3 = integ.integrate_stm2(X0s,tfs,LFs)
    
                    
                    for tfi,Res1,Res2,Res3 in zip(tfs,Result1,Result2,Result3):
                        
                        Xf1 = Res1
                        Xf2,J2 = Res2
                        Xf3,J3,H3 = Res3
                        
                        
                        
                        KX = np.copy(X0)
                        KX[6]=tfi
                        fx,jx,gx,hx = kprop.vf().computeall(KX,np.ones((6)))
                        
                        for Xf in [Xf1,Xf2,Xf3]:
                            
                            Xerr = np.linalg.norm(Xf[0:6]-fx)
                            
                            
                            self.assertLess(Xerr, Xtol,
                                             "State Integration Error exceeds expected maximum")
                        for J in [J2,J3]:
                        
                            Jerr = abs((J[0:6,0:6]-jx[0:6,0:6])).max()
                            
                            self.assertLess(Jerr, Jtol,
                                             "STM Integration Error exceeds expected maximum")
                            
                        Herr = abs((H3[0:6,0:6]-hx[0:6,0:6])/hx[0:6,0:6]).max()
                        
                        self.assertLess(Herr, Htol,
                                         "Hessian Integration Error exceeds expected maximum")
                    

            
            
        
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    
    
    
    unittest.main(exit=False)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    