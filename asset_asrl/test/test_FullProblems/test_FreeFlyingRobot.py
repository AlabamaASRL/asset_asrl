import numpy as np
import asset as ast
import unittest

import matplotlib.pyplot as plt

vf = ast.VectorFunctions
oc = ast.OptimalControl
sol = ast.Solvers
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

class FreeFlyingRobotODE(oc.ode_x_u.ode):
    def __init__(self,alpha,beta):
        Xvars = 6
        Uvars = 4
        ############################################################
        args = oc.ODEArguments(6,4)
        theta = args[4]
        omega = args[5]
        u     =  args.UVec()
        xdot  = args.segment2(2)
        vscale = vf.SumElems([u[0],u[1],u[2],u[3]],
                             [1,     -1,   1 ,-1])
        
        vdot = vf.stack([vf.cos(theta),vf.sin(theta)])*vscale
        
        theta_dot=omega
       
        omega_dot= vf.SumElems([u[0],u[1],u[2],u[3]],
                               [alpha, -alpha, -beta ,beta])
        ode = vf.stack([xdot,vdot,theta_dot,omega_dot])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)

    class obj(vf.ScalarFunction):
        def __init__(self):
            u = Args(4)
            obj = u[0] + u[1] + u[2] + u[3]
            super().__init__(obj)

class test_FreeFlyingRobot(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.FinalObj = 7.9115
        self.MaxObjError= .01
        self.MaximumIters = 100
        
    def problem_impl(self,tmode,cmode,nsegs):
        ode = FreeFlyingRobotODE(.2,.2)

        t0 = 0
        tf = 12

        X0 =np.array([-10,-10,0,0,np.pi/2.0,0, 0])
        XF =np.array([0,0,0,0,0,0,tf])

        IG = []
        ts = np.linspace(0,tf,100)

        for t in ts:
            T = np.zeros((11))
            T[0:7] = X0[0:7] + ((t-t0)/(tf-t0))*( XF[0:7]- X0[0:7])
            T[7:11] = np.ones((4))*.50
            IG.append(T)


        phase = ode.phase(tmode,IG,nsegs)
        phase.setControlMode(cmode)
        phase.addBoundaryValue("Front",range(0,7),X0)
        phase.addBoundaryValue("Back" ,range(0,7),XF)
        phase.addLUVarBounds("Path"   ,range(7,11),0.0,1.0,1)
        phase.addIntegralObjective(Args(4).sum(),range(7,11))
        phase.optimizer.PrintLevel=3
        phase.optimizer.OptLSMode = sol.LineSearchModes.L1
        phase.optimizer.MaxLSIters =1
        phase.optimizer.set_tols(1.0e-9,1.0e-9,1.0e-9)
        Flag = phase.optimize()
        
        Obj = phase.optimizer.LastObjVal
        ObjError = abs(Obj-self.FinalObj)
        
        self.assertEqual(Flag,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        
        self.assertLess(phase.optimizer.LastIterNum, self.MaximumIters,
                         "Optimizer iterations exceeded expected maximum")
        
        self.assertLess(ObjError, self.MaxObjError,
                         "Final objective significantly differs from known answer")
    def test_FullProblem(self):
        
        tmodes = ["LGL3","LGL5","LGL7","Trapezoidal"]
        nsegs  = [256   ,256   ,256   ,256]
        for tmode,nseg in zip(tmodes,nsegs):
            with self.subTest(TranscriptionMode=tmode):
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode,"HighestOrderSpline",nseg)
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode,"BlockConstant",nseg)

if __name__ == "__main__":
    
    unittest.main(exit=False)

    
