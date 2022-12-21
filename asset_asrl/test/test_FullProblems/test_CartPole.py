import numpy as np
import asset as ast
import unittest

vf = ast.VectorFunctions
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Args = vf.Arguments

class CartPole(oc.ode_x_u.ode):
    
    def __init__(self,l,m1,m2,g):
        
        args = oc.ODEArguments(4,1)
        
        q1  = args.XVar(0)
        q2  = args.XVar(1)
        q1d = args.XVar(2)
        q2d = args.XVar(3)
        
        q1,q2,q1d,q2d = args.XVec().tolist()
        
        u = args.UVar(0)
        
        q1dd = (l*m2*vf.sin(q2)*(q2d**2) + u + m2*g*vf.cos(q2)*vf.sin(q2))/( m1 + m2*((1-vf.cos(q2)**2)))
        q2dd = -1*(l*m2*vf.cos(q2)*vf.sin(q2)*(q2d**2) +u*vf.cos(q2) +(m1*g+m2*g)*vf.sin(q2))/( l*m1 + l*m2*((1-vf.cos(q2)**2)))
        
        ode = vf.stack([q1d,q2d,q1dd,q2dd])
        super().__init__(ode,4,1)
        
###############################################################################
class test_CartPole(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        
        self.FinalObj = 58.83219229674185 ## Sensitive to Segments
        self.MaxObjError = .1
        self.MaximumIters = 20
        
    def problem_impl(self,tmode,cmode,nsegs):
        m1 = 1
        m2 =.3
        l=.5
        g = 9.81
        
        umax = 20
        dmax = 2
        
        tf = 2
        d = 1
        
        ts = np.linspace(0,tf,100)
        IG = [[d*t/tf,np.pi*t/tf,0,0,t,.00] for t in ts]
        ode = CartPole(l,m1,m2,g)
        
        
        
        phase = ode.phase(tmode,IG,nsegs)
        phase.setControlMode(cmode)
        phase.addBoundaryValue("Front",range(0,5),[0,0,0,0,0])
        phase.addBoundaryValue("Back",range(0,5),[d,np.pi,0,0,tf])
        phase.addLUVarBound("Path",5,-umax,umax,1.0)
        phase.addLUVarBound("Path",0,-dmax,dmax,1.0)
        phase.addIntegralObjective(Args(1)[0]**2,[5])    
        phase.optimizer.PrintLevel= 3
        Flag = phase.optimize()
        
        Obj = phase.optimizer.LastObjVal
        ObjError = abs(Obj-self.FinalObj)
        
        self.assertLess(phase.optimizer.LastIterNum, self.MaximumIters,
                         "Optimizer iterations exceeded expected maximum")
        self.assertEqual(Flag,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        self.assertLess(ObjError, self.MaxObjError,
                         "Final objective significantly differs from known answer")
        
        
    
    def test_FullProblem(self):
        
        tmodes = ["LGL3","LGL5","LGL7","Trapezoidal"]
        nsegs  = [256   ,128   ,96   ,256]
        for tmode,nseg in zip(tmodes,nsegs):
            with self.subTest(TranscriptionMode=tmode):
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode,"HighestOrderSpline",nseg)
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode,"BlockConstant",nseg)


##############################################################################        
        
if __name__ == "__main__":
    
    unittest.main(exit=False)

            
    
   
   

    ###########################################################################
