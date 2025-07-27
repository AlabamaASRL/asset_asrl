import numpy as np
import matplotlib.pyplot as plt
import asset_asrl as ast
import asset_asrl.VectorFunctions as vf
import asset_asrl.OptimalControl as oc
from asset_asrl.VectorFunctions import Arguments as Args
import unittest



class ODE(oc.ODEBase):
    def __init__(self):
        
        XVars =1
        UVars =1
        
        args = oc.ODEArguments(XVars,UVars)
        x = args.XVar(0)
        u = args.UVar(0)
        xdot = .5*x + u
        super().__init__(xdot,XVars,UVars)

    class obj(vf.ScalarFunction):
        def __init__(self):
            x,u = Args(2).tolist()
            obj = u*u + x*u + 1.25*x**2
            super().__init__(obj)
            
###############################################################################
class test_ObjScaling(unittest.TestCase):
    '''
    Testing to see whether we get same solution with and without autoscaling
    '''
    @classmethod
    def setUpClass(self):
        
        self.FinalState = 0.3185865574270634
        self.MaxStateError = .001
        
        
        
        
    def problem_impl(self,tmode,cmode,mtol):
        
        ode = ODE()
        
       
        nsegs = 20
        
        
        iscale = np.pi
        vscale = np.e
        
        xstar = 2.11
        tstar = 3.3
        ustar = 5.1
        pstar = 8.2
        
        
        x0 = 1.0
        t0 = 0.0
        tf = 1.0
        u0 = .4
        
        TrajIG = [[x0,t,u0] for t in np.linspace(t0,tf,100)]
        
        
        #### Ground Truth, No AutoScaling #################
        phase = ode.phase(tmode,TrajIG,nsegs)
        phase.setAdaptiveMesh(True)
        phase.setControlMode(cmode)
        phase.setMeshTol(mtol)

        phase.addBoundaryValue("Front",[0,1],[x0,t0])
        phase.addBoundaryValue("Back", [1],  [tf])
        phase.addIntegralObjective(ODE.obj()*iscale,[0,2])
        phase.addValueObjective("Back",0,vscale)
        
        phase.setThreads(1,1)
        phase.optimizer.CNRMode =True
        phase.optimizer.set_QPOrderingMode("MINDEG")
        
        if __name__ != "__main__":
            phase.optimizer.PrintLevel = 3
            phase.PrintMeshInfo = False
            
        phase.optimize()
        
        xf1 = phase.returnTraj()[-1][0]
        ###################################################
        
        #### AutoScaling, Value + Integral ###############
        phase = ode.phase(tmode,TrajIG,nsegs)
        phase.setAdaptiveMesh(True)
        phase.setMeshTol(mtol)

        phase.setUnits([xstar,tstar,ustar])
        phase.addBoundaryValue("Front",[0,1],[x0,t0])
        phase.addBoundaryValue("Back", [1],  [tf])
        phase.addIntegralObjective(ODE.obj()*iscale,[0,2])
        phase.addValueObjective("Back",0,vscale)

        phase.setAutoScaling(True)
        phase.setThreads(1,1)
        phase.optimizer.CNRMode =True
        phase.optimizer.set_QPOrderingMode("MINDEG")
        
        if __name__ != "__main__":
            phase.optimizer.PrintLevel = 3
            phase.PrintMeshInfo = False
            
        phase.optimize()
        xf2 = phase.returnTraj()[-1][0]
        ###################################################
        
        #### AutoScaling, Value + IntegralParam ###############
        phase = ode.phase(tmode,TrajIG,nsegs)
        phase.setAdaptiveMesh(True)
        phase.setMeshTol(mtol)

        phase.setUnits([xstar,tstar,ustar])
        phase.setStaticParams([0.0],[pstar])
        
        phase.addBoundaryValue("Front",[0,1],[x0,t0])
        phase.addBoundaryValue("Back", [1],  [tf])
        phase.addIntegralParamFunction(ODE.obj(),[0,2],0)
        phase.addValueObjective("Back",0,vscale)
        phase.addValueObjective("StaticParams",0,iscale)

        phase.setAutoScaling(True)
        
        phase.setThreads(1,1)
        phase.optimizer.CNRMode =True
        phase.optimizer.set_QPOrderingMode("MINDEG")
        
        if __name__ != "__main__":
            phase.optimizer.PrintLevel = 3
            phase.PrintMeshInfo = False
            
        
        phase.optimize()
        xf3 = phase.returnTraj()[-1][0]
        ###################################################
        
        #### AutoScaling, Value + IntegralParam style 2 ###############
        phase = ode.phase(tmode,TrajIG,nsegs)
        phase.setAdaptiveMesh(True)
        phase.setMeshTol(mtol)

        phase.setUnits([xstar,tstar,ustar])
        phase.setStaticParams([0.0],[pstar])
        
        phase.addBoundaryValue("Front",[0,1],[x0,t0])
        phase.addBoundaryValue("Back", [1],  [tf])
        phase.addIntegralParamFunction(ODE.obj(),[0,2],0)
        
        phase.addStateObjective("Back",Args(2).dot([vscale,iscale]),[0],[],[0])

        phase.setAutoScaling(True)
        
        phase.setThreads(1,1)
        phase.optimizer.CNRMode =True
        phase.optimizer.set_QPOrderingMode("MINDEG")
        
        if __name__ != "__main__":
            phase.optimizer.PrintLevel = 3
            phase.PrintMeshInfo = False
            
        
        phase.optimize()
        xf4 = phase.returnTraj()[-1][0]
        ###################################################
        
        
        #### AutoScaling, Multi Phase Value + Integral ###############
        phase1 = ode.phase(tmode,TrajIG[0:int(nsegs/2)],nsegs)
        phase1.setAdaptiveMesh(True)
        phase1.setMeshTol(mtol)

        phase1.setUnits([xstar,tstar,ustar])
        phase1.addBoundaryValue("Front",[0,1],[x0,t0])
        phase1.addIntegralObjective(ODE.obj()*iscale,[0,2])
        phase1.addDeltaTimeEqualCon(tf/2.0)
        
        phase2 = ode.phase(tmode,TrajIG[int(nsegs/2):],nsegs)
        phase2.setAdaptiveMesh(True)
        phase2.setMeshTol(mtol)

        phase2.setUnits([xstar*.99,tstar*.783,ustar*1.1])
        phase2.setAdaptiveMesh(True)
        phase2.setMeshTol(mtol)
        
        phase2.addIntegralObjective(ODE.obj()*iscale,[0,2])
        phase2.addBoundaryValue("Back", [1],  [tf])
        phase2.addValueObjective("Back",0,vscale)
        
        ocp = oc.OptimalControlProblem()
        
        ocp.addPhase(phase1)
        ocp.addPhase(phase2)
        
        ocp.addForwardLinkEqualCon(phase1,phase2,range(0,3))

        ocp.setAutoScaling(True,True)
        ocp.setThreads(1,1)
        ocp.optimizer.CNRMode =True
        ocp.optimizer.set_QPOrderingMode("MINDEG")
        
        if __name__ != "__main__":
            ocp.optimizer.PrintLevel = 3
            ocp.PrintMeshInfo = False
            
        ocp.optimize()
        xf5 = phase2.returnTraj()[-1][0]
        ###################################################
        
        
        
        
        
        xferr1 = abs(xf1-self.FinalState)
        self.assertLess(xferr1, self.MaxStateError,
                 "Final state significantly differs from known answer")
        
        xferr2 = abs(xf2-self.FinalState)
        self.assertLess(xferr2, self.MaxStateError,
                 "Final state significantly differs from known answer")
        
        xferr3 = abs(xf3-self.FinalState)
        self.assertLess(xferr3, self.MaxStateError,
                 "Final state significantly differs from known answer")
        
        xferr4 = abs(xf4-self.FinalState)
        self.assertLess(xferr4, self.MaxStateError,
                 "Final state significantly differs from known answer")
        
        xferr5 = abs(xf5-self.FinalState)
        self.assertLess(xferr5, self.MaxStateError,
                 "Final state significantly differs from known answer")
        
    
    def test_FullProblem(self):
        
        for tmode in ["LGL3","LGL5","LGL7","Trapezoidal"]:
            with self.subTest(TranscriptionMode=tmode):
                mtol = 1.0e-5 if tmode=="Trapezoidal" else 1.0e-7
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode,"HighestOrderSpline",mtol)
                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode,"BlockConstant",mtol)


if __name__ == "__main__":
    unittest.main(exit=False)