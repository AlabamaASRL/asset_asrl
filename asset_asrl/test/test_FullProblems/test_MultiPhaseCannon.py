import numpy as np
import asset as ast
import unittest


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments
Tmodes    = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
Cmodes    = oc.ControlModes


##########################################

g0      =  9.81 
Lstar   =  1000           ## m
Tstar   =  60.0           ## sec
Mstar   =  10             ## kgs
Astar   =  Lstar/Tstar**2
Vstar   =  Lstar/Tstar
Rhostar =  Mstar/Lstar**3
Estar   =  Mstar*(Vstar**2)




CD      = .5
RhoAir  = 1.225     /Rhostar
RhoIron = 7870      /Rhostar
h_scale = 8.44e3    /Lstar
E0      = 400000    /Estar
g       = g0/Astar

###########################################


def MFunc(rad,RhoIron):return (4/3)*(np.pi*RhoIron)*(rad**3)
def SFunc(rad):  return np.pi*(rad**2)
    
    
##############################################################################

class Cannon(oc.ode_x_u_p.ode):
    def __init__(self, CD,RhoAir,RhoIron,h_scale,g):
        ############################################################
        args  = oc.ODEArguments(4,0,1)
        
        v     = args.XVar(0)
        gamma = args.XVar(1)
        h     = args.XVar(2)
        r     = args.XVar(3)
        
        rad = args.PVar(0)
        
        S    = SFunc(rad)
        M    = MFunc(rad,RhoIron)
        
        rho     = RhoAir * vf.exp(-h / h_scale)
        
        D       = (0.5*CD)*rho*(v**2)*S
        
        vdot     = -D/M - g*vf.sin(gamma)
        gammadot = -g*vf.cos(gamma)/v
        hdot     = v*vf.sin(gamma)
        rdot     = v*vf.cos(gamma)
        
        ode = vf.stack([vdot,gammadot,hdot,rdot])
        ##############################################################
        super().__init__(ode,4,0,1)
        
        
def EFunc():
    v,rad =  Args(2).tolist()
    M = MFunc(rad,RhoIron)
    E = 0.5*M*(v**2)
    return E - E0
##############################################################################        


class test_MultiPhaseCannon(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        self.FinalObj = -3280.2039356471037
        self.MaxObjError= 1.0
        self.MaximumIters = 25
        self.NumSegments = 64
        
    def problem_impl(self,tmode,nsegs):
        
        rad0   = .1 /Lstar
        h0     = 100 /Lstar
        r0     = 0
        m0     = MFunc(rad0,RhoIron)
        gamma0 = np.deg2rad(45)
        v0     = np.sqrt(2*E0/m0)*.99
        
        
        
        ode = Cannon(CD,RhoAir,RhoIron,h_scale,g)
        integ = ode.integrator(.01)
        integ.Adaptive = True
        
        
        IG = np.zeros((6))
        IG[0] = v0
        IG[1] =gamma0
        IG[2] = h0
        IG[3] = r0
        IG[5] = rad0
        
        
        AscentIG = integ.integrate_dense(IG,
                                         60/Tstar,
                                         1000,
                                         lambda x:x[0]*np.sin(x[1])<0)
        DescentIG = integ.integrate_dense(AscentIG[-1],
                                          AscentIG[-1][4]+ 30/Tstar,
                                          1000,
                                          lambda x:x[2]<0)
        
        ##########################################################################
        
        aphase = ode.phase(tmode,AscentIG, nsegs)
        aphase.addLowerVarBound("ODEParams",0,0.0,1)
        aphase.addLowerVarBound("Front",1,0.0,1.0)
        aphase.addBoundaryValue("Front",[2,3,4],[h0,r0,0])
        
        aphase.addInequalCon("Front",EFunc()*.01,[0],[0],[])
        aphase.addBoundaryValue("Back",[1],[0.0])
            
        dphase = ode.phase(tmode,DescentIG, nsegs)
        dphase.addBoundaryValue("Back",[2],[0.0])
        dphase.addValueObjective("Back",3,-1.0)
        
        ocp = oc.OptimalControlProblem()
        ocp.addPhase(aphase)
        ocp.addPhase(dphase)
        
        ocp.addForwardLinkEqualCon(aphase,dphase,[0,1,2,3,4])
        ocp.addDirectLinkEqualCon(aphase,"ODEParams",[0],
                                  dphase,"ODEParams",[0])
        
        ocp.optimizer.set_OptLSMode("L1")
        ocp.optimizer.MaxLSIters = 2
        ocp.optimizer.PrintLevel = 3
        ocp.optimizer.QPThreads = 1
        ocp.Threads = 1
        
        Flag = ocp.optimize()
        
        Ascent  = aphase.returnTraj()
        Descent = dphase.returnTraj()
        
        
        Obj = ocp.optimizer.LastObjVal*Lstar
        ObjError = abs(Obj-self.FinalObj)
        
        self.assertEqual(Flag,ast.Solvers.ConvergenceFlags.CONVERGED, 
                         "Problem did not converge")
        
        self.assertLess(ocp.optimizer.LastIterNum, self.MaximumIters,
                         "Optimizer iterations exceeded expected maximum")
        
        self.assertLess(ObjError, self.MaxObjError,
                         "Final objective significantly differs from known answer")
        
    def test_FullProblem(self):
        
        tmodes = ["LGL3","LGL5","LGL7","Trapezoidal"]
        nsegs  = [16   ,16   ,16   ,128]

        for tmode,nseg in zip(tmodes,nsegs):
            with self.subTest(TranscriptionMode=tmode):
                self.problem_impl(tmode,nseg)
                
                    
        

##############################################################################        
if __name__ == "__main__":
    unittest.main(exit=False)


