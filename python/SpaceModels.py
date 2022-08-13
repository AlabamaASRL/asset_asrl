import numpy as np
import asset as ast
vf = ast.VectorFunctions
oc = ast.OptimalControl
s=oc.OptimalControlProblem()
#s.addPhase()
class IdealSail(ast.VectorFunctional):
    def __init__(self,r,n,scale):
        ndr = n.dot(r)
        thrust = scale*vf.square(ndr)*n.normalized_power3()
        super().__init__(thrust)
class LowThrust(ast.VectorFunctional):
    def __init__(self,n,scale):
        thrust =n*scale
        super().__init__(thrust)
        
class CR3BPacc(ast.VectorFunctional):
    def __init__(self,args,mu):
        r = args.head_3()
        x    = args[0]
        y    = args[1]
        xdot = args[3]
        ydot = args[4]

        t1     = vf.SumElems([ydot,x],[ 2.0,1.0])
        t2     = vf.SumElems([xdot,y],[-2.0,1.0])
    
        rterms = vf.StackScalar([t1,t2]).padded_lower(1)
        p1loc = np.array([-mu,0,0])
        p2loc = np.array([1.0-mu,0,0])
        
        g1 = r.normalized_power3(-p1loc,(mu-1.0))
        g2 = r.normalized_power3(-p2loc,(-mu))
        acc = vf.Sum([g1,g2,rterms])
        super().__init__(acc)
        
class CR3BP_LTN(oc.ode_6_3.ode):
    def __init__(self,mu,scale):
        args = vf.Arguments(10)
        r = args.head_3()
        v = args.segment_3(3)
        n = args.tail_3()
        thrust = LowThrust(n,scale)
        ode= vf.Stack([v,CR3BPacc(args,mu)+thrust])
        super().__init__(ode)

        
        
        






