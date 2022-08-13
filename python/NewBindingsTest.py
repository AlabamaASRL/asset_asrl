import numpy as np
import asset as ast


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags


from DerivChecker import FDDerivChecker


n=9

    
X = Args(n)

R = X.head3()
V = X.segment3(3)
U = X.segment3(6)

N = R.cross(V).normalized()
T = R.cross(V).cross(R).normalized()

M = vf.RowMatrix([R.normalized(),T,N])

V = M.inverse()*U.normalized()



Xin = np.zeros((n))
Xin[0:9] = [1,1,0,0,1,0,0,1,1]


(V).rpt(Xin,100000)

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

#FDDerivChecker(V,Xin)







X = Args(6)


print(vf.normalize(ast.Astro.Kepler.KeplerPropagator(1)))


vf.cross(np.array([0,8,0.0]),X.tail3().vf())
vf.cross(X.head(3),X.tail3())
vf.cross(X.head3(),X.tail(3))

vf.cross(X.head3().normalized(),X.tail3())
vf.cross(X.head(3).normalized(),X.tail3())
vf.cross(X.head3(),X.tail(3).normalized())
vf.cross(X.head3(),X.tail(3).normalized())
vf.cross(np.array([1,1,1.]),X.head3().vf())
vf.cross(np.array([1,1,1.]),X.head3())
vf.cross([1,1,1.],X.head3())
