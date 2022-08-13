import numpy as np
import asset as ast

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments



r,v,u = Args(9).tolist([(0,3),(3,3),(6,3)])

nvec = r.cross(v)
tvec = nvec.cross(r)

RTNTran = vf.ColMatrix([r.normalized(),
                        tvec.normalized(),
                        nvec.normalized()])
Uinert = RTNTran*u



print(r(range(0,9)))
print(v(range(0,9)))
print(u(range(0,9)))



F1 = Args(2).normalized()


X,Y,Z = Args(3).tolist() 

F2 = F1(X,3)



print(F2([1,2,3]))