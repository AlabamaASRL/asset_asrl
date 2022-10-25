import numpy as np
import asset as ast

vf        = ast.VectorFunctions
Args      = vf.Arguments


R,V,U = Args(9).tolist([(0,3),(3,3),(6,3)])

print([1,1,1][7])

print(R[0:3](range(0,9)))



Rhat = R.normalized()
Nhat = R.cross(V).normalized()
That = Nhat.cross(Rhat).normalized()

M1 = vf.ColMatrix([Rhat,Nhat,That])
M2 = vf.RowMatrix([Rhat,Nhat,That])



#help(vf.ColMatrix)

#print(np.eye(3))

a = np.array([[1,0,0],
              [0,1.0,0],
              [0,0,1.0]])



print(V)

MK =  M1*M2


#(M1*M2)
Uout = (MK + MK)*U

#print(Uout(range(0,9)))



