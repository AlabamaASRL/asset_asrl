import numpy as np
import asset as ast
import matplotlib.pyplot as plt

norm = np.linalg.norm

S1 = np.array([-6,3,0])
S1m = 6

S2 = np.array([-4,-2,0])
S2m = 2

P = np.array([0,0,0])
Pm = 3


RCM = (S1*S1m + S2*S2m)/(S1m+S2m)
print(RCM)

F = -Pm*( S1*S1m/(norm(S1)**3) + S2*S2m/(norm(S2)**3))
FN = norm(F)
print(FN)
print(F/FN)

RCG = np.sqrt((Pm*(S1m + S2m))/FN)
print(RCG)
rcg = -(F/FN)*RCG

M = np.cross((RCM-rcg),F)

print(M)


plt.scatter(S1[0],S1[1],label='S1')
plt.scatter(S2[0],S2[1],label='S2')
plt.scatter(P[0],P[1],label='P')
plt.scatter(RCM[0],RCM[1],label='CM')
plt.scatter(rcg[0],rcg[1],label='CG')

plt.arrow(rcg[0], rcg[1],-.9*rcg[0], -.9*rcg[1], head_width=0.2, head_length=0.2, fc='k', ec='k',label="F")
plt.axis("Equal")
plt.xlabel("a1")
plt.ylabel("a2")
plt.legend()
plt.grid(True)
plt.show()
