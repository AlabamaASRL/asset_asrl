import numpy as np
import asset as ast
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from QuatPlot import AnimSlew
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

class QuatModel(oc.ode_x_x.ode):
    def __init__(self,I):
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = vf.Arguments(Ivars)
        qvec = args.head3()
        q4 = args[3]
        w = args.segment3(4)
        T = args.tail3()
        qvdot = (w*q4 + vf.cross(qvec,w))*0.5
        q4dot = -0.5*(vf.dot(w,qvec))
        wd1 = T[0]/I[0] + ((I[1]-I[2])/(I[0]))*(w[1].dot(w[2]))
        wd2 = T[1]/I[1] + ((I[2]-I[0])/(I[1]))*(w[0].dot(w[2]))
        wd3 = T[2]/I[2] + ((I[0]-I[1])/(I[2]))*(w[0].dot(w[1]))
        ode = vf.Stack([qvdot,q4dot,wd1,wd2,wd3])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)
        
def GetData(Traj):
    C33=[]
    C13=[]
    C23=[]
    Prec=[]
    Nut =[]
    Ts =[]
    for T in Traj:
        Ts.append(T[7])
        Rot = R.from_quat(T[0:4])
        DCM = Rot.as_matrix()
        ANGS = Rot.as_euler('ZXZ',degrees=True)
        Prec.append(ANGS[0])
        Nut.append(ANGS[1])
        C13.append(DCM[0,2])
        C23.append(DCM[1,2])
        C33.append(DCM[2,2])
        nut = np.arccos(DCM[2,2])
        prec = np.arcsin(C13[-1]/np.sin(nut))
    #print((Prec[-1] - np.rad2deg(prec)),np.rad2deg(prec))
    return Prec,Nut,C33,C13,C23

1
tf = 9*np.pi/2

###################Problem 1#######################
Ts = np.linspace(0,tf,1000)
W1A = np.cos(4*Ts/3) -(19/20)*np.sin(4*Ts/3)
W2A = -np.sin(4*Ts/3) -(19/20)*np.cos(4*Ts/3) -.05
plt.plot(Ts,W1A,color='red',label='w1')
plt.plot(Ts,W2A,color='blue',label='w2')
plt.plot(Ts,np.full_like(Ts,2.0),color='green',label='w3')
plt.xlabel("Time(s)")
plt.ylabel("rad/s ")
plt.title("Analytic Solution")
plt.legend()
plt.grid(True)
plt.show()

##################################################
Ivec = [9000,4000,4000]
ode = QuatModel(Ivec)
integ = ode.integrator(.01)
IG = np.zeros((11))
IG[3]=1.0
IG[4]=np.sqrt(2)
IG[5]=-np.sqrt(2)
IG[6]=0
IG[8]=20
Traj = integ.integrate_dense(IG,tf,5000)
Traj2 = integ.integrate_dense(IG,1.25,5000)
#AnimSlew(Traj2,Anim=False)
IG0 = np.copy(IG)
IG0[8]=0
Traj0 = integ.integrate_dense(IG0,tf,2000)
IG100 = np.copy(IG)
IG100[8]=100
Traj100 = integ.integrate_dense(IG100,tf,2000)
IG200 = np.copy(IG)
IG200[8]=200
Traj200 = integ.integrate_dense(IG200,tf,2000)

##################################################
TT = np.array(Traj).T
Ts = TT[7]
K = TT[0]**2 + TT[1]**2 + TT[2]**2 + TT[3]**2
Q1 = TT[0]
Q2 = TT[1]
Q3 = TT[2]
2
Q4 = TT[3]
plt.plot(TT[7],Q1,label='q1')
plt.plot(TT[7],Q2,label='q2')
plt.plot(TT[7],Q3,label='q3')
plt.plot(TT[7],Q4,label='q4')
plt.title("Integrated Quaternion")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(TT[7],K)
plt.xlabel("Time(s)")
plt.ylabel("Quaternion Norm")
plt.legend()
plt.grid(True)

plt.title("Integrated Quaternion Constraint")
plt.show()
W1 = TT[4]
W2 = TT[5]
W3 = TT[6]

plt.plot(TT[7],W1,color='red',label='w1')
plt.plot(TT[7],W2,color='blue',label='w2')
plt.plot(TT[7],W3,color='green',label='w3')
plt.xlabel("Time(s)")
plt.ylabel("rad/s ")
plt.title("Integrated Angular Rates")
plt.legend()
plt.grid(True)
plt.show()

Table = oc.LGLInterpTable(ode.vf(),7,3,Tmodes.LGL3, Traj,1000)
Tss =[0,2,4,6,8,12,14]
PT = [Table.Interpolate(t) for t in Tss]
Prec,Nut,C33,C13,C23 = GetData(PT)
for i in range(0,len(PT)):
    msg = 't = ' + "{:.3f}".format(int(PT[i][7])) +' s ;'
    msg += 'w1 =' + "{:.3f}".format(PT[i][4]) +' rad/s;'
    msg += 'w2 =' + "{:.3f}".format(PT[i][5]) +' rad/s;'
    msg += 'w3 =' + "{:.3f}".format(PT[i][6]) +' rad/s;'
    msg += 'C13 =' + "${:.3f}".format(C13[i]) +' ;'
    msg += 'C23 =' + "${:.3f}".format(C23[i]) +' ;'
    msg += 'C33 =' + "${:.3f}".format(C33[i]) +' ;'
    3
    msg += r'gamma =' + "{:.3f}".format(Prec[i]) +' deg ;'
    msg += r'eta =' + "{:.3f}".format(Nut[i]) +' deg;'
    msg += r'k =' + "{:.3f}".format(np.linalg.norm(PT[i][0:4])) +' ;'
    print(msg)
####################################################
Prec,Nut,C33,C13,C23 = GetData(Traj)
tl = 4.3
q=Table.Interpolate(tl)
print(R.from_quat(q[0:4]).as_euler('xyz')*180.0/np.pi)
PrecP,NutP,C33P,C13P,C23P = GetData([Table.Interpolate(t) for t in [tl]])
Rm = R.from_quat(Table.Interpolate(tl)[0:4])
b1 = Rm.apply([1,0,0])
b2 = Rm.apply([0,1,0])
print(PrecP,NutP)
plt.plot(C13,C23)
plt.scatter(C13P,C23P)
plt.plot([0,1],[0,0],color ='red',label='n1')
plt.plot([0,b1[0]],[0,b1[1]],color ='red',linestyle='--',label='b1')
plt.plot([0,0],[0,1],color ='green',label='n2')
plt.plot([0,b2[0]],[0,b2[1]],color ='green',linestyle='--',label='b2')
plt.plot([0,C13P[0]],[0,C23P[0]],color ='blue',linestyle='--',label='b3')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis("Equal")
plt.show()
####################################################
fig, axs = plt.subplots(2,1,sharex=True)
axs[0].plot(TT[7],Prec)
axs[0].grid(True)
axs[0].set_ylabel("Precession Angle (deg)")
axs[1].plot(TT[7],Nut)
axs[1].grid(True)
axs[1].set_ylabel("Nutation Angle (deg)")
axs[1].set_xlabel("Time(s)")
print(C13P[0]**2 + C23P[0]**2)
plt.show()
####################################################
labels = ["0 Nm",'100 Nm',' 200 nm']
Trajs =[Traj0,Traj100,Traj200]
fig, axs = plt.subplots(2,1)
axs[0].plot([0,1],[0,0],color ='red',label='n1')
axs[0].plot([0,0],[0,1],color ='green',label='n2')
axs[0].grid(True)
axs[1].grid(True)
4
axs[0].axis("Equal")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].set_xlabel("Time(s)")
axs[1].set_ylabel("Nutation Angle (deg)")
for i,Traj in enumerate(Trajs):
    Prec,Nut,C33,C13,C23 = GetData(Traj)
    axs[0].plot(C13,C23,label=labels[i])
    axs[1].plot(TT[7],Nut,label=labels[i])
axs[0].legend()
axs[1].legend()
plt.show()
5

