import numpy as np
import asset as ast
from QuatPlot import AnimSlew
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify, implemented_function
import scipy as scipy
from sympy import fourier_series
import mpmath as mp


vf    = ast.VectorFunctions
oc    = ast.OptimalControl
Args  = vf.Arguments
Tmodes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags
'''
In this example we will demonstrate defining an ode model and numerically integrating
The model in question will be quaternion based attitude dynamics with controllable
torque values, we will then write a

'''

class QuatModel(oc.ode_x_x.ode):
    def __init__(self,I):
        Xvars = 7
        Uvars = 3
        Ivars = Xvars + 1 + Uvars
        ############################################################
        args = vf.Arguments(Ivars)
        
        qvec = args.head3()
        q4   = args[3]
        w    = args.segment3(4)
        T    = args.tail3()
    
        qvdot = (w*q4 + vf.cross(qvec,w))*0.5
        q4dot = -0.5*(vf.dot(w,qvec))
        wd1   = T[0]/I[0] + ((I[1]-I[2])/(I[0]))*(w[1].dot(w[2]))
        wd2   = T[1]/I[1] + ((I[2]-I[0])/(I[1]))*(w[0].dot(w[2]))
        wd3   = T[2]/I[2] + ((I[0]-I[1])/(I[2]))*(w[0].dot(w[1]))
        ode = vf.Stack([qvdot,q4dot,wd1,wd2,wd3])
        ##############################################################
        super().__init__(ode,Xvars,Uvars)
    class DetumblePCon(ast.VectorFunctional):
        def __init__(self,Pweight):
            w = Args(3)
            con = -w*Pweight
            super().__init__(con)
    class PCon2(ast.VectorFunctional):
        def __init__(self,Ivec,wvec,h):
            
            I1 = Ivec[0]
            I2 = Ivec[1]
            I3 = Ivec[2]
            
            wi1 = wvec[0]
            wi2 = wvec[1]
            wi3 = wvec[2]
            
            K = (I2 - I3)/I1
            
            
            C1 =wi2/(h)
            C2 =wi1/(h)
            args = Args(4)
            t = args[3]
            Eta = K*wi3*(t.sf()-(t*t)/(2*h))
            u1 = -(C1*vf.sin(Eta) + C2*vf.cos(Eta))*I1
            u2 = -(C1*vf.cos(Eta) - C2*vf.sin(Eta))*I1
            u3 = -(wi3/h)*(t*0 + [1.0])*I3
            
            mag = I1*(C1**2 + C2**2)**.5
            
            w1 = args[0]
            w2 = args[1]
            
            
            u12 = -mag*args.head2().normalized()
            u12 = -I1*args.head2()/( -1*t.sf() + np.array([h*1.0000001]) )
            
            u1  = -I1*args[0]/(-1*t.sf() + np.array([h*1.00000000001]) )
            u2  = -I2*args[1]/(-1*t.sf() + np.array([h*1.00000000001]) )
            u3  = -I3*args[2]/(-1*t.sf() + np.array([h*1.00000000001]) )
            
            
            u = vf.Stack([u1,u2,u3])
            #u = vf.Stack([u12,u3])
            super().__init__(u)
    class PCon3(ast.VectorFunctional):
        def __init__(self,Ivec,tmax):
            
            I1 = Ivec[0]
            I2 = Ivec[1]
            I3 = Ivec[2]
            
            args = Args(4)
            
            
            u1  = I1*args[0]
            u2  = I2*args[1]
            u3  = I3*args[2]
            
            
            u = -vf.Stack([u1,u2,u3]).normalized()*tmax
            
            #u = vf.Stack([u12,u3])
            super().__init__(u)
            


def Analytic(Ivec,wvec,h,t):
    I1 = Ivec[0]
    I2 = Ivec[1]
    I3 = Ivec[2]
    
    wi1 = wvec[0]
    wi2 = wvec[1]
    wi3 = wvec[2]
    
    w3 = wi3*(1-t/h)
    
    K1 = (I2 - I3)/I1
    K2 = (I1 - I3)/I2
    
    KK = (I2 - I3)/I1
    
    K = np.sqrt(K1*K2)*K1/abs(K1)
   
    C1 = wi2*np.sqrt(K/K2)
    C2 = wi1*np.sqrt(K/K1)
    
    
    Eta = K*wi3*(t-(t**2)/(2*h))
    Etadot = K*w3
    
    fw = (1-t/h)
    
    w1 = (C1*fw*np.sin(Eta) +  C2*fw*np.cos(Eta))*np.sqrt(K1/K)
    w2 = (C1*fw*np.cos(Eta) -  C2*fw*np.sin(Eta))*np.sqrt(K2/K)
    
    return w1,w2,w3


def AnalyticDT(Ivec,wvec,h,ts):
    I1 = Ivec[0]
    I2 = Ivec[1]
    I3 = Ivec[2]
    
    wi1 = wvec[0]
    wi2 = wvec[1]
    wi3 = wvec[2]
    
    t = sp.symbols("t")
    
    K1 = (I2 - I3)/I1
    K2 = (I1 - I3)/I2
    K3 = (I1-I2)/I3
    K = np.sqrt(K1*K2)*K1/abs(K1)
    
    C1 = wi2*np.sqrt(K/K2)
    C2 = wi1*np.sqrt(K/K1)
    FW= (1-t/h)
    
    Eta  = K*wi3*(t-(t**2)/(2*h))
    ##############################################
    KW = abs(K*wi3)
    print(KW)
    OScale = 1/(4*h*h*KW**1.5)
    FScale = np.sqrt(2*np.pi)*(h**1.5)
    TrigScale = 2*h*np.sqrt(KW)*(h-t)
    TrigArg = KW*t*(2*h-t)/h
    FresArgScale = np.sqrt(2*KW/(np.pi*h))
    FresArg = FresArgScale*(h-t)
    
    Kinv = sp.sqrt(K1/K2)
   
    SinScale =  np.sign(K*wi3)*K3*OScale*((wi2**2)*(Kinv)-(wi1**2)/(Kinv))/2
    CosScale =  K3*OScale*(wi2*wi1)

    SinFres = SinScale*(
      FScale*np.cos(h*KW)*sp.fresnelc(FresArg) 
    + FScale + np.sin(h*KW)*sp.fresnels(FresArg)
    - TrigScale*sp.cos(TrigArg)
    )
    

    
    CosFres = CosScale*(
      -FScale*np.sin(h*KW)*sp.fresnelc(FresArg) 
    + FScale + np.cos(h*KW)*sp.fresnels(FresArg)
    + TrigScale*sp.sin(TrigArg)
    )
    
    CosInt = sp.sin(TrigArg)*h*h/np.sqrt(KW)
    SinInt = -sp.cos(TrigArg)*h*h/np.sqrt(KW)
    
    CosFresInt1 = sp.fresnelc(FresArg)*(t-h) + sp.sin(0.5*np.pi*(FresArg**2))/(np.pi*FresArgScale)
    SinFresInt1 = sp.fresnels(FresArg)*(t-h) - sp.cos(0.5*np.pi*(FresArg**2))/(np.pi*FresArgScale)

    ######################################
    SinFresInt = SinScale*(
      FScale*np.cos(h*KW)*CosFresInt1 
    + FScale*t + np.sin(h*KW)*SinFresInt1
    - CosInt
    )
    

    
    CosFresInt = CosScale*(
      -FScale*np.sin(h*KW)*CosFresInt1
    + FScale*t + np.cos(h*KW)*SinFresInt1
    + SinInt
    )
    ################################

    W3temp = CosFres + SinFres
    W3Inttemp = CosFresInt + SinFresInt

    W3C    = CosFres + SinFres - W3temp.evalf(subs={t:0})
    W3CInt = CosFresInt + SinFresInt - W3Inttemp.evalf(subs={t:0})

    CE =  -W3C.evalf(subs={t:h})
    W3I = (1-t/h)*(wi3 - CE)
    
    W3F = W3I + W3C + CE
    
    Eta = K*sp.integrate(W3I + CE - W3temp.evalf(subs={t:0})) + K*W3CInt
    Eta = Eta - Eta.evalf(subs={t:0})
    #print(Eta.evalf(subs={t:0}))
    #########################################
    
    Etas = np.array([float(Eta.evalf(subs={t:ti})) for ti in ts])
    FW= (1-ts/h)

    W1 = (C1*FW*np.sin(Etas) +  C2*FW*np.cos(Etas))*np.sqrt(K1/K)
    W2 = (C1*FW*np.cos(Etas) -  C2*FW*np.sin(Etas))*np.sqrt(K2/K)
    W3 = np.array([float(W3F.evalf(subs={t:ti})) for ti in ts])
    
    
    
    
    return W1,W2,W3


    


def Analytic2(Ivec,wvec,h,ts,W33):
    I1 = Ivec[0]
    I2 = Ivec[1]
    I3 = Ivec[2]
    
    wi1 = wvec[0]
    wi2 = wvec[1]
    wi3 = wvec[2]
    
    t = sp.symbols("t")
    
    K1 = (I2 - I3)/I1
    K2 = (I1 - I3)/I2
    
    K = (I2 - I3)/I1
    K = np.sqrt(K1*K2)*K1/abs(K1)
    #K = (K1+K2)/2

    eps = (I1-I2)
    
    Eta  = K*wi3*(t-(t**2)/(2*h))
    Eta1 = ((I1 - I3)/I1)*wi3*(t-(t**2)/(2*h))
    Eta2 = ((I2 - I3)/I2)*wi3*(t-(t**2)/(2*h))
    Etadot = K*wi3*(1-t/h)
    fw = (1-t/h)
    print(K-K1)
    print(K-K2)
    Kf = I1/I2
    Kf=1
    w1c = (wi2*fw*sp.sin(Eta1) +  wi1*fw*sp.cos(Eta1))
    w2c = (wi2*fw*sp.cos(Eta2) -  wi1*fw*sp.sin(Eta2))
    w3 = wi3*(1-t/h)
    
    w1 = (wi2*fw*sp.sin(Eta) +  wi1*fw*sp.cos(Eta))*np.sqrt(K1/K)
    w2 = (wi2*fw*sp.cos(Eta) -  wi1*fw*sp.sin(Eta))*np.sqrt(K2/K)
    
    WI1, WI2,WI3, H, KK = sp.symbols(" WI1 WI2  WI3 H KK")
    ETA  = KK*WI3*(t-(t**2)/(2*H))
    FW = (1-t/H)
    w1T = (WI2*FW*sp.sin(ETA) +  WI1*FW*sp.cos(ETA))
    w2T = (WI2*FW*sp.cos(ETA) -  WI1*FW*sp.sin(ETA))
    
    
    wd3c =  lambdify(t,eps*w1*w2/I3)
    
    kj = (sp.trigsimp(sp.simplify(w2T*w1T)))

    
    print(kj)
    p = .5
    EtaR = K*wi3*((t**2)/(2*h)) + 10.5
    fw2 = t/h
    Etah = K*wi3*((h)/(2)) + 10.5
    C2 = (wi1 - wi2*np.tan( Etah))/(np.tan( Etah)*np.sin( Etah) + np.cos( Etah))
    C1 = C2*np.tan( Etah) + wi2/np.cos( Etah)
    w11 = (C1*fw2*sp.sin(EtaR) +  C2*fw2*sp.cos(EtaR))
    w21 = (C1*fw2*sp.cos(EtaR) -  C2*fw2*sp.sin(EtaR))
    kk = sp.integrate(sp.simplify(w21*w11),t)
    
    test1 = lambdify(t,w2)(ts)

    
    '''
    polw = np.polyfit(ts,wd3c(ts),31)
    
    print(polw)
    print(polw[::-1])
    
    f3 = 0
    for p,c in enumerate(polw[::-1]):
        f3+= c*(t**p)

    F3 = lambdify(t,f3)
    
    cs = mp.fourier(lambdify(t,eps*w1*w2/I3,modules='mpmath'),[0,h],35)
    coeff = [mp.fourierval(cs, [0,h], ti) for ti in ts]
    plt.plot(ts,wd3c(ts))
    plt.plot(ts,coeff)
    plt.plot(ts,F3(ts))
    plt.show()
    '''

    
    W1 = lambdify(t,w1)(ts)
    W2 = lambdify(t,w2)(ts)
    W3 = lambdify(t,w3)(ts)
    
    W3C = np.array([scipy.integrate.quad(wd3c,0,t)[0] for t in ts])
    
    w1dx = K*W2*W3C
    
    W1C = [0]
    
    for i in range(1,len(ts)):
        W1C.append(scipy.integrate.trapz(w1dx[0:i],ts[0:i]))

    W1C= np.array(W1C)#*(1-ts)
    
    
    
    #W1 += W1C
    '''
    test3 = np.array([float(kk.evalf(subs={t:-ti})*eps/I3) for ti in ts[::-1]])
    test2 = lambdify(t,w21.subs(t,-t))(ts[::-1])

    plt.plot(ts,test1)
    plt.plot(ts, test2)
    plt.show()
    '''
    W3 += (W3C)/1.0
    
    wd3c =  lambdify(t,w3 + eps*w1*w2/I3)

    '''
    for j in range(0,1):
        W3CI = [0]
        for i in range(1,len(ts)):
            W3CI.append(scipy.integrate.trapz(W3[0:i],ts[0:i]))
            
        
        C1 = wi2*np.sqrt(K/K2)
        C2 = wi1*np.sqrt(K/K1)
        for i in range(0,len(ts)):
            
            FW= (1-ts[i]/h)
            #FW = W3[i]/wi3
            W1[i] = (C1*FW*np.sin(W3CI[i]*K) +  C2*FW*np.cos(W3CI[i]*K))*np.sqrt(K1/K)
            W2[i] = (C1*FW*np.cos(W3CI[i]*K) -  C2*FW*np.sin(W3CI[i]*K))*np.sqrt(K2/K)
           
        W3 = lambdify(t,w3)(ts)
        
        W3D = W1*W2*eps/I3
        W3C=np.zeros((len(ts)))
        
        for i in range(1,len(ts)):
            W3C[i]+=scipy.integrate.trapz(W3D[0:i],ts[0:i])
            
        CE = -W3C[-1]
        CI =wi3 - CE
        
        W3 =  CI*(1-ts/h) + W3C + CE
        '''
    
    #W3 += W3C
    return W1,W2,W3



    
Ivec = np.array([105,100,200])
h   =320
ode = QuatModel(Ivec)
integ = ode.integrator(.01)
IG = np.zeros((11))
IG[3]=1.0
IG[4]= .3
IG[5]= .4
IG[6]= 0.5
IG[8]=0

'''
IG[4]=-.165
IG[5]=-.165
IG[6]=.165
'''

Tmax = 0.5

h = np.linalg.norm(Ivec*IG[4:7])/Tmax

Traj = integ.integrate_dense(IG,9*np.pi/2,1000)
#AnimSlew(Traj,Anim=False,Elev=45,Azim=315)

dtinteg  = oc.ode_x_x.integrator(ode,.001,ode.PCon2(Ivec,IG[4:7],h),[4,5,6,7])
dtinteg2 = oc.ode_x_x.integrator(ode,.001,ode.PCon3(Ivec,1.45),[4,5,6,7])

#dtinteg = oc.ode_7_3.integrator(ode,.001,ode.DetumblePCon(12),[4,5,6])

def BreakFunc(x):
        mr = max(abs(x[4:7]))
        if(mr<.01):return True
        else:return False


Traj = dtinteg.integrate_dense(IG,h*.9999999,5500)
Traj2 = dtinteg2.integrate_dense(IG,h,5500)

ts = [T[7] for T in Traj]
U1 = [np.linalg.norm(T[8:11]) for T in Traj]
U2 = [np.linalg.norm(T[8:11]) for T in Traj2]

plt.plot(ts,U1)
plt.plot(ts,U2)
plt.show()




AnimSlew(Traj,Anim=True,Elev=45,Azim=315,Ivec=Ivec)

phase= ode.phase(Tmodes.LGL3,Traj,1500)
#phase.setControlMode(oc.BlockConstant)
phase.addBoundaryValue(PhaseRegs.Front,range(0,8),IG[0:8])
phase.addBoundaryValue(PhaseRegs.Back,range(4,7),[0,0,0])
phase.addLUVarBounds(PhaseRegs.Path,[8,9,10],-50,50,1.0)
#phase.addDeltaTimeObjective(0.1)
phase.addIntegralObjective(0.5*Args(3).head3().vf().squared_norm(),[8,9,10])
phase.optimizer.MaxAccIters=150
phase.optimizer.deltaH=1.0e-7
phase.optimizer.QPThreads=6
#phase.addUpperVarBound(PhaseRegs.Back,7,h)
phase.addBoundaryValue(PhaseRegs.Back,[7],[h])
#phase.addDeltaTimeObjective(1.0)

phase.optimize()

Traj  = phase.returnTraj()
Ctraj = phase.returnCostateTraj()


TT = np.array(Traj).T

t = TT[7]

w1 = TT[4]
w2 = TT[5]
w3 = TT[6]

n1 = -1
n2 = 150


w1A,w2A,w3A = Analytic(Ivec,IG[4:7],h,t)
w1B,w2B,w3B = AnalyticDT(Ivec,IG[4:7],h,t)

t2 = t


plt.plot(t,w1,color='red',label='w1')
plt.plot(t,w2,color='blue',label='w2')
plt.plot(t,w3,color='green',label='w3')

plt.plot(t2,w1A,color='red',linestyle='--')
plt.plot(t2,w2A,color='blue',linestyle='--')
plt.plot(t2,w3A,color='green',linestyle='--')

plt.plot(t2,w1B,color='red',linestyle='dotted')
plt.plot(t2,w2B,color='blue',linestyle='dotted')
plt.plot(t2,w3B,color='green',linestyle='dotted')

plt.xlabel("t(s)")
plt.ylabel("w (rad/s)")
plt.legend()
plt.grid(True)
plt.show()


plt.plot(t2, ((w1A*Ivec[0])**2+(w2A*Ivec[1])**2+(w3A*Ivec[2])**2)**.5)
plt.plot(t2, ((w1*Ivec[0])**2+(w2*Ivec[1])**2+(w3*Ivec[2])**2)**.5)

plt.grid(True)
plt.show()


plt.plot(t2,w1-w1B,color='red')
plt.plot(t2,w2-w2B,color='blue')
plt.plot(t2,w3-w3B,color='green')
plt.xlabel("t(s)")
plt.ylabel("w error (rad/s)")

plt.grid(True)
plt.show()
#plt.plot(t2,(((w1-w1A)*Ivec[0])**2 + ((w2-w2A)*Ivec[0])**2)**.5)


plt.grid(True)
plt.show()



#for i,T in enumerate(Traj):
    #T[4:6] = vf.Normalized(2).compute(T[4:6])
    #T[6]=0
    #T[8:11] = Ctraj[i][4:7]/Ivec
    

AnimSlew(Traj,Anim=False,Elev=45,Azim=315)





