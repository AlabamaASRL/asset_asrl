import numpy as np
import asset as ast
from QuatPlot import AnimSlew
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify, implemented_function
import scipy as scipy
from sympy import fourier_series
import mpmath as mp
from scipy.spatial.transform import Rotation as R
norm = np.linalg.norm
def normalize(x): return np.copy(x)/norm(x)
    

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
            
            u1  = -I1*args[0]/(-1*t.sf() + np.array([h*1.000000000000]) )
            u2  = -I2*args[1]/(-1*t.sf() + np.array([h*1.000000000000]) )
            u3  = -I3*args[2]/(-1*t.sf() + np.array([h*1.000000000000]) )
            
            
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

def AnalyticQ(Ivec,wvec,q0,h,ts):
    
    I1 = Ivec[0]
    I2 = Ivec[1]
    I3 = Ivec[2]
    
    wi1 = wvec[0]
    wi2 = wvec[1]
    wi3 = wvec[2]
    
   
    
    K1 = (I2 - I3)/I1
    K2 = (I1 - I3)/I2
    
    KK = (I2 - I3)/I1
    
    K = np.sqrt(K1*K2)*K1/abs(K1)
   
    C1 = wi2*np.sqrt(K/K2)
    C2 = wi1*np.sqrt(K/K1)
    Ans=[]
    for t in ts:
        
        Eta = K*wi3*(t-(t**2)/(2*h))
       
        
        fw = (1-t/h)
        
        w1 = (C1*fw*np.sin(Eta) +  C2*fw*np.cos(Eta))*np.sqrt(K1/K)
        w2 = (C1*fw*np.cos(Eta) -  C2*fw*np.sin(Eta))*np.sqrt(K2/K)
        w3 = wi3*(1-t/h)
        
        
        w1I = (-C1*np.cos(Eta) +  C2*np.sin(Eta) + C1)*np.sqrt(K1/K)/(K*wi3)
        w2I = (C1*np.sin(Eta) +  C2*np.cos(Eta) - C2)*np.sqrt(K2/K)/(K*wi3)
        w3I = Eta/K
        
        WI = np.array([w1I,w2I,w3I])*0.5
        qw= np.zeros((4))
        qw[3]   = np.cos(norm(WI))
        if(t>0):
            qw[0:3] = np.sin(norm(WI))*normalize(WI)
        
        M1 = np.array([[0, -w1,-w2,-w3],
                       [w1,  0, w3,-w2],
                       [w2,-w3,  0, w1],
                       [w3, w2,-w1,  0]])
    
        M2 = np.array([[0, -w1I,-w2I,-w3I],
                       [w1I,  0, w3I,-w2I],
                       [w2I,-w3I,  0, w1I],
                       [w3I, w2I,-w1I,  0]])
    
        A=(np.dot(M1,M2) - np.dot(M2,M1))
        B=(2*np.dot(M1,M2))
       
        
        R1 = R.from_quat(qw)
        R2 = R.from_quat(q0)
        
        
        qt = (R1*R2).as_quat()
       
        a  = np.array([qt[0],qt[1],qt[2],qt[3],w1,w2,w3,t])
        Ans.append(a)
    return Ans
    
    
def FreeTest(Ivec,wvec,h,ts):
    I1 = Ivec[0]
    I2 = Ivec[1]
    I3 = Ivec[2]
    
    K1 = (I2 - I3)/I1
    K2 = (I1 - I3)/I2
    K3 = (I1 - I2)/I3
    K = np.sqrt(abs(K1*K2*K3))*K1/abs(K1)
    
    wi1 = wvec[0]
    wi2 = wvec[1]
    wi3 = wvec[2]
    WT = []
    
    Lv=Ivec*wvec
    L = norm(Lv)
    Er =np.dot(Lv,Lv/Ivec)/2.0
    m = (L**2 -2*I3*Er)*(I1 - I2)/(  (L**2 -2*I1*Er)*(I3 - I2)  )
    
    w1m = (abs(wi1)/wi1)* np.sqrt(   (L**2 -2*I3*Er)/(I1*(I1-I3))   )
    w2m = -(abs(wi1)/wi1)* np.sqrt(   (L**2 -2*I3*Er)/(I2*(I2-I3))   )
    w3m = (abs(wi3)/wi3)* np.sqrt(   (L**2 -2*I1*Er)/(I3*(I3-I1))   )
    wp  = (abs(wi3)/wi3)*(abs(I2-I3)/(I2-I3))*np.sqrt(  (L**2 -2*I1*Er)*(I3-I2)/(I1*I2*I3)   )
    
    print(K*wi3,wp)
    eps = scipy.special.ellipkinc(np.arcsin(wi2/w2m),m)
    
    
    K   = scipy.special.ellipkinc(np.arcsin(1),m)
    Kp  = scipy.special.ellipkinc(np.arcsin(1),1-m)
    q   = np.exp(-np.pi*Kp/K)
    eta = (abs(wi3)/wi3)*Kp - scipy.special.ellipkinc(np.arcsin(I3*w3m/L),1-m)
    zeta = np.exp(np.pi*eta/K)
    
    A2 = L/I1 + np.pi*wp*(zeta + 1)/(2*K*(zeta-1))
    
    
    
    for i in range(1,15):
        dA = -(np.pi*wp/K)*((q**(2*i))/(1-(q**(2*i))))*(zeta**i - zeta**(-i))
        A2 +=dA
        
    NT = 15
    cr = []
    ci = []
    r0 = 0
    i0 =0
    for i in range(0,NT):
        CR = ((-1)**(i))*(2*q**(i*i+i + .25))*np.cosh((2*i+1)*(np.pi*eta)/(2*K))
        CI = ((-1)**(i+1))*(2*q**(i*i+i + .25))*np.sinh((2*i+1)*(np.pi*eta)/(2*K))
        r0 +=CR*np.sin((2*i+1)*(np.pi*eps)/(2*K))
        i0 +=CI*np.cos((2*i+1)*(np.pi*eps)/(2*K))
        cr.append(CR)
        ci.append(CI)
        
    if(r0>0):k=0
    else:k = abs(i0)/i0
    
    A1 = np.arctan(i0/r0) + k*np.pi
    
    for j,t in enumerate(ts):
        
        Re1 =0
        Im1 =0
        
        for i in range(0,NT):
            Re1+=cr[i]*np.sin((2*i+1)*np.pi*(wp*(t-0.5*t*t/h) + eps)/(2*K))
            Im1+=ci[i]*np.cos((2*i+1)*np.pi*(wp*(t-0.5*t*t/h) + eps)/(2*K))
        
        C = np.cos(A1 + A2*(t-0.5*t*t/h))
        S = np.sin(A1 + A2*(t-0.5*t*t/h))
        
        cphi = (C*Re1 + S*Im1)/np.sqrt(Re1**2 + Im1**2)
        sphi = (S*Re1 - C*Im1)/np.sqrt(Re1**2 + Im1**2)
        
        
        sn,cn,dn,ph = scipy.special.ellipj(wp*(t-0.5*t*t/h) + eps ,m)
        #sn,cn,dn,ph = scipy.special.ellipj(wp*(t) + eps ,m)

        w1 = w1m*cn*(1-t/h)
        w2 = w2m*sn*(1-t/h)
        w3 = w3m*dn*(1-t/h)
       
        
        
        Li = np.array([w1,w2,w3])*Ivec
        Ln  = norm(Li)
        Lp = norm(Li[0:2])
        
        LnLp = Ln*Lp
        
        UT1 = np.array([[Li[0]*Li[2]/LnLp, -Li[1]/Lp,  Li[0]/Ln],
                        [Li[1]*Li[2]/LnLp, Li[0]/Lp,  Li[1]/Ln],
                        [-Lp/Ln,           0     ,       Li[2]/Ln]])
    
        T2 = np.array([[cphi,sphi,0],
                       [-sphi,cphi,0],
                       [0,0,1]])
    
        if(j==0):
            B = np.copy(UT1.T)
            print(R.from_dcm(B).as_quat()[0:3])
            
        
        M = np.matmul(np.matmul(UT1,T2),B)
        F=R.from_dcm(M.T)
        q = R.from_dcm(M.T).as_quat()
        
        if(j==0):
            ql = np.copy(q)
        else:
            k =np.dot(ql,q)
            if(k<0):
                q = q*-1.0
            ql = np.copy(q)
            
        
        
        WT.append([q[0],q[1],q[2],q[3],w1,w2,w3,t])
        
        
    return WT
        
        
        
    
     

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



    
Ivec = np.array([120,150.0,200])
h   =320
ode = QuatModel(Ivec)
integ = ode.integrator(.01)
IG = np.zeros((11))
IG[3]=  1.0
IG[4]= 0.15
IG[5]= 0.25
IG[6]= 0.4
IG[8]=  0

'''
IG[4]=-.165
IG[5]=-.165
IG[6]=.165
'''

Tmax = 10.00

h = np.linalg.norm(Ivec*IG[4:7])/Tmax
print(norm(IG[4:7])*h/2.0)
Traj = integ.integrate_dense(IG,9*np.pi/2,1000)
#AnimSlew(Traj,Anim=False,Elev=45,Azim=315)

dtinteg  = oc.ode_x_x.integrator(ode,.001,ode.PCon2(Ivec,IG[4:7],h),[4,5,6,7])
dtinteg2 = oc.ode_x_x.integrator(ode,.001)
dtinteg2  = oc.ode_x_x.integrator(ode,.0005,ode.PCon2(Ivec,IG[4:7],h),[4,5,6,7])

#dtinteg = oc.ode_7_3.integrator(ode,.001,ode.DetumblePCon(12),[4,5,6])

def BreakFunc(x):
        mr = max(abs(x[4:7]))
        if(mr<.01):return True
        else:return False


Traj = dtinteg.integrate_dense(IG,h*.99999999,9500)


TrajF = dtinteg2.integrate_dense(IG,h*.999999999,1500)


TT = np.array(TrajF).T
t=TT[7]
AT = FreeTest(Ivec,IG[4:7],h,t)
AT = np.array(AT).T





fig,axs= plt.subplots(2,1)


axs[0].plot(t,AT[0],label=r'$q_1$')
axs[0].plot(t,AT[1],label=r'$q_2$')
axs[0].plot(t,AT[2],label=r'$q_3$')
axs[0].plot(t,AT[3],label=r'$q_4$')
axs[0].set_ylabel(r'$q_i(t)$')
axs[0].grid(True)
#axs[0].set_xlabel(r'$t$')

axs[1].plot(t,AT[4],color='r',label=r'$\omega_1$')
axs[1].plot(t,AT[5],color='g',label=r'$\omega_2$')
axs[1].plot(t,AT[6],color='b',label=r'$\omega_3$')
axs[1].set_ylabel(r'$\omega_i(t)$' )
axs[1].grid(True)
axs[1].set_xlabel(r'$t$')
axs[0].legend()
axs[1].legend()

plt.show()

######################
fig,axs= plt.subplots(2,1)

ET0 = abs(AT[0]-TT[0])
ET1 = abs(AT[1]-TT[1])
ET2 = abs(AT[2]-TT[2])
ET3 = abs(AT[3]-TT[3])

ET4 = abs(AT[4]-TT[4])
ET5 = abs(AT[5]-TT[5])
ET6 = abs(AT[6]-TT[6])


QE = (ET0**2 + ET1**2 +ET2**2 +ET3**2 )**.5
WE = (ET4**2 + ET5**2 +ET6**2 )**.5


axs[0].plot(t,QE)
axs[0].set_ylabel('Quaternion Error')
axs[0].grid(True)

axs[1].plot(t,WE)
axs[1].set_ylabel(r'$\omega$' +' Error' )
axs[1].grid(True)
axs[1].set_xlabel(r'$t$')
axs[0].legend()
axs[1].legend()
axs[0].set_yscale('log')
axs[1].set_yscale('log')

plt.show()




AnimSlew(TrajF,Anim=False,Elev=45,Azim=315,Ivec=Ivec)



AnimSlew(Traj,Anim=True,Elev=45,Azim=315,Ivec=Ivec)

phase= ode.phase(Tmodes.LGL3,Traj,1000)
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
#Traj = dtinteg.integrate_dense(IG,h*.9999999,5500)


TT = np.array(Traj).T

t = TT[7]

w1 = TT[4]
w2 = TT[5]
w3 = TT[6]


AT = AnalyticQ(Ivec,IG[4:7],[0,0,0,1],h,t)
AT = np.array(AT).T

plt.plot(t,TT[0],color='r')
plt.plot(t,TT[1],color='g')
plt.plot(t,TT[2],color='b')
plt.plot(t,TT[3],color='k')

plt.plot(t,AT[0],color='r',linestyle='--')
plt.plot(t,AT[1],color='g',linestyle='--')
plt.plot(t,AT[2],color='b',linestyle='--')
plt.plot(t,AT[3],color='k',linestyle='--')

plt.show()


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





