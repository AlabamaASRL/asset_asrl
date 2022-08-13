import asset as ast
import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import default_rng
import scipy.io
import time
################################################################################
# Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes
################################################################################
# Constants\
aMars=227937168.370000         #Km
wMars=1.05857597261100e-07     #rad/second
aEarth=149657721.670000        #km
wEarth=1.99096676795790e-07    #rad/secon
earthAngle0=-2.10405033895696-.01

c1=1 #Max Thrust in Newtons (using 7RIT2X engine)
isp=1600 #seconds(using RIT2X engine)
g0=9.81/1000 #Must be in km s**-2 to be consistant with the rest of the cosntants


m0=3366./2.
mStar=m0
mStar=1965
mu=132712440041.940
print(mu)
input('s')
lStar=aEarth
tStar=np.sqrt((lStar**3)/mu)
dt =(1/1)*30000./tStar#31536.0/self.tStar
tof=1.2*365.*24.*60.*60./tStar
#arg_Val=np.array([132712440041.940,35*1.33/1000,	37.9843])
arg_Val=np.array([132712440041.940,tStar*tStar/(1000*lStar*mStar),	(isp*g0)*tStar/lStar])


entries=400#0#400
nSeg = 1
#################################################################################
# Functions
def ETF_Fun(t):
    ## 3 term
    tStar = 5025649.82622603

    a_val= 0.990737004443420
    c_val=-0.004191597003507
    d_val=-0.014777642420120

    a_val= 1.
    c_val=-0.01083
    d_val=-0.005042
    #breakpoint()
    #ETF=a_val + c_val*vf.exp(d_val*t.coeff(1)*tStar/(24.*60.*60.))
    ETF=a_val + c_val*vf.exp(d_val*t[0]*tStar/(24.*60.*60.))
    return ETF

def noCont():
    args = Args(7)
    rhat = args.head3().normalized()
    vhat = 1.*args.segment_3(3).normalized()#args.tail3().normalized()
    temp =  rhat*0.
    cont = vf.Stack([(args[0]*0)+1,vhat])
    #cont = vf.Stack([(args[0]*0)+1,rhat.normalized()])
    #cont = vf.Stack([(args[0]*0),vhat])
    return cont

def noCont0():
    args = Args(7)
    rhat = args.head3().normalized()
    vhat = 1.*args.segment_3(3).normalized()#args.tail3().normalized()
    temp =  rhat*0.
    cont = vf.Stack([(args[0]*0),vhat])
    #cont = vf.Stack([(args[0]*0)+1,rhat.normalized()])
    #cont = vf.Stack([(args[0]*0),vhat])
    return cont

def state_Target(tab):
    args = Args(7)
    r_val = args.head(6)
    t= args[6]
    #My old way
    stateFun = oc.InterpFunction(tab,range(0,6)).vf()
    e_Val=(stateFun.eval(t) - r_val)
    #breakpoint()
    return e_Val#.norm()

def Pos_Target(tab):
    args = Args(4)
    r_val = args.head_3()
    t= args[3]
    #breakpoint()
    posStateFun = oc.InterpFunction(tab,range(0,3)).vf()
    #
    #e_Val=earthStateFun.compute(earthStateFun.eval(t) - r_val)
    e_Val=(posStateFun.eval(t) - r_val)
    
    return e_Val#.norm()


def Launcher_Constraint(tab):
    args = Args(14)
    t_0= args[9]
    m_0 = args[6]
    v_val = args[3:6]

    #output=integ.integrate_dense(Args,t_0-timeDuration , 3000)
    
    #breakpoint()
    velStateFun = oc.InterpFunction(tab,range(3,6)).vf()
    #
    #e_Val=earthStateFun.compute(earthStateFun.eval(t) - r_val)
    vInf_vec=(velStateFun.eval(t_0) - v_val)
    vInf = vInf_vec.norm()
    maxMassHi = (16.607*(vInf**3)) - (297.41*(vInf**2)) - (107.18*vInf) + 15010.
    maxMassLow = (10.821*(vInf**3)) - (187.12*(vInf**2)) - (25.429*vInf) + 6690.
    maxMass = (1*maxMassHi/3)+(2*maxMassLow/3)
    return m_0-(maxMass/mStar)

def v_Infty_Arrival(tab,max_Vinf_allow):
    args = Args(4)
    v =args.head_3()
    t = args[3]
    velStateFun = oc.InterpFunction(tab,range(3,6)).vf()
    vInfArrival=(velStateFun.eval(t)-v).norm() 
    return vInfArrival- max_Vinf_allow

def vInfMatch(minusState,plusState,tab):
    #angle = (vel_state_minus.normalized*vel_state_plus.normalized).norm()
    v_minus = minusState[0:3]
    v_plus = plusState[0:3]
    t_minus = minusState[3]
    t_plus = plusState[3]

    vel_Mars = oc.InterpFunction(tab,range(3,6)).vf()
    vInfPlus=(vel_Mars.eval(t_plus)-v_plus).norm()
    vInfMinus=(vel_Mars.eval(t_minus)-v_minus).norm()
    return (vInfPlus-vInfMinus)

def flyby_Angle_Constraint(minusState,plusState,tab):
    #angle = (vel_state_minus.normalized*vel_state_plus.normalized).norm()
    v_minus = minusState[0:3]
    v_plus = plusState[0:3]
    t_minus = minusState[3]
    t_plus = plusState[3]

    vel_Mars = oc.InterpFunction(tab,range(3,6)).vf()
    vInfPlus=(vel_Mars.eval(t_plus)-v_plus).normalized() 
    vInfMinus=(vel_Mars.eval(t_minus)-v_minus).normalized() 
    cosThetaActual=vf.dot(vInfPlus,vInfMinus)

    vInfVal=vInfMinus.norm()
    
    r_Mars = 3389.5#rADIUS OF MARS IN KM 
    r_Mars_Plus_Atm_100 = r_Mars+11+100
    r_Mars_Psyche = r_Mars+500
    mu_Val_Mars =4.28283e13* mu/(1.327e20) #Mu value of mars
    phi_2 = vf.arccos(1/(1+(1*r_Mars_Psyche*vInfVal/mu_Val_Mars))) #Old value is 10, new value is 2
    max_Turning_Angle = np.pi - (2*phi_2)
    return (vf.arccos(cosThetaActual)-max_Turning_Angle)

class two_body_coast(oc.ode_x_x.ode):
    def __init__(self,arg_Val,ETF_flag,nuVal):

        self.PrintLevel=2
        Xvars = 9
        Uvars = 4
        Ivars = Xvars + 1 + Uvars
        ############
        args = Args(Ivars)
        r = args.head_3()
        v = args.segment_3(3)
        mass = args[6]
        t = args[9]#args.segment(9,9)#args[9]#args[7]
        t_Forw= args[7]
        t_In_Phase= args[8]#args.segment(8,8)#
        cont_Val = args[10]
        mu =1.#arg_Val[0]

        acc = -mu*r.normalized_power3()

        m_Dot =mass *0
        t_Forw_Dot = (t_Forw[0]*0.)+1.
        t_In_Phase_Dot =  (t_Forw[0]*0.)+1.
        #breakpoint()
        odeeq = vf.Stack([v,acc,m_Dot,t_Forw_Dot,t_In_Phase_Dot])

        super().__init__(odeeq,Xvars,Uvars)

class two_body(oc.ode_x_x.ode):
    def __init__(self,arg_Val,ETF_flag,nuVal):

        self.PrintLevel=2
        Xvars = 9
        Uvars = 4
        Ivars = Xvars + 1 + Uvars
        ############
        args = Args(Ivars)
        r = args.head_3()
        v = args.segment_3(3)
        mass = args[6]
        t = args[9]#args.segment(9,9)#args[9]#args[7]
        t_Forw= args[7]
        t_In_Phase= args[8]#args.segment(8,8)#
        cont_Val = args[10]
        n = args.tail(3)
        angles = n.normalized()
        mu =1.#arg_Val[0]
        c1 = arg_Val[1]
        c2 = arg_Val[2]

        
        #n1 = cont[1:4]
        #breakpoint()
        #u_Val=cont[0]
        radius = r.norm()#.eval(args)

        #power = 10.0*40/(radius**2) #from ERO
        power = (21./(radius**2))-1.
        #power = (2.3*(3.3**2)/(radius**2))-1.
        #power = np.minimum(power,40);
        powerToThrustM =0.0471
        #powerToThrustB =0.0
        thrustAvailableDim=(powerToThrustM*power)#In Netwons
        thrustAvailable = (thrustAvailableDim/1000)*(tStar**2)/(lStar*mStar) #ND
        if(ETF_flag):
        
            ETF = ETF_Fun(t_Forw[0])**nuVal
        else:
            ETF=1.0*(1.-nuVal)

        #x_F=thrustAvailable*c1*ETF * u_Val *angles[0]/mass[0]
        #y_F=thrustAvailable*c1*ETF * u_Val *angles[1]/mass[0]
        #z_F=thrustAvailable*c1*ETF * u_Val *angles[2]/mass[0]
        #acc_Thrust = vf.Stack([x_F,y_F,z_F])
        #x_F=thrustAvailable*c1*ETF * u_Val/mass[0]
        #y_F=thrustAvailable*c1*ETF * u_Val/mass[0]
        #z_F=thrustAvailable*c1*ETF * u_Val/mass[0]
        acc_Thrust = thrustAvailable*ETF *cont_Val[0] * angles/mass[0]
        acc = (-mu*r.normalized_power3())+ acc_Thrust

        m_Dot =-1.0*thrustAvailable * cont_Val[0] /c2
        t_Forw_Dot = (t_Forw[0]*0.)+1.
        t_In_Phase_Dot =  (t_Forw[0]*0.)+1.
        #breakpoint()
        odeeq = vf.Stack([v,acc,m_Dot,t_Forw_Dot,t_In_Phase_Dot])

        super().__init__(odeeq,Xvars,Uvars)

def launchVehicleConstraint():
    #TODO **Implement launch vehicle constrint**
    return

def durationConstrint(duration_Val):
    args = Args(2)
    t=args[0]
    return t-duration_Val
        
def coast_con(dur):
    args = Args(2)
    t=args[0]
    u=args[1]
    uMax=1.2
    if(t<dur):
        uMax=.001
    
    return u-uMax

    
def opt_Traj_free(IG, arg_Res):

    
    ETF_On=arg_Res['ETF']
    nu_val=arg_Res['nu']
    DutyCycleMax=arg_Res['Max Duty Cycle']
    DutyCycleMaxFinal=arg_Res['Final Duty Cycle']
    arrival_Coast = 30.+arg_Res['Pre MGA Excess Coast']
    #TODO **Update trajecotry generator for multiple Phases**
    
    #IG2=np.array(IG).T

    
    #Earth-Mars-Ceres Flyby
    ocp = oc.OptimalControlProblem()

    ocp.optimizer.PrintLevel =3
    ocp.optimizer.QPThreads =6
    ocp.optimizer.OptLSMode =ast.LineSearchModes.L1
    ocp.optimizer.MaxLSIters =1
    ocp.Threads =12
    #ocp.optimizer.EContol = 1e-4
    #ocp.optimizer.IContol = 1e-4
    #ocp.optimizer.KKTtol = 1e-4
    #ocp.optimizer.deltaH = 1.0e-6
   

    ## Setup Earth Departure Coast
    phase_E2M_c1 = two_body_coast(arg_Val,ETF_On,nu_val).phase(tModes.LGL3)
    phase_E2M_c1.setTraj(IG[0], 40) #TODO **Setup better entries calculator**
    phase_E2M_c1.addLowerDeltaTimeBound(0)

    ###Earth and Launcher Constraint
    phase_E2M_c1.addEqualCon(PhaseRegs.Front, Pos_Target(earthTab), [0,1,2,9])
    phase_E2M_c1.addInequalCon(PhaseRegs.Front, Launcher_Constraint(earthTab), range(0,14))

    ### Duration Constraint
    phase_E2M_c1.addBoundaryValue(PhaseRegs.Back,[8], [90.*(60*60*24)/tStar] )

    ### Regularization constraints
    phase_E2M_c1.addLUVarBound(PhaseRegs.Path, 6, .5, 10.)
    phase_E2M_c1.addLUVarBound(PhaseRegs.Front, 9, start_times[0]/tStar, start_times[-1]/tStar)  
    phase_E2M_c1.addBoundaryValue(PhaseRegs.Front, [7], [0.])
    phase_E2M_c1.addBoundaryValue(PhaseRegs.Front, [8], [0.])
    
    ## Setup Low Thrust to Mars
    phase_E2M_c2 = two_body(arg_Val,ETF_On,nu_val).phase(tModes.LGL3)
    phase_E2M_c2.setTraj(IG[1], entries)
    phase_E2M_c2.addLowerDeltaTimeBound(0)

    ### Regularization constraints
    phase_E2M_c2.addLUVarBound(PhaseRegs.Path, 10, 0., DutyCycleMax)
    phase_E2M_c2.addLUVarBound(PhaseRegs.Path, 6, .5, 10.)
    phase_E2M_c2.addLUNormBound(PhaseRegs.Path, [11, 12, 13], 0.9, 1.1)#0.7, 1.3
    phase_E2M_c2.addLUNormBound(PhaseRegs.Path, [0, 1, 2], 0.7, np.sqrt(21))#0.7, 1.3
    phase_E2M_c2.addBoundaryValue(PhaseRegs.Front, [8], [0.])


    ## Setup Pre-Martian Flyby Coast
    phase_E2M_c3 = two_body_coast(arg_Val,ETF_On,nu_val).phase(tModes.LGL3)
    phase_E2M_c3.setTraj(IG[2], 40) #TODO **Setup better entries calculator**
    phase_E2M_c3.addLowerDeltaTimeBound(0)

    ### Martian Arrival Constraint
    phase_E2M_c3.addEqualCon(PhaseRegs.Back, Pos_Target(marsTab), [0,1,2,9])

    ### Duration Constraint
    phase_E2M_c3.addBoundaryValue(PhaseRegs.Back, [8], [(arrival_Coast)*(60*60*24)/tStar])

    ### Regularization constraints
    phase_E2M_c3.addBoundaryValue(PhaseRegs.Front, [8], [0.])
    phase_E2M_c3.addLUVarBound(PhaseRegs.Back, 9, mars_t_Range_ext[0]/tStar, mars_t_Range_ext[-1]/tStar)

    

    ## Setup Martian Departure Coast
    phase_M2C_c4 = two_body_coast(arg_Val,ETF_On,nu_val).phase(tModes.LGL3)
    phase_M2C_c4.setTraj(IG[3], 30) #TODO **Setup better entries calculator**
    phase_M2C_c4.addLowerDeltaTimeBound(0)
  
    ### Martian Position Constraint
    phase_M2C_c4.addEqualCon(PhaseRegs.Front, Pos_Target(marsTab), [0,1,2,9])

    ### Duration Constraint
    phase_M2C_c4.addBoundaryValue(PhaseRegs.Back, [8], [10.*(60*60*24)/tStar])

    ### Regularization constraints
    phase_M2C_c4.addLUVarBound(PhaseRegs.Front, 9, mars_t_Range_ext[0]/tStar, mars_t_Range_ext[-1]/tStar)
    phase_M2C_c4.addBoundaryValue(PhaseRegs.Front, [8], [0.])

    ## Setup Low Thrust - Full to Psyche
    phase_M2C_c5 = two_body(arg_Val,ETF_On,nu_val).phase(tModes.LGL3)
    phase_M2C_c5.setTraj(IG[4], entries)
    phase_M2C_c5.addLowerDeltaTimeBound(0)

    #### Psyche Arrival Constraint
    #phase_M2C_c5.addEqualCon(PhaseRegs.Back,Pos_Target(psycheTab), [0,1,2,9]) #Old rondesou constrint
    #phase_M2C_c5.addInequalCon(PhaseRegs.Back, v_Infty_Arrival(psycheTab,max_Vinf),[3,4,5,9])
    
    ### Regularization constraints
    phase_M2C_c5.addLUVarBound(PhaseRegs.Path, 10, 0., DutyCycleMax)
    phase_M2C_c5.addLUVarBound(PhaseRegs.Path, 6,.5, 10.)
    phase_M2C_c5.addLUNormBound(PhaseRegs.Path, [11, 12, 13], 0.9, 1.1)#0.7, 1.3
    phase_M2C_c5.addLUNormBound(PhaseRegs.Path, [0, 1, 2], 0.7, np.sqrt(21))#0.7, 1.3
    phase_M2C_c5.addBoundaryValue(PhaseRegs.Front, [8], [0.])
    
    #phase_M2C_c5.addLUVarBound(PhaseRegs.Back, 9, end_times[0]/tStar, end_times[-1]/tStar)

    ## Setup Low Thrust - 50\% to Psyche
    phase_M2C_c6 = two_body(arg_Val,ETF_On,nu_val).phase(tModes.LGL3)
    phase_M2C_c6.setTraj(IG[5], entries)
    phase_M2C_c6.addLowerDeltaTimeBound(0)

    ### Psyche Arrival Constraint
    phase_M2C_c6.addEqualCon(PhaseRegs.Back,state_Target(psycheTab), [0,1,2,3,4,5,9]) #Old rondesou constrint
    #phase_M2C_c6.addInequalCon(PhaseRegs.Back, v_Infty_Arrival(psycheTab,max_Vinf),[3,4,5,9])
    phase_M2C_c6.addLUVarBound(PhaseRegs.Back, 9, end_times[0]/tStar, end_times[-1]/tStar)
    ### Regularization constraints
    phase_M2C_c6.addLUVarBound(PhaseRegs.Path, 10, 0., (1-DutyCycleMaxFinal))
    #phase_M2C_c6.addLUVarBound(PhaseRegs.Path, 10, 0., (1-DutyCycleMaxFinal))
    phase_M2C_c6.addLUVarBound(PhaseRegs.Path, 6,.5, 10.)
    phase_M2C_c6.addLUNormBound(PhaseRegs.Path, [11, 12, 13], 0.9, 1.1)#0.7, 1.3
    phase_M2C_c6.addLUNormBound(PhaseRegs.Path, [0, 1, 2], 0.7, np.sqrt(21))#0.7, 1.3
    phase_M2C_c6.addBoundaryValue(PhaseRegs.Front, [8], [0.])
    phase_M2C_c6.addBoundaryValue(PhaseRegs.Back, [8], [100.*60*60*24/tStar])


    ### Cost Function
    #phase_M2C.addValueObjective(PhaseRegs.Back,7,1)
    phase_M2C_c6.addValueObjective(PhaseRegs.Back,6,-1.)
    #phase_M2C_c6.addValueObjective(PhaseRegs.Back,7,1.)
    #i_val_obj=phase_M2C.addValueObjective(PhaseRegs.Back,6,-1.0)
    

    #print('\t Combining Phases')
    ocp.addPhase(phase_E2M_c1)
    ocp.addPhase(phase_E2M_c2)
    ocp.addPhase(phase_E2M_c3)
    ocp.addPhase(phase_M2C_c4)
    ocp.addPhase(phase_M2C_c5)
    ocp.addPhase(phase_M2C_c6)


    ## Link 0-1, 1-2, 3-4
    ocp.addForwardLinkEqualCon(0, 1, [0, 1, 2,3,4,5,6,7,9])
    ocp.addForwardLinkEqualCon(1, 2, [0, 1, 2,3,4,5,6,7,9])
    ocp.addForwardLinkEqualCon(3, 4, [0, 1, 2,3,4,5,6,7,9])
    ocp.addForwardLinkEqualCon(4, 5, [0, 1, 2,3,4,5,6,7,9])

    ## Link 2-3
    ocp.addForwardLinkEqualCon(2, 3, [0, 1, 2,6,7,9])

    PhaseRegs.Back
    #linkFun = oc.LinkConstraint(calcAngleTest(Args(6).head(3),Args(6).tail(3)),oc.LinkFlags.BackToFront,[[0,1]],range(3,6))
    linkFun_eqalVel = oc.LinkConstraint(vInfMatch(Args(8).head(4),Args(8).tail(4),marsTab),oc.LinkFlags.BackToFront,[[2,3]],[3,4,5,9])
    linkFun_flyby = oc.LinkConstraint(flyby_Angle_Constraint(Args(8).head(4),Args(8).tail(4),marsTab),oc.LinkFlags.BackToFront,[[2,3]],[3,4,5,9])

    ocp.addLinkEqualCon(linkFun_eqalVel)
    ocp.addLinkInequalCon(linkFun_flyby)

    #LinkObjective =asset.OptimalControl.LinkConstraint()

    #ocp.addLinkInequalCon(linkFun)
    #breakpoint()
    #convFlag = phase_E2M.optimize()
    #print('\t Optimization')
    #c_flag =ocp.solve_optimize()
    c_flag =ocp.solve_optimize()


    #phase_E2M.setTraj( ocp.Phases[0].returnTraj(), 3*entries)
    #phase_M2C.setTraj( ocp.Phases[1].returnTraj(), 5*entries)
    #phase_M2C.addValueObjective(PhaseRegs.Back,6,-1.)
    #phase_M2C.addValueObjective(PhaseRegs.Back,7,1)
    #ocp.solve_optimize()
    return c_flag,ocp






#################################################################################
spice.furnsh("standard.html")

steps =1000#81
# we are going to get positions between these two dates
utc_start = ['Aug 1, 2022']
utc_MarsFlyBy = ['May 15, 2023']
utc_end = ['Jan 1, 2026']
secondsInAYear=  365.*24.*60.*60.
# get et values one and two, we could vectorize str2et
etOne = spice.str2et(utc_start[0])
etTwo = spice.str2et(utc_end[0])
etMars = spice.str2et(utc_MarsFlyBy[0])

etOneD =etOne# 847574110.2730876-(secondsInAYear*.5)#Local Optimal for Flyby
etTwoA = etTwo#984823554.183919

tSpan_tot = etTwoA-etOneD
yearMult=.06
#yearMult=.15
#yearMult=4
start_times = np.linspace(etOneD-(yearMult*secondsInAYear),etOneD+(yearMult*secondsInAYear),steps)
end_times = np.linspace(etTwoA-(yearMult*secondsInAYear),etTwoA+(yearMult*secondsInAYear),steps+1)

#mars_t_Range_ext  = np.linspace(etMars-(.5*secondsInAYear),etMars+(.5*secondsInAYear),steps)#np.linspace(start_times[0],end_times[1],steps*100)
mars_t_Range_ext  = np.linspace(etMars-(1*yearMult*secondsInAYear),etMars+(1*yearMult*secondsInAYear),steps)

dvHolder = np.zeros((steps+1,steps))

stateMars, lightTimes = spice.spkezr('Mars', mars_t_Range_ext, 'J2000', 'NONE', 'SUN')
statePsyche, lightTimes = spice.spkezr('PSYCHE', end_times, 'J2000', 'NONE', 'SUN')
stateEarth, lightTimes = spice.spkezr('EARTH', start_times, 'J2000', 'NONE', 'SUN')
stateMars_ext = np.asarray(stateMars)
stateEarth  = np.asarray(stateEarth)
statePsyche = np.asarray(statePsyche)


stateMarsPlot, lightTimes = spice.spkezr('Mars',  np.linspace(etMars-(1*secondsInAYear),etMars+(1*secondsInAYear),steps), 'J2000', 'NONE', 'SUN')
statePsychePlot, lightTimes = spice.spkezr('PSYCHE', np.linspace(etTwoA-(2.5*secondsInAYear),etTwoA+(2.5*secondsInAYear),steps+1), 'J2000', 'NONE', 'SUN')
stateEarthPlot, lightTimes = spice.spkezr('EARTH', np.linspace(etOneD-(.5*secondsInAYear),etOneD+(.5*secondsInAYear),steps), 'J2000', 'NONE', 'SUN')
stateMarsPlot = np.asarray(stateMarsPlot)
stateEarthPlot = np.asarray(stateEarthPlot)
statePsychePlot = np.asarray(statePsychePlot)

start_times.shape = (start_times.size, 1)
mars_t_Range_ext.shape = (mars_t_Range_ext.size, 1)
end_times.shape = (end_times.size, 1)


stateEarthCombo = np.hstack((stateEarth[:,0:3]/lStar,stateEarth[:,3:6]*tStar/lStar,start_times/tStar))
stateMarsCombo = np.hstack((stateMars_ext[:,0:3]/lStar,stateMars_ext[:,3:6]*tStar/lStar,mars_t_Range_ext/tStar))
statePsycheCombo = np.hstack((statePsyche[:,0:3]/lStar,statePsyche[:,3:6]*tStar/lStar,end_times/tStar))
    

earthTab = oc.LGLInterpTable(6,stateEarthCombo,steps)
marsTab = oc.LGLInterpTable(6,stateMarsCombo,steps)
psycheTab = oc.LGLInterpTable(6,statePsycheCombo,steps)

#breakpoint()
 
earthState = np.zeros((6,))
marsState = np.zeros((6,))
#tof = .25*(etTwoA-etOneD)/tStar
for c in range(6):
    earthState[c]= np.interp(etOne,start_times[:,0],stateEarth[:,c])
    marsState[c]= np.interp(etMars,mars_t_Range_ext[:,0],stateMars_ext[:,c])
# generate Initial guess
#breakpoint()
thrustAvailablePhase = [0,1,0,0,1]
tsPhases = [etOneD,etOneD+(90*24*60*60),etMars-(60*24*60*60),etMars,etMars+(10*24*60*60),etTwoA-(100.*60*60*24),etTwoA]/tStar
IG_List = []


print('Setup inital Guesses')

ode = two_body(arg_Val,False,0)
integ= ode.integrator(0.01,noCont(),range(0,9))
ode_coast = two_body_coast(arg_Val,False,1)
integ_coast = ode_coast.integrator(0.01,noCont0(),range(0,9))

vInf_extra =4*tStar/lStar
ic_E2M_c1= np.hstack((earthState,np.array([1.5,0.,0,etOne/tStar,1.,0.,1.,0.]) ))#Mass, t_0_Front, t_0_Back, t_Ephem,Thrust_Mag, Direction
ic_E2M_c1[0:3]*= 1/lStar
ic_E2M_c1[3:6]*= tStar/lStar
ic_E2M_c1[3:6]+= vInf_extra*ic_E2M_c1[3:6]/np.linalg.norm(ic_E2M_c1[3:6])

mars_Earth_IG_c1 = integ_coast.integrate_dense(ic_E2M_c1,tsPhases[1] , 3000)
IGT_E2M_c1 = np.array(mars_Earth_IG_c1).T

ic_E2M_c2 = IGT_E2M_c1[:,-1]
mars_Earth_IG_c2 = integ.integrate_dense(ic_E2M_c2,tsPhases[2] , entries)
IGT_E2M_c2 = np.array(mars_Earth_IG_c2).T

ic_E2M_c3 = IGT_E2M_c2[:,-1]
mars_Earth_IG_c3 = integ_coast.integrate_dense(ic_E2M_c3,tsPhases[3] , 3000)
IGT_E2M_c3 = np.array(mars_Earth_IG_c3).T

ic_M2C_c4= np.hstack((marsState,np.array([1.,(etMars-etOne)/tStar,0,(etMars/tStar),1.,0.,1.,0.]) ))#Mass, t_0_Front, t_0_Back, t_Ephem,Thrust_Mag, Direction
ic_M2C_c4[0:3]*= 1/lStar
ic_M2C_c4[3:6]*= tStar/lStar

mars_Psyche_IG_c4 = integ_coast.integrate_dense(ic_M2C_c4,tsPhases[4] , 300)
IGT_M2P_c4 = np.array(mars_Psyche_IG_c4).T

ic_M2P_c5 = IGT_M2P_c4[:,-1]
mars_Psyche_IG_c5 = integ.integrate_dense(ic_M2P_c5,tsPhases[5] , entries)
IGT_M2P_c5 = np.array(mars_Psyche_IG_c5).T

ic_M2P_c6 = IGT_M2P_c5[:,-1]
mars_Psyche_IG_c6 = integ_coast.integrate_dense(ic_M2P_c6,tsPhases[6] , 300)
IGT_M2P_c6 = np.array(mars_Psyche_IG_c6).T

IG = [mars_Earth_IG_c1,mars_Earth_IG_c2,mars_Earth_IG_c3,mars_Psyche_IG_c4,mars_Psyche_IG_c5,mars_Psyche_IG_c6]
IG_plot =[IGT_E2M_c1,IGT_E2M_c2,IGT_E2M_c3,IGT_M2P_c4,IGT_M2P_c5,IGT_M2P_c6]
#breakpoint()



args_system = {'mu':132712440041.940, 'c1':tStar*tStar/(1000*lStar*mStar), 'c2':(isp*g0)*tStar/lStar,'lStar':lStar, 'mStar':mStar, 'tStar':tStar}
args_res = {'ETF':True,'nu':0.0,'Pre MGA Excess Coast':30, 'Final Duty Cycle':.5, 'Max Duty Cycle': .9,'pre MGE Steps':10,'post MGE Steps':10,'nuSteps':0,'nuStepSize':0}
#breakpoint()
args_res.update({'pre MGE Step size':args_res['Pre MGA Excess Coast']/args_res['pre MGE Steps'],'post MGE Step size':(args_res['Final Duty Cycle']-.1)/args_res['post MGE Steps']})

args_res_Trad = args_res.copy()
args_res_ETF = args_res.copy()

args_res_ETF.update({'nu':1.0,'nuSteps':10,'nuStepSize':.1})

t_Start_elapse = time.time() - np.sqrt(np.finfo(float).eps)

print('Next line is First Optimization Call')

tests=30
tempHolder=np.zeros((6,tests))
tempHolder+=-1


for c in range(tests):

    tempPerf = []
    cFlag_Traj , ocp=opt_Traj_free(IG, args_res_Trad)
    #cFlag_ETF , ocp_ETF=opt_Traj_free(IG, True,2)

    IG_redo = [ocp.Phases[0].returnTraj(), ocp.Phases[1].returnTraj(),ocp.Phases[2].returnTraj(), ocp.Phases[3].returnTraj(),ocp.Phases[4].returnTraj(),ocp.Phases[5].returnTraj()]
    
    flag=np.nan
    if(cFlag_Traj==0):
        flag=0
    elif(cFlag_Traj==1):
        flag=1
    elif(cFlag_Traj==2):
        flag=2
    elif(cFlag_Traj==3):
        flag=3

    
    tempHolder[0,c]=flag
    tempHolder[3,c]= ocp.Phases[5].returnTraj()[-1][6]

    #print(str(cFlag_Traj) + "-" + str(flag) + "-" + str(tempHolder[0,c]))
    if(tempHolder[0,c]==-1):
        breakpoint()

    if(cFlag_Traj==0 or cFlag_Traj==1):
        #break
    #print('Next line is Second Optimization Call')


        cFlag_Traj , ocp=opt_Traj_free(IG_redo, args_res_Trad)
        #cFlag_ETF , ocp_ETF=opt_Traj_free(IG_redo, args_res_ETF)

        flag=np.nan
        if(cFlag_Traj==0):
            flag=0
        elif(cFlag_Traj==1):
            flag=1
        elif(cFlag_Traj==2):
            flag=2
        elif(cFlag_Traj==3):
            flag=3
        tempHolder[1,c]=flag
        tempHolder[4,c]= ocp.Phases[5].returnTraj()[-1][6]

        #flag=np.nan
        #if(cFlag_ETF==0):
        #    flag=0
        #elif(cFlag_ETF==1):
        #    flag=1
        #elif(cFlag_ETF==2):
        #    flag=2
        #elif(cFlag_ETF==3):
        #    flag=3
        #tempHolder[2,c]=flag
        #tempHolder[5,c]= ocp_ETF.Phases[5].returnTraj()[-1][6]



    t_Curr_elapse = time.time()
    t_Elapsed = (t_Curr_elapse-t_Start_elapse)
    fractionDone = (c+1.)/tests
    runTimePerEntry = t_Elapsed/(c+1.)
    timeRemaining=(t_Elapsed/fractionDone)*(1-fractionDone)

    print('**************************')
    print("Percent Complete: {}%".format( round(100*c/tests,3)))
    print("Current Runtime Per Entry: {} (seconds)".format(round(runTimePerEntry,2)))
    
    print("Expected Total Remaining Runtime: {} (minutes)".format(round(timeRemaining/60,2)))
    
    
#traj = [ocp_ETF.Phases[0].returnTraj(), ocp_ETF.Phases[1].returnTraj(),ocp_ETF.Phases[2].returnTraj(), ocp_ETF.Phases[3].returnTraj(),ocp_ETF.Phases[4].returnTraj(),ocp_ETF.Phases[5].returnTraj()]
#traj_Base = traj.copy()

fig21a, ax21a = plt.subplots()
weights = np.ones_like(tempHolder[0,:]) / len(tempHolder[0,:])
ax21a.hist(tempHolder[0,:],weights=weights)


succ =  np.where(tempHolder[0,:]==1)[0] # np.where(np.any(tempHolder[0,:]==1))
fig21b, ax21b = plt.subplots()
weights = np.ones_like(tempHolder[1,succ]) / len(tempHolder[1,succ])
ax21b.hist(tempHolder[1,succ],weights=weights)

#plt.show()

plt.show()

breakpoint()