import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import seaborn as sns    # pip install seaborn if you dont have it
import matplotlib.animation as animation

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

'''
Vehicle Parallel parking example derived from

http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html

who got it from

https://ieeexplore.ieee.org/document/7463491  (See case 7 in Table 2)

Goal is to parallel park a car in minumum time into a slot
that is only marginally longer that the car. ICLOS uses slightly different initial
conditions than the paper and gets a significantly longer maneuver time. Here,
we use the same initial conditions as paper and get the same answer to within
less than a percent.
'''


##################################################################
 
def CornerLoc(theta,locx,locy):
    xl = vf.cos(theta)*locx - vf.sin(theta)*locy
    yl = vf.sin(theta)*locx + vf.cos(theta)*locy
    return xl,yl

def Heavyside(x,k=10):
    return (1+vf.tanh(k*x))/2

def Fslot(x,k,SL,SW):
    return (-Heavyside(x,k) +Heavyside(x-SL,k))*SW

def Area(A,B,C):
    x1,y1 = A
    x2,y2 = B
    x3,y3 = C
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2.0



###################################################################

class CarODE(oc.ODEBase):
    
    def __init__(self,l_front,l_axes,l_rear,b_width):
        
        self.l_front = l_front
        self.l_axes  = l_axes
        self.l_rear  = l_rear
        self.b_width = b_width
        
        self.AreaRef=(l_axes+l_front+l_rear)*2*b_width;

        # Car corners in body frame, measured from center of rear axle
        self.Aloc = [l_front+l_axes,b_width]
        self.Bloc = [l_front+l_axes,-b_width]
        self.Cloc = [-l_rear,b_width]
        self.Dloc = [-l_rear,-b_width]

        self.Locs = [self.Aloc,self.Bloc,self.Cloc,self.Dloc]
        
        #####################################################
        XtU = oc.ODEArguments(6,2)
        
        x,y,v,a,theta,phi = XtU.XVec().tolist()
        
        u1,u2 = XtU.UVec().tolist()
        
        xdot = v*vf.cos(theta)
        ydot = v*vf.sin(theta)
        vdot = a
        adot = u1
        thetadot = v*vf.tan(phi)/self.l_axes
        phidot = u2
        
        ode = vf.stack([xdot,ydot,vdot,adot,thetadot,phidot])
        
        super().__init__(ode,6,2)
        #######################################################
        
    def SlotBounds(self,SL,SW,CL):
        '''
        Bounds the car to be between the far curb and the 
        lower curb/parking slot. Parking slot modeled with a tanh
        approxmiation of the heavyside step function
        '''
        
        x,y,theta,k = Args(4).tolist()
        terms = []
        for Loc in self.Locs:
            locx,locy = Loc
        
            xl,yl = CornerLoc(theta,locx,locy)  
            
            X = x + xl
            Y = y + yl
            
            eq1 = Y-CL
            eq2 = -Y+Fslot(X,k,SL,SW)
            terms.append(eq1)
            terms.append(eq2)
        
        return vf.stack(terms)
    def CornerCon(self,SL):
        '''
        Prevents the side of the car from colliding with the top corners
        of the slot. This is done by ensuring that the four traingles we can draw
        between each of the cars corners and the slot corners have a total area 
        greater than the car's planform area
        '''

        x,y,theta = Args(3).tolist()
        
        O = [0,0]
        E = [SL,0]
        
        ABCD = []
        
        for Loc in self.Locs:
              locx,locy = Loc
              xl,yl = CornerLoc(theta,locx,locy)  
              
              X = x + xl
              Y = y + yl
              
              ABCD.append([X,Y])
              
        
        A,B,C,D = ABCD
        
        AO1 = Area(O,A,B) 
        AO2 = Area(O,C,B) 
        AO3 = Area(O,A,D) 
        AO4 = Area(O,D,C) 
        
        eq1 = self.AreaRef -vf.sum([AO1,AO2,AO3,AO4]) 
        
        AE1 = Area(E,A,B) 
        AE2 = Area(E,C,B) 
        AE3 = Area(E,A,D) 
        AE4 = Area(E,D,C) 
        
        eq2 = self.AreaRef -vf.sum([AE1,AE2,AE3,AE4]) 
        
        return vf.stack(eq1,eq2)
    
    def FinalYCon(self):
        '''
        Bound all four corners of the car to be in the slot
        '''
        y,theta = Args(2).tolist()
        
        terms = []
        for Loc in self.Locs:
            locx,locy = Loc
            xl,yl = CornerLoc(theta,locx,locy)  
            Y = y + yl
            terms.append(Y)
        
        return vf.stack(terms)
    
    def CurvatureFunc(self):
        phi,u2 = Args(2).tolist()
        return u2/(self.l_axes*vf.cos(phi)**2)  

###################################################################

def MakeState(x,y,thetadeg,t):

    XtU = np.zeros(9)
    XtU[0] =x
    XtU[1] =y
    XtU[4]=np.deg2rad(thetadeg)
    XtU[6]=t
    
    return XtU


###################################################################

def PlotCorners(ode,XtU,SL,ax,col='b'):
    x = XtU[0]
    y = XtU[1]
    theta = XtU[4]
    
    mat = np.array([[np.cos(theta),-np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]])
    
    xs = []
    ys = []
    
    E = [SL,0]
    Xs =[]
    Ys =[]
    for Loc in [ode.Aloc,ode.Cloc,ode.Dloc,ode.Bloc]:
        
        xyl = np.dot(mat,Loc)
        X = x+xyl[0]
        Y = y+xyl[1]
        Xs.append(X)
        Ys.append(Y)
        xs = [E[0],X]
        ys = [E[1],Y]
                
        
    for Loc in [ode.Aloc,ode.Cloc,ode.Dloc,ode.Bloc]:
        
        xyl = np.dot(mat,Loc)
        X = x+xyl[0]
        Y = y+xyl[1]
       
        xs = [0,X]
        ys = [0,Y]
        
    
def PlotCar(ode,XtU,ax,col='b'):
    x = XtU[0]
    y = XtU[1]
    theta = XtU[4]
    phi = XtU[5]
    
    mat = np.array([[np.cos(theta),-np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]])
    
    mat2 = np.array([[np.cos(phi),-np.sin(phi)],
                    [np.sin(phi),np.cos(phi)]])
    
    xs = []
    ys = []
    
    for Loc in [ode.Aloc,ode.Cloc,ode.Dloc,ode.Bloc,ode.Aloc]:
        xyl = np.dot(mat,Loc)
        xs.append(x+xyl[0])
        ys.append(y+xyl[1])
        
        
    axlen = ode.b_width/1.5
    
    lenwhl = .3
    widwhl  =.12

    
    axles = []
    axles.append(ax.plot([],[],color='k',linestyle='solid')[0])
    axles.append(ax.plot([],[],color='k',linestyle='solid')[0])

    
    backwheels =[]
    backwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])
    backwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])

    frontwheels =[]
    frontwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])
    frontwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])
    
        
    ax.plot(xs,ys,color=col,label='Car')
    
    ######################################
    
    blcen =np.array([0,axlen])
    brcen =np.array([0,-axlen])
    
    axleloc1 = np.dot(mat,np.array([0,axlen]))
    axleloc2 = np.dot(mat,np.array([0,-axlen]))
    
    

    axles[0].set_data(np.array([axleloc1[0],axleloc2[0]])+x,np.array([axleloc1[1],axleloc2[1]])+y)
    ####################################
    
    
    wheelcens = [blcen,brcen]
    
    for i,cen in enumerate(wheelcens):
        Aloc = np.array([lenwhl,widwhl]) +cen
        Bloc = np.array([lenwhl,-widwhl]) +cen
        Cloc = np.array([-lenwhl,widwhl]) +cen
        Dloc = np.array([-lenwhl,-widwhl]) +cen
        
        
        xs = []
        ys = []
        
        for Loc in [Aloc,Cloc,Dloc,Bloc,Aloc]:
            xyl = np.dot(mat,Loc)
            xs.append(x+xyl[0])
            ys.append(y+xyl[1])
            
        backwheels[i].set_data(xs,ys)


    flcen =np.array([ode.l_axes,axlen])
    frcen =np.array([ode.l_axes,-axlen])
    
    axleloc1 = np.dot(mat,np.array([ode.l_axes,axlen]))
    axleloc2 = np.dot(mat,np.array([ode.l_axes,-axlen]))
    axles[1].set_data(np.array([axleloc1[0],axleloc2[0]])+x,np.array([axleloc1[1],axleloc2[1]])+y)

    wheelcens = [flcen,frcen]
    
    for i,cen in enumerate(wheelcens):
        Aloc = np.dot(mat2,np.array([lenwhl,widwhl])) +cen
        Bloc = np.dot(mat2,np.array([lenwhl,-widwhl])) +cen
        Cloc = np.dot(mat2,np.array([-lenwhl,widwhl])) +cen
        Dloc = np.dot(mat2,np.array([-lenwhl,-widwhl])) +cen
        
        
        xs = []
        ys = []
        
        for Loc in [Aloc,Cloc,Dloc,Bloc,Aloc]:
            xyl = np.dot(mat,Loc)
            xs.append(x+xyl[0])
            ys.append(y+xyl[1])
            
        frontwheels[i].set_data(xs,ys)
        
      
def PlotSlot(ax,k,SL,SW,CL,nxlim=-8,pxlim=8):
    
    f = Fslot(Args(1)[0],k,SL,SW)
    
    xs = np.linspace(nxlim,pxlim,100000)
    
    ys = [f([x])[0] for x in xs]
    
    ax.plot(xs,ys,color='r',label='Boundary')
    
    ax.fill_between(xs, 0, CL,alpha=.2,color='grey',label='Roadway')
    
    ax.fill_between(xs, ys, 0,alpha=.2,color='g',label='Parking Spot')

    ax.plot([nxlim,pxlim],[CL,CL],color='r')
    
def PlotTraj(ode,Traj,k,SL,SW,CL):
    n= len(Traj)
    cols=sns.color_palette("viridis",n)
    
    PlotSlot(plt,k,SL,SW,CL,-6.3,10)
    plt.legend()
    for i,X in enumerate(Traj):
       
        PlotCar(ode,X,plt,cols[i])
    plt.axis("Equal")
    plt.grid(True)

def Animate(ode,Traj,k,SL,SW,CL,save =False):
    n = len(Traj)
    fig = plt.figure()
    
    axlen = ode.b_width/1.5
    
    lenwhl = .3
    widwhl  =.12
    
    ax = fig.add_subplot(111, aspect='equal')
    
    PlotSlot(ax,k,SL,SW,CL)
    
    car, = ax.plot([],[],color='blue',label='Car')
    xy, =ax.plot([],[],color='b',linestyle='dotted')
    
    xydot, =ax.plot([],[],color='k',marker='',linestyle='',markersize=10)

    
    axles = []
    axles.append(ax.plot([],[],color='k',linestyle='solid')[0])
    axles.append(ax.plot([],[],color='k',linestyle='solid')[0])

    
    backwheels =[]
    backwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])
    backwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])

    frontwheels =[]
    frontwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])
    frontwheels.append(ax.plot([],[],color='k',linestyle='solid')[0])
    
    time_text = ax.text(-5.5, 3.1, '', fontsize=15)

    spf = Traj[-1][6]/n

    interval = int(spf*1000)
    
    def init():
        car.set_data([],[])
        xy.set_data([],[])
        xydot.set_data([],[])
        for a in axles:
            a.set_data([],[])
        
        for b in backwheels:
            b.set_data([],[])
            
        for f in frontwheels:
            f.set_data([],[])
        
        
        time_text.set_text('')

        return [car,xy,xydot,time_text]+axles +backwheels + frontwheels
    
    def animate(i):
        XtU = Traj[i]
        x = XtU[0]
        y = XtU[1]
        theta = XtU[4]
        phi = XtU[5]
        
        
        time_text.set_text(" t = {0:.2f} s".format(XtU[6]))
        
        mat = np.array([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
        
        mat2 = np.array([[np.cos(phi),-np.sin(phi)],
                        [np.sin(phi),np.cos(phi)]])
        
        xs = []
        ys = []
        
        for Loc in [ode.Aloc,ode.Cloc,ode.Dloc,ode.Bloc,ode.Aloc]:
            xyl = np.dot(mat,Loc)
            xs.append(x+xyl[0])
            ys.append(y+xyl[1])
        car.set_data(xs,ys)
        
        xis=[]
        yis=[]
        
        for j in range(0,i+1):
            xis.append(Traj[j][0])
            yis.append(Traj[j][1])
            
        xy.set_data(xis,yis)
        xydot.set_data([x],[y])

        ######################################
        
        blcen =np.array([0,axlen])
        brcen =np.array([0,-axlen])
        
        axleloc1 = np.dot(mat,np.array([0,axlen]))
        axleloc2 = np.dot(mat,np.array([0,-axlen]))
        
        

        axles[0].set_data(np.array([axleloc1[0],axleloc2[0]])+x,np.array([axleloc1[1],axleloc2[1]])+y)
        ####################################
        
        
        wheelcens = [blcen,brcen]
        
        for i,cen in enumerate(wheelcens):
            Aloc = np.array([lenwhl,widwhl]) +cen
            Bloc = np.array([lenwhl,-widwhl]) +cen
            Cloc = np.array([-lenwhl,widwhl]) +cen
            Dloc = np.array([-lenwhl,-widwhl]) +cen
            
            
            xs = []
            ys = []
            
            for Loc in [Aloc,Cloc,Dloc,Bloc,Aloc]:
                xyl = np.dot(mat,Loc)
                xs.append(x+xyl[0])
                ys.append(y+xyl[1])
                
            backwheels[i].set_data(xs,ys)


        flcen =np.array([ode.l_axes,axlen])
        frcen =np.array([ode.l_axes,-axlen])
        
        axleloc1 = np.dot(mat,np.array([ode.l_axes,axlen]))
        axleloc2 = np.dot(mat,np.array([ode.l_axes,-axlen]))
        axles[1].set_data(np.array([axleloc1[0],axleloc2[0]])+x,np.array([axleloc1[1],axleloc2[1]])+y)

        wheelcens = [flcen,frcen]
        
        for i,cen in enumerate(wheelcens):
            Aloc = np.dot(mat2,np.array([lenwhl,widwhl])) +cen
            Bloc = np.dot(mat2,np.array([lenwhl,-widwhl])) +cen
            Cloc = np.dot(mat2,np.array([-lenwhl,widwhl])) +cen
            Dloc = np.dot(mat2,np.array([-lenwhl,-widwhl])) +cen
            
            
            xs = []
            ys = []
            
            for Loc in [Aloc,Cloc,Dloc,Bloc,Aloc]:
                xyl = np.dot(mat,Loc)
                xs.append(x+xyl[0])
                ys.append(y+xyl[1])
                
            frontwheels[i].set_data(xs,ys)
            
        return [car,xy,xydot,time_text]+axles +backwheels + frontwheels
        
    ani = animation.FuncAnimation(fig, animate, frames=len(Traj),
                                  interval=interval, blit=True, init_func=init,
                                  repeat_delay=5000)
    
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([-6.1,8])
    fig.set_size_inches(15.5, 7.5, forward=True)
    fig.tight_layout()
    
    if(save):
        FFwriter = animation.FFMpegWriter( fps=30)
        ani.save('animation2.mp4', writer = FFwriter,dpi=300)
        
    
    plt.show()
    
####################################################################

def Main():

    
    SL=5.0             ## Slot length (m)
    SW=2               ## Slot width  (m)
    CL=3.5             ## Street width(m)
    
    # car dimensions (m)
    l_front=0.839
    l_axes=2.588
    l_rear=0.657
    b_width=1.771/2
    
    
    phi_max=np.deg2rad(33)  # Max steering angle (rad)
    v_max=2                 # Max velocity (m/s)
    a_max=0.75              # Max acelleration (m/s^2)
    u1_max=0.5              # Max Jerk (m/s^3)
    curvature_dot_max=0.6   # Max steering rate curvature
    
    xmin = -10
    xmax = 7.5
    
    ###############################
    
    # initial position and orientation from iee ref, all else is 0
    x0        = -5.14
    y0        = 1.41
    theta0deg = 13.18
    
    ode = CarODE(l_front,l_axes,l_rear,b_width)

    
    # Fixed Initial State
    XtU0 = MakeState(x0, y0, theta0deg, 0)
    
    ## Rougly guess 4 points of turn, 25 sec total
    XtU1 = MakeState(-0.0,y0,0,5)
    XtU2 = MakeState(5.5,y0,10,10)
    XtU3 = MakeState(1,-0.5,20,15)
    XtU4 = MakeState(1,-1,0,   25)
    
    TrajIG = [XtU0,XtU1,XtU2,XtU3,XtU4]
    
    PlotTraj(ode,TrajIG,1000,SL,SW,CL)
    plt.show()

    ###############################
    nsegs1 = 50
    nsegs2 = 200
    
    k1 = 75   # Parameter of tanh approx to heavyside slot func
    k2 = 150  # Larger value to better approx slot

    
    phase = ode.phase("LGL5",TrajIG,nsegs1,True)  ## 5 and 7 work best for this problem
    phase.setStaticParams([k1])
    phase.setControlMode("BlockConstant")
    phase.addBoundaryValue("First",range(0,7),XtU0[0:7])
    phase.addInequalCon("Path",ode.SlotBounds(SL, SW, CL),[0,1,4],[],[0])

    phase.addInequalCon("Last",ode.FinalYCon(),[1,4])
    phase.addBoundaryValue("Last",[2,3],[0,0])
    phase.addLUVarBound("Path",0,xmin,xmax)
    phase.addLUVarBound("Path",2,-v_max,v_max)
    phase.addLUVarBound("Path",3,-a_max,a_max)
    phase.addLUVarBound("Path",5,-phi_max,phi_max)
    phase.addLUVarBound("Path",7,-u1_max,u1_max)
    
    phase.addLUFuncBound("Path",ode.CurvatureFunc(),[5,8],
                         -curvature_dot_max,curvature_dot_max)
    phase.addInequalCon("Path",ode.CornerCon(SL),[0,1,4])
    
    phase.addValueLock("StaticParams",[0])  # k is locked to its initial guess
    
    phase.addDeltaTimeObjective(1)
    phase.optimizer.set_BoundFraction(.995)
    phase.optimizer.set_MaxIters(2000)
    phase.optimizer.set_PrintLevel(1)

    phase.solve_optimize()
    phase.refineTrajManual(nsegs2)
    phase.subVariable("StaticParams",0,k2)  # Change k to higher value
    phase.optimizer.set_KKTtol(1.0e-8)
    
    phase.optimize()
    
    Traj = phase.returnTraj()
    Tab  = phase.returnTrajTable()
    
    
    integ = ode.integrator(.1,Tab)
    TrajReint = integ.integrate_dense(Traj[0],Traj[-1][6],500)
    
    FinalTime = Traj[-1][6]
    
    print("ASSET Maneuver Time: ",FinalTime,' s')
    print("PAPER Maneuver Time: ",18.426,' s')

    
    # Terminal Position errors less that a millimeter
    print(Traj[-1][0:7]-TrajReint[-1][0:7])
    PlotTraj(ode,Traj,k2,SL,SW,CL)
    plt.show()
    Animate(ode,TrajReint,k2,SL,SW,CL)
    
    
if __name__ == "__main__":
    Main()
