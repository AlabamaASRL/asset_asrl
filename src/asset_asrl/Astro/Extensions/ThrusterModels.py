import asset as ast
vf = ast.VectorFunctions
Args = vf.Arguments



class LowThrustAcc():
    """
    Low thrust acceleration model
    """
    def __init__(self,NonDim_LTacc = True, LTacc=False):
        """
        Initialize desired acceleration value, if dimensional, with LTacc. NonDim_LTacc should be false.
        If non-dimensional use NonDim_LTacc. LTacc should be false. Units should be the same as outermost ODE,
        or scaled appropriately.

        Parameters
        ----------
        NonDim_LTacc : float
            Non-dimensional acceleration value.
            Set false if using LTacc.
        LTacc : float
            Dimensional acceleration value m/s^2. 
            Set false if using NonDim_LTacc
        """
        if(LTacc==False) and (NonDim_LTacc == False):
            raise ValueError("Please Specify either a dimensional or nondimensional Engine Acceleration")
        self.LTacc = LTacc
        self.NDLTacc = NonDim_LTacc

    def ThrustExpr(self,u,astar):
        """
        Acceleration expression for thruster. 

        Parameters
        ----------
        u : float
            Control magnitude. 
            Should be scaled between 0 to 1.
        astar : float
            Characteristic acceleration of top level ODE.

        Returns
        -------
        Acceleration magnitude : float
            Either dimensional or non, depending on initialization parameters.
        """
        if(self.LTacc==False):return u*(self.NDLTacc)
        else :return u*(self.LTacc/astar)

class CSIThruster():
    """
    Constant Specific Impulse Thruster model.

    Attributes
    ----------

    LTacc : float
        Acceleration magnitude
    Mdot : float
        Mass consumption rate
    M0 : float
        Initial mass
    """
    def __init__(self,F,Isp,M):
        """
        Initialize CSI model with force, specific impulse, and initial mass.
        Be sure inputs are scaled appropriately to dynamics model.

        Parameters
        ----------
        F : float
            Thruster force magntiude, dimensional
        Isp : float
            Specific impulse of thruster
        M : float
            Initial mass
        """
        self.LTacc = F/M
        self.Mdot  = F/(Isp*9.8065)
        self.M0 = M

    def ThrustExpr(self,u,m,astar):
        """
        Acceleration expression for thruster. 

        Parameters
        ----------
        u : float
            Control magnitude. 
            Should be scaled between 0 to 1.
        astar : float
            Characteristic acceleration of top level ODE.

        Returns
        -------
        Acceleration magnitude : float
            Non-dimensional acceleration
        """
        return u*(self.LTacc/astar)/m
    def MdotExpr(self,u,tstar):
        """
        Mass consumption expression 

        Parameters
        ----------
        u : float
            Control magnitude. 
            Should be scaled between 0 to 1.
        tstar : float
            Characteristic time of top level ODE.

        Returns
        -------
        Mass Consumption rate : float
            Non-dimensional mass consumption
        """
        return u.norm()*(-self.Mdot*(tstar/self.M0))

class SolarSail():
    """
    Solar sailing model. Can be used for ideal and non-ideal models.
    See Heaton, A., and Artusio-Glimpse, A., "An Update to the NASA Reference Solar Sail Thrust Model"
    for description of non-ideal sail parameters.
    """
    def __init__(self,beta,Ideal=False,rbar=.91,sbar=.89,Bf=.79,Bb=.67,ef=.025,eb=.27):
        """
        Initialize sail model. If ideal set to True, non-ideal sail coefficients are unused.

        Parameters
        ----------
        beta : float
            Sail lightness parameter
        Ideal : bool
            True if ideal sail
        rbar :  float
            Sail reflectance coefficient
        sbar : float
            Specular reflectance coefficient
        Bf : float
            Front non-Lambertion coefficient
        Bb : float
            Back non-Lambertion coefficient
        ef : float
            front emissivity coefficient
        eb : float
            back  emissivity coefficient
        """

        self.Ideal=Ideal
        self.beta = beta
        self.rbar = rbar
        self.sbar =sbar
        self.Bf =Bf
        self.Bb =Bb
        self.ef=ef
        self.eb=eb
       
        
        self.n1 = 1 + self.rbar*self.sbar
        self.n2 = self.Bf*(1-self.sbar)*self.rbar + (1-self.rbar)*(self.ef*self.Bf - self.eb*self.Bb)/(self.ef+self.eb)
        self.t1 = 1 - self.sbar*self.rbar
        
        if(Ideal==True):self.Normalbeta = self.beta
        else:self.Normalbeta = self.beta*(self.n1+self.n2)/2.0
        
    def ThrustExpr(self,r,n,mu):
        """
        Acceleration expression for solar sail.

        Parameters
        ----------
        r : ASSET VectorFunction
            3x1 solar sail position vector
        n : ASSET VectorFunction
            3x1 solar sail normal vector
        mu : float
            ODE gravitational parameter

        Returns
        -------
        Acceleration magnitude : float
            Non-dimensional acceleration
        """
        if(self.Ideal==True):return self.IdealSailExpr(r,n,mu)
        else :return self.MccinnesSailExpr(r,n,mu)

    def IdealSailExpr(self,r, n, mu):
        """
        Ideal sail acceleration expression

        Parameters
        ----------
        r : ASSET VectorFunction
            3x1 solar sail position vector from sun to sail
        n : ASSET VectorFunction
            3x1 solar sail normal vector
        mu : float
            ODE gravitational parameter

        Returns
        -------
        acc : ASSET VectorFunction
            Non-dimensional ideal sail acceleration

        """
        ndr2 = vf.dot(r, n)**2
        scale = self.beta*mu
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc
    def MccinnesSailExpr(self,r,n,mu):
        """
        Non-Ideal sail acceleration expression

        Parameters
        ----------
        r : ASSET VectorFunction
            3x1 solar sail position vector from sun to sail
        n : ASSET VectorFunction
            3x1 solar sail normal vector
        mu : float
            ODE gravitational parameter

        Returns
        -------
        acc1 : ASSET VectorFunction
            Non-dimensional non-ideal sail acceleration

        """
        ndr  = r.dot(n)
        rn   = r.norm()*n.norm()
        #ncr  = n.cross(r)
        ncrn = n.cross(r).cross(n)
        ncrn = vf.doublecross(n,r,n)
        
        N3DR4 = vf.dot(n.normalized_power3(),r.normalized_power4())
        sc= (self.beta*mu/2.0)
        acc1 = N3DR4*(((self.n1*sc)*ndr + (self.n2*sc)*rn)*n  + (self.t1*sc)*ncrn)
        
        return acc1
