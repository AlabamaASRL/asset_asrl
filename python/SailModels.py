import asset as ast

vf = ast.VectorFunctions
Args = vf.Arguments

class SolarSail():
    def __init__(self,beta,Ideal=False,rbar=.91,sbar=.89,Bf=.79,Bb=.67,ef=.025,eb=.27):
        self.Ideal=Ideal
        self.beta = beta
        self.rbar = rbar
        self.sbar =sbar
        self.Bf =Bf
        self.Bb =Bb
        self.ef=ef
        self.eb=eb
        print((1-self.rbar)*(self.ef*self.Bf - self.eb*self.Bb)/(self.ef+self.eb))
        self.n1 = 1 + self.rbar*self.sbar
        self.n2 = self.Bf*(1-self.sbar)*self.rbar + (1-self.rbar)*(self.ef*self.Bf - self.eb*self.Bb)/(self.ef+self.eb)
        self.t1 = 1 - self.sbar*self.rbar
        
        if(Ideal==True):self.Normalbeta = self.beta
        else:self.Normalbeta = self.beta*(self.n1+self.n2)/2.0
        
    def GetThrustExpr(self,r,n,mu):
        if(self.Ideal==True):return self.IdealSailExpr(r,n,mu)
        else :return self.MccinnesSailExpr(r,n,mu)
    def IdealSailExpr(self,r, n, mu):
        ndr2 = vf.dot(r, n)**2
        scale = self.beta*mu
        acc = scale * ndr2 * r.inverse_four_norm() * n.normalized_power3()
        return acc
    def MccinnesSailExpr(self,r,n,mu):
        
        ndr  = vf.dot(r, n)
        rn   = r.norm()*n.norm()
        ncr  = vf.cross(n,r)
        ncrn = vf.cross(ncr,n)
        
        N3DR4 = vf.dot(n.normalized_power3(),r.normalized_power4())
        sc= (self.beta*mu/2.0)
        acc = N3DR4*(((self.n1*sc)*ndr + (self.n2*sc)*rn)*n  + (self.t1*sc)*ncrn)
        
        return acc

