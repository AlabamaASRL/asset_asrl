import numpy as np
import asset as ast



vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class CR3BPFrame:
    def __init__(self, mu1,mu2,lstar):
        self.P1mu  = mu1
        self.P2mu  = mu2
        
        self.mu = mu2/(mu1+mu2)
        self.lstar = lstar
        self.tstar = np.sqrt((lstar**3)/(mu1 + mu2))
        self.vstar = lstar/self.tstar
        self.astar = (mu1 + mu2)/(lstar**2)
        self.mustar= mu1+mu2
        
        self.CalcPoints()
    
    def CalcPoints(self):
        mu = self.mu
        self.P1 = np.array([-mu,0,0])
        self.P2 = np.array([1.0-mu,0,0])
        self.L4 = np.array([.5 - mu, np.sqrt(3.0) / 2.0, 0.0])
        self.L5 = np.array([.5 - mu, -np.sqrt(3.0) / 2.0, 0.0])
        
        ###L1
        gamma0 = pow((mu*(1.0 - mu)) / 3.0, 1.0 / 3.0)
        guess = gamma0 + 1.0
        while (abs(guess - gamma0) > 10e-15):
            gamma0 = guess
            guess = pow((mu*(gamma0 - 1)*(gamma0 - 1.0)) / (3.0 - 2.0 * mu - gamma0 * (3.0 - mu - gamma0)), 1.0 / 3.0);
        self.L1 = np.array([1 - mu - guess, 0, 0]);

		###L2
        gamma0 = pow((mu*(1.0 - mu)) / 3.0, 1.0 / 3.0);
        guess = gamma0 + 1.0;
        while (abs(guess - gamma0) > 10e-15):
            gamma0 = guess;
            guess = pow((mu*(gamma0 + 1)*(gamma0 + 1.0)) / (3.0 - 2.0 * mu + gamma0 * (3.0 - mu + gamma0)), 1.0 / 3.0);
        self.L2 =  np.array([1 - mu + guess, 0, 0]);

		#### L3
        gamma0 = pow((mu*(1.0 - mu)) / 3.0, 1.0 / 3.0);
        guess = gamma0 + 1.0;
        while (abs(guess - gamma0) > 10e-15): 
            gamma0 = guess;
            guess = pow(((1.0 - mu)*(gamma0 + 1)*(gamma0 + 1.0)) / (1.0 + 2.0 * mu + gamma0 * (2.0 + mu + gamma0)), 1.0 / 3.0);  
        self.L3 =  np.array([-mu - guess, 0, 0])
        
    def GenLissajousImpl(self,func,X,xnd,znd,phideg,psideg,nplanrev,npo,t0 = 0):
        J = func.jacobian(X)
        Oxx = J[3,0]
        Oyy = J[4,1]
        Ozz = J[5,2]
        
        pi = np.pi;
        b1 = 2.0 - (Oxx + Oyy) / 2.0;
        b2sq = -Oxx * Oyy;
        s = np.sqrt(b1 + np.sqrt(b1*b1 + b2sq));
        b3 = (s*s + Oxx) / (2.0*s);
        pp = 2.0*pi / s;
        nu = np.sqrt(abs(Ozz));
        
        traj = []
        dtr = np.pi / 180.0;
        phi = phideg * dtr;
        psi = psideg * dtr;
        
        ynd = xnd*b3
        tt  = nplanrev * pp;
        dt  = tt / float(npo - 1);
        for i in range(0,npo): 
            st = np.zeros((7));
            ti  = float(i)*dt;
            st[0] = -(ynd / b3)*np.cos(s*ti + phi);
            st[1] = ynd * np.sin(s*ti + phi);
            st[2] = znd * np.sin(nu*ti + psi);
            st[3] = (ynd / b3)*s*np.sin(s*ti + phi);
            st[4] = ynd * s*np.cos(s*ti + phi);
            st[5] = znd * nu*np.cos(nu*ti + psi);
            st[6] =  t0 + ti;
            st[0:3] += X[0:3];
            traj.append(st)
        return traj;
    def GenL1Lissajous(self,xnd,znd,phideg,psideg,nplanrev,npo,t0 = 0):
        args = Args(6)
        func = self.CR3BPEOMs(args.head(3),args.tail(3))
        X = np.zeros((6))
        X[0:3] = self.L1
        return self.GenLissajousImpl(func,X,xnd,znd,phideg,psideg,nplanrev,npo,t0)
    def GenL2Lissajous(self,xnd,znd,phideg,psideg,nplanrev,npo,t0 = 0):
        args = Args(6)
        func = self.CR3BPEOMs(args.head(3),args.tail(3))
        X = np.zeros((6))
        X[0:3] = self.L2
        return self.GenLissajousImpl(func,X,xnd,znd,phideg,psideg,nplanrev,npo,t0)
        
    
    def CalcSubPoint(self,func,IG):
        X = np.zeros((func.IRows()))
        X[0:3]=IG
        while(max(abs(func.compute(X)))> 1.0e-14):
            F = func.compute(X)[3:5]
            Jt = func.jacobian(X)
            J = np.array([[Jt[3,0],Jt[3,1]],
                          [Jt[4,0],Jt[4,1]]])
            X[0:2] = X[0:2] - np.dot(np.linalg.inv(J),F)
        return np.array(X[0:3])
    
    def CalcSubPoints(self,func):
        self.SubL1 = self.CalcSubPoint(func,self.L1)
        self.SubL2 = self.CalcSubPoint(func,self.L2)
        self.SubL3 = self.CalcSubPoint(func,self.L3)
        self.SubL4 = self.CalcSubPoint(func,self.L4)
        self.SubL5 = self.CalcSubPoint(func,self.L5)
        
    def CR3BPEOMs(self,r,v,otherAccs = [],otherEOMs = []):
        x    = r[0]
        y    = r[1]
        xdot = v[0]
        ydot = v[1]
        
        t1     = vf.SumElems([ydot,x],[ 2.0,1.0])
        t2     = vf.SumElems([xdot,y],[-2.0,1.0])
        
        rterms = vf.stack([t1,t2]).padded_lower(1)
        
        g1 = r.normalized_power3(-self.P1,(self.mu-1.0))
        g2 = r.normalized_power3(-self.P2,(-self.mu))
        
        accterms   = [g1,g2,rterms] + otherAccs
        acc = vf.sum(accterms)
        terms = [v,acc] + otherEOMs
        return vf.stack(terms)
###############################################################################                
