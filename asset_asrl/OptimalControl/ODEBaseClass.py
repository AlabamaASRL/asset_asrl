import asset as _asset
import inspect

class ODEBase:
 
    def __init__(self,odefunc,Xvars,Uvars = None,Pvars = None):

        mlist = inspect.getmembers(_asset.OptimalControl)
        
        
        hasUvars = (Uvars != None and Uvars != 0)
        hasPvars = (Pvars != None and Pvars != 0)
        name = 'ode_'+str(Xvars)

        if(hasUvars):
            name +='_'+str(Uvars)
        if(hasPvars):
            if(not hasUvars):
                name+='_0'
            name +='_'+str(Pvars)
            
        
        constsize =False
        for m in mlist:
            if(name == m[0]):
                
                self.ode = m[1].ode(odefunc)
                constsize =True
                break
        if(constsize==False):
            if(hasUvars and hasPvars):
                self.ode = _asset.OptimalControl.ode_x_u_p.ode(odefunc,Xvars,Uvars,Pvars)
            elif(hasUvars):
                self.ode = _asset.OptimalControl.ode_x_u.ode(odefunc,Xvars,Uvars)
            elif(hasPvars):
                self.ode = _asset.OptimalControl.ode_x_u_p.ode(odefunc,Xvars,0,Pvars)
            else:
                self.ode = _asset.OptimalControl.ode_x.ode(odefunc,Xvars)
                
        
        
    def phase(self,*args):
        return self.ode.phase(*args)
    def integrator(self,*args):
        return self.ode.integrator(*args)
    def vf(self):
        return self.ode.vf()
    
    
    def XVars(self):
        return self.ode.XVars()
    def UVars(self):
        return self.ode.UVars()
    def PVars(self):
        return self.ode.PVars()
    def TVar(self):
        return self.ode.TVar()
    
    def XtVars(self):
        return self.ode.XtVars()
    def XtUVars(self):
        return self.ode.XtUVars()
    def XtUPVars(self):
        return self.ode.XtUPVars()
    
    
   
    def Xidxs(self,*args):
        if(len(args)>1):
            return self.ode.Xidxs(list(args))
        else:
            return self.ode.Xidxs(*args)
        
    def Xtidxs(self,*args):
        if(len(args)>1):
            return self.ode.Xtidxs(list(args))
        else:
            return self.ode.Xtidxs(*args)
        
    def XtUidxs(self,*args):
        if(len(args)>1):
            return self.ode.XtUidxs(list(args))
        else:
            return self.ode.XtUidxs(*args)
        
    def Uidxs(self,*args):
        if(len(args)>1):
            return self.ode.Uidxs(list(args))
        else:
            return self.ode.Uidxs(*args)
        