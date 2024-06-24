import asset as _asset
import inspect
import numpy as np
import numbers


class ODEBase:
 
    def __init__(self,odefunc,Xvars,Uvars = None,Pvars = None,Vgroups = None):

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
                
        if(Vgroups!=None):
            self.add_Vgroups(Vgroups)
                
    
    def _make_index_set(self,idxs):
        
        if(hasattr(idxs, 'input_domain') and hasattr(idxs, 'vf') and hasattr(idxs, 'is_linear')):
            ## Handle Vector Functions
            input_domain = idxs.input_domain()
            idxstmp = []
            for i in range(0,input_domain.shape[1]):
                start = input_domain[0,i]
                size = input_domain[1,i]
                idxstmp+= list(range(start,start+size))
            return idxstmp
        elif(isinstance(idxs, (int,np.int32,np.intc))):
            return [idxs]
        elif(hasattr(idxs, '__iter__') and not isinstance(idxs, str)):
            if(len(idxs)==0):
                raise Exception("Index list is empty")
            idxtmp = []
            for idx in idxs:
                idxtmp+=self._make_index_set(idx)
            return idxtmp
        else:
            raise Exception("Invalid index: {}".format(str(idxs)))
        
    def add_Vgroups(self,Vgroups):
        for name in Vgroups:
            idxs = self._make_index_set(Vgroups[name])
            if(isinstance(name, str)):
               self.ode.add_idx(name,idxs)            
            elif(hasattr(name, '__iter__')):
                for n in name:
                    self.ode.add_idx(n,idxs)
    
    def make_units(self,**kwargs):
        """
        Makes a vector of non-dim units from variable group names and unit values

        Parameters
        ----------
        **kwargs : float,list,ndarray
            Keyword list of the names of the variable groups and corresponding units.
        Returns
        -------
        units : np.ndarray
            Vector of units.
        """
        units = np.ones((self.XtUPVars()))
        
        for key, value in kwargs.items():
            idx = self.idx(key)
            
            if(isinstance(value, numbers.Number)):
                units_t = np.ones((len(idx)))*value
            elif(hasattr(value, '__iter__') and not isinstance(value, str)):
                units_t = value
            else:
                raise Exception("Invalid unit: {}".format(str(value)))

                
            for i in range(0,len(idx)):
                units[idx[i]] = units_t[i]
        
        return units
    
    def make_input(self,**kwargs):
        """
        Makes an ODE input vector. Assigns sub-components to variable group names
        and assembles them into  single vector with proper ordering

        Parameters
        ----------
        **kwargs : float,list,ndarray
            Keyword list of the sub-vectors of the full input.

        Returns
        -------
        state : np.ndarray
            ODE input vector.

        """
        state = np.zeros((self.XtUPVars()))
        for key, value in kwargs.items():
            idx = self.idx(key)
            
            if(isinstance(value, numbers.Number)):
                state_t = np.ones((len(idx)))*value
            elif(hasattr(value, '__iter__') and not isinstance(value, str)):
                state_t = value
            else:
                raise Exception("Invalid unit: {}".format(str(value)))

            for i in range(0,len(idx)):
                state[idx[i]] = state_t[i]
        return state
    
    
    
    def get_vars(self,names,source,retscalar = False):
        """
        Retrieve variables from a ODE input vector or trajectory(2darray/listoflists)
        
        Parameters
        ----------
        names : str,list
            Name,list of names/indices of the variables or sub-vectors to retrieve.
        source : np.ndarray,list
            Source vector or 2d-array for retreiving variables.
        retscalar : bool, optional
            Return a scalar instead of vector if result has only one element. 
            The default is False.

        Returns
        -------
        np.ndarray
            Concatenated array of the specified variables from source
        """
        #############################################################
        
        idxs = []
        if(hasattr(names, '__iter__') and not isinstance(names, str)):
            for name in names:
                if(isinstance(name, str)): 
                    idxs+= list(self.idx(name))
                elif(isinstance(name, (int,np.int32,np.intc))):
                    idxs.append(name)
                else:
                    raise Exception("Invalid name specifier: {}".format(str(name)))
        elif(isinstance(names, str)):
            idxs+=list(self.idx(names))
        else:
            raise Exception("Invalid name specifier: {}".format(str(names)))
            
            
        #############################################################

        if(hasattr(source, '__iter__') and isinstance(source[0],numbers.Number)):
            # Its a single vector
            output = np.zeros(len(idxs))
            for i,idx in enumerate(idxs):
                output[i] = source[idx]
                
            if(len(output)==1 and retscalar):
                return output[0]
            else:
                return output
            
        else: 
            # Its a trajectory, nd array or list of lists/arrays
            size = len(source)
            output = np.zeros((size,len(idxs)))
            for j in range(0,size):
                for i,idx in enumerate(idxs):
                    output[j][i] = source[j][idx]
            
            return output
            
            
    
    def idx(self,Vname):
        return self.ode.idx(Vname)
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
        