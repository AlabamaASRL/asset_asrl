import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments


def make_index_set(idxs):
    
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
            idxtmp+=make_index_set(idx)
        return idxtmp
    else:
        raise Exception("Invalid index: {}".format(str(idxs)))


def make_index_map(alias_dict):
    
    imap = {}
    
    for name in alias_dict:
        idxs = make_index_set(alias_dict[name])
        
        if(isinstance(name, str)):
           imap[name]= idxs
        
        elif(hasattr(name, '__iter__')):
            for n in name:
                imap[name]= idxs
    return imap

                


        


if __name__ == "__main__":

    X = Args(10)
    
    V1 = X.head(3)
    
    ind = np.array([0,1,2])
    
    Map = {}
    
    Map["V"] = [V1,X[4],None]

    print(make_index_map(Map))
    
    
    
