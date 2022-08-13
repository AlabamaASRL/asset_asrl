import numpy as np

def meshgrid2(*arrs):
        arrs = tuple(reversed(arrs))
        lens = list(map(len, arrs))
        dim = len(arrs)
        sz = 1
        for s in lens:
           sz *= s
        ans = []
        for i, arr in enumerate(arrs):
            slc = [1]*dim
            slc[i] = lens[i]
            arr2 = np.asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
            ans.append(arr2)
        return tuple(ans)


def grid_to_points(*arrs):
    g = meshgrid2(*arrs)
    k= np.vstack(map(np.ravel, g))
    return np.flip(k.T,1)
    '''
    points=[]
    for i in range(0,len(k[0])):
        p=[]
        for j in range(0,len(k)):
            p.append(k[j][i])
        points.append(p)
    return points
            '''
    