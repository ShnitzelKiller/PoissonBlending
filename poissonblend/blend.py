import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

def poisson_problem(image, mask, guide=None, threshold = 0.5):
    indices = np.full(mask.shape, -1) #map of coordinates inside the domain to unique indices
    invdices = [] #list of coordinates inside the domain
    ind = 0
    for p,m in np.ndenumerate(mask):
        if m > threshold:
            indices[p] = ind
            ind += 1
            invdices.append(p)
    N = len(invdices)
    b = np.zeros([N, image.shape[2]]) #for RHS of equation

    #build sparse matrix
    data = []
    I = []
    J = []
    for i,p in enumerate(invdices):
        data.append(-4)
        I.append(i)
        J.append(i)
        for dim in (0, 1):
            for dir in (-1,1):
                q = [*p]
                q[dim] += dir
                if q[dim] < 0 or q[dim] >= mask.shape[dim]:
                    continue
                j = indices[(*q,)]
                if j > -1:
                    #contribution from inside the domain
                    data.append(1)
                    I.append(i)
                    J.append(j)
                else:
                    b[i,:] -= image[(*q,)] #boundary term (outside domain)
                if guide is not None:
                    #vector guide term
                    b[i,:] -= guide[p]
                    b[i,:] += guide[(*q,)]
    L = sp.csc_matrix((data, (I,J)), shape=(N,N))
    return L, b, invdices

def blend(image, mask, guide=None, threshold=0.5, debug=False):
    L, b, I = poisson_problem(image, mask, guide, threshold=threshold)
    factor = linalg.factorized(L)
    xs = np.stack([factor(b[:,i]) for i in range(b.shape[1])], 1)

    if debug:
        ys = np.stack([L.dot(xs[:,i]) for i in range(b.shape[1])], 1)
        res = b - ys
        res *= res
        res = np.sum(res.flat)
        print('min value:',np.min(xs.flat))
        print('max value:',np.max(xs.flat))
        print('total residual:',res)

    #composite and display results
    img2 = np.zeros(image.shape)
    for i, p in enumerate(I):
        img2[p] = xs[i]
    mask = np.expand_dims(mask,2)
    return image * (1-mask) + img2 * mask
