import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

def poisson_problem(image, mask, guide=None, threshold = 0.5, boundary_guide = True):
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
                    if boundary_guide or j > -1:
                        b[i,:] -= guide[p]
                        b[i,:] += guide[(*q,)]
    L = sp.csc_matrix((data, (I,J)), shape=(N,N))
    return L, b, invdices

def blend(image, mask, guide=None, offset=(0,0), threshold=0.5, boundary_guide = False, debug=False):
    if mask.shape[:2] != image.shape[:2]:
        resized = True
        ends = [offset[0]+mask.shape[0], offset[1]+mask.shape[1]]
        if ends[0] <= image.shape[0] and ends[1] <= image.shape[1] and offset[0] >= 0 and offset[1] >= 0:
            new_offsets = [0,0]
            new_ends = [d+2 for d in mask.shape]
            image_cropped = np.zeros((mask.shape[0]+2, mask.shape[1]+2, image.shape[2]), image.dtype)
            image_cropped[1:-1,1:-1] = image[offset[0]:ends[0],offset[1]:ends[1]]
            if offset[0] > 0:
                image_cropped[0,1:-1] = image[offset[0]-1,offset[1]:ends[1]]
            else:
                new_offsets[0] += 1
            if offset[1] > 0:
                image_cropped[1:-1,0] = image[offset[0]:ends[0],offset[1]-1]
            else:
                new_offsets[1] += 1
            if ends[0] < image.shape[0]:
                image_cropped[-1,1:-1] = image[ends[0],offset[1]:ends[1]]
            else:
                new_ends[0] -= 1
            if ends[1] < image.shape[1]:
                image_cropped[1:-1,-1] = image[offset[0]:ends[0],ends[1]]
            else:
                new_ends[1] -= 1
            image_cropped = image_cropped[new_offsets[0]:new_ends[0],new_offsets[1]:new_ends[1]]
            mask_padded = np.zeros(image_cropped.shape[:2], mask.dtype)
            inv_offsets = [1-o for o in new_offsets]
            inv_ends = [o+ms for o,ms in zip(inv_offsets, mask.shape)]
            mask_padded[inv_offsets[0]:inv_ends[0],inv_offsets[1]:inv_ends[1]] = mask
            if guide is not None:
                if guide.shape[0:2] != mask.shape:
                    print('guide and mask must have the same shape')
                    return
                guide_padded = np.pad(guide, ((1,1),(1,1),(0,0)), mode='edge')
                guide_padded = guide_padded[new_offsets[0]:new_ends[0],new_offsets[1]:new_ends[1]]
                guide = guide_padded
            mask = mask_padded
            image_orig = image
            image = image_cropped
        else:
            print('invalid offset or image sizes')
            return
    else:
        resized = False
    L, b, I = poisson_problem(image, mask, guide, threshold=threshold, boundary_guide = boundary_guide)
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
    mask = np.expand_dims(mask > threshold,2)
    if resized:
        image_copy = image_orig.astype(np.float)
        image_copy[offset[0]:ends[0],offset[1]:ends[1]] = (image * (1-mask) + img2 * mask)[inv_offsets[0]:inv_ends[0],inv_offsets[1]:inv_ends[1]]
        return image_copy
    else:
        return image * (1-mask) + img2 * mask
