import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import argparse
import cv2

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import argparse

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
                    b[i,:] -= guide[p] - guide[(*q,)] #vector guide term
    L = sp.csc_matrix((data, (I,J)), shape=(N,N))
    return L, b, invdices




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('--circle', action='store_true')
    parser.add_argument('--guide', nargs='?', default=0, const=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mask')
    args = parser.parse_args()
    if args.image is None:
        img = np.zeros([256, 256, 3], dtype=np.uint8)
        img[:128,:,2] = 255
        img[129:,:,1] = 255
    else:
        img = cv2.imread(args.image)
        if img is None:
            print('invalid image')
            exit(1)

    if args.mask is None:
        if args.circle:
            inds = np.indices(img.shape, dtype=np.float)[0:2,:,:,0] - np.array([[[img.shape[0]//2]], [[img.shape[1]//2]]])
            mask = ((inds[0]*inds[0]+inds[1]*inds[1]) < min(*img.shape[0:2])**2/16).astype(np.float)
        else:
            mask = np.zeros(img.shape[0:2])
            mask[img.shape[0]//4:-img.shape[0]//4, img.shape[1]//4:-img.shape[1]//4] = 1
    else:
        mask = cv2.imread(args.mask)
        if mask is None:
            print('invalid mask')
            exit(1)
        if mask.shape[:2] != img.shape[:2]:
            print('mask and image must have same dimensions')
            exit(1)
        mask = (mask[:,:,0] > 0.5).astype(np.uint8)

    if args.guide == 0:
        guide = None
    elif args.guide == 1:
        guide = img
    else:
        guide = cv2.imread(args.guide)
        if guide is None:
            print('invalid guide')
            exit(1)

    if args.debug:
        cv2.imshow('mask', 255*np.stack([mask, mask, mask], 2).astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('original image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if args.guide != 0:
            cv2.imshow('guide image', guide)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    #solve poisson problem for each color channel (only BCs change)
    floatimg = img.astype(np.float)
    floatguide = guide.astype(np.float)
    L, b, I = poisson_problem(floatimg, mask, floatguide)
    factor = linalg.factorized(L)
    xs = np.stack([factor(b[:,i]) for i in range(b.shape[1])], 1)

    if args.debug:
        ys = np.stack([L.dot(xs[:,i]) for i in range(b.shape[1])], 1)
        res = b - ys
        res *= res
        res = np.sum(res.flat)
        print('min value:',np.min(xs.flat))
        print('max value:',np.max(xs.flat))
        print('total residual:',res)

    #composite and display results
    img2 = np.zeros(img.shape)
    for i, p in enumerate(I):
        img2[p] = xs[i]
    mask = np.expand_dims(mask,2)
    img3 = floatimg * (1-mask) + img2 * mask
    img3 = np.clip(img3, 0, 255).astype(np.uint8)
    cv2.imshow('result', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
