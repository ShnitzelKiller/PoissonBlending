import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import argparse
import cv2

def poisson_problem(image, mask, guide=None, threshold = 0.5):
    indices = np.full(mask.shape, -1)
    invdices = []
    ind = 0
    for p,m in np.ndenumerate(mask):
        if m > threshold:
            indices[p] = ind
            ind += 1
            invdices.append(p)
    N = len(invdices)
    b = np.zeros([N, image.shape[2]]) #boundary values
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
                    data.append(1)
                    I.append(i)
                    J.append(j)
                else:
                    b[i,:] = -image[(*q,)]
                if guide is not None:
                    b[i,:] -= image[p] - image[(*q,)]
    L = sp.csc_matrix((data, (I,J)), shape=(N,N))
    return L, b, invdices




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--circle', action='store_true')
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print('invalid image')
        exit()
    img = img[:,:,::-1].astype(np.float)/255

    if args.circle:
        inds = np.indices(img.shape, dtype=np.float)[0:2,:,:,0] - np.array([[[img.shape[0]//2]], [[img.shape[1]//2]]])
        mask = ((inds[0]*inds[0]+inds[1]*inds[1]) < min(*img.shape[0:2])**2/16).astype(np.float)
    else:
        mask = np.zeros(img.shape[0:2])
        mask[img.shape[0]//4:-img.shape[0]//4, img.shape[1]//4:-img.shape[1]//4] = 1

    plt.imshow(mask)
    plt.show()
    plt.imshow(img)
    plt.show()
    L, b, I = poisson_problem(img, mask, img)
    factor = linalg.factorized(L)
    xs = np.stack([factor(b[:,i]) for i in range(b.shape[1])], 1)
    img2 = np.zeros(img.shape)
    for i, p in enumerate(I):
        img2[p] = xs[i]
    mask = np.expand_dims(mask,2)
    img3 = img * (1-mask) + img2 * mask
    plt.imshow(img3)
    plt.colorbar()
    plt.show()
