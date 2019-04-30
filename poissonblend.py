import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import argparse
import cv2

def laplacian(image, mask, threshold = 0.5):
    indices = np.full(mask.shape, -1)
    invdices = []
    ind = 0
    #print('length:',len(values))
    for p,m in np.ndenumerate(mask):
        if m > threshold:
            indices[p] = ind
            ind += 1
            invdices.append(p)
    #print(indices)
    N = len(invdices)
    #b = np.zeros([N, image.shape[2]]) #boundary values
    b = np.zeros(N)
    #print('index length (should be same):',len(invdices))
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
                        b[i] = -image[(*q,)]
    #print(b)
    L = sp.csc_matrix((data, (I,J)), shape=(N,N))
    #print(L)
    return L, b, invdices




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print('invalid image')
        exit()
    img = img[:,:,0].astype(np.float)

    #inds = np.indices(img.shape, dtype=np.float)[0:2,:,:,0] - np.array([[[img.shape[0]//2]], [[img.shape[1]//2]]])
    inds = np.indices(img.shape, dtype=np.float) - np.array([[[img.shape[0]//2]], [[img.shape[1]//2]]])
    mask = ((inds[0]*inds[0]+inds[1]*inds[1]) < min(*img.shape[0:2])**2/16).astype(np.float)
    plt.imshow(mask)
    plt.show()
    plt.imshow(img)
    plt.show()
    L, b, I = laplacian(img, mask)
    print(b)
    factor = linalg.factorized(L)
    #xs = np.stack([factor(b[:,i]) for i in range(b.shape[1])], 1)
    x = factor(b)
    img2 = np.zeros(img.shape)
    for i, p in enumerate(I):
        img2[p] = x[i]
    #mask = np.expand_dims(mask,2)
    img3 = img * (1-mask) + img2 * mask
    plt.imshow(img3)
    plt.colorbar()
    plt.show()
