import argparse
import cv2
from blend import blend
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('--circle', action='store_true')
    parser.add_argument('--guide', nargs='?', default=0, const=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mask')
    parser.add_argument('--offset', type=int, nargs=2, default=(0,0), metavar=('Y','X'))
    parser.add_argument('--no_boundary_guide', action='store_true', help='guide gradients do not cross mask boundary')
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
            inds = np.indices(img.shape, dtype=np.float)[0:2,:,:,0] - np.array([[[img.shape[0]/2]], [[img.shape[1]/2]]])
            mask = ((inds[0]*inds[0]+inds[1]*inds[1]) < min(*img.shape[0:2])**2/16).astype(np.uint8)
        else:
            mask = np.zeros(img.shape[0:2])
            mask[img.shape[0]//4:-img.shape[0]//4, img.shape[1]//4:-img.shape[1]//4] = 1
    else:
        mask = cv2.imread(args.mask)
        if mask is None:
            print('invalid mask')
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
        print('mask range:', np.min(mask.flat), 'to', np.max(mask.flat))
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
    result = blend(img, mask, guide, offset=args.offset, debug=args.debug, boundary_guide = not args.no_boundary_guide)
    if result is not None:
        result = np.clip(result, 0, 255).astype(np.uint8)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
