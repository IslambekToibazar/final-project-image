

import cv2
import numpy as np
import argparse
import ntpath
import os


def adaptative_thresholding(path, threshold):
   
    I = cv2.imread(path)

    
    grey = cv2.cvtColor(I, cv2.COLOR_BGR2grey)

    
    orignrows, origncols = grey.shape

    
    M = int(np.floor(orignrows / 16) + 1)
    N = int(np.floor(origncols / 16) + 1)

    
    Mextend = round(M / 2) - 1
    Nextend = round(N / 2) - 1

    
    aux = cv2.copyMakeBorder(grey, top=Mextend, bottom=Mextend, left=Nextend,
                             right=Nextend, borderType=cv2.BORDER_REFLECT)

    window = np.zeros((M, N), np.int32)

    
    imageIntegral = cv2.integral(aux, window, -1)

   
    nrows, ncols = imageIntegral.shape

    
    result = np.zeros((orignrows, origncols))

    
    for i in range(nrows - M):
        for j in range(ncols - N):
            result[i, j] = imageIntegral[i + M, j + N] - imageIntegral[i, j + N] + imageIntegral[i, j] - imageIntegral[
                i + M, j]


    binar = np.ones((orignrows, origncols), dtype=np.bool)


    greymult = (grey).astype('float64') * M * N


    binar[greymult <= result * (100.0 - threshold) / 100.0] = False


    binar = (255 * binar).astype(np.uint8)

    return binar


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='applies adaptative binarization and saves output.')
    parser.add_argument('-i', '--input_path', dest="input_path", type=str, required=True, help="image path")
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, default=25, help="binarization threshold")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise IOError('input file does not exit')

    if not 0 < args.threshold < 100:
        raise IOError('threshold must be between 0 and 100')

    output = adaptative_thresholding(args.input_path, args.threshold)

    nameOfImage = ntpath.basename(args.input_path)

    cv2.imwrite(os.path.splitext(nameOfImage)[0] + '_bin' + os.path.splitext(nameOfImage)[1], output)