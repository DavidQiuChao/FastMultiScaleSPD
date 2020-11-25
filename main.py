#!/usr/bin/env python
#cdoing=utf-8

import os
import cv2
import time
import numpy as np
import deGost as DG
import scaleCalc as SC
import boxfilter as BX



def loadImgSeq(seqDir):
    names = os.listdir(seqDir)
    names.sort()
    names[0],names[1],names[2] = names[2],names[0],names[1]

    for idx, name in enumerate(names):
        if name[-3:]!='jpg' and name[-3:]!='png':
            continue
        impath = os.path.join(seqDir,name)
        im = cv2.imread(impath)
        im = im*1.0/255
        if idx == 0:
            ims =  np.expand_dims(im,3)
        else:
            im = np.expand_dims(im,3)
            ims = np.concatenate((ims,im),axis=3)
    return ims


def downSample2(im,r=2):
    _H,_W,C,N = im.shape
    H = int(_H/r)
    W = int(_W/r)
    rim = np.zeros((H,W,C,N))
    for i in range(N):
        rim[...,i] = cv2.resize(im[...,i],(W,H))
    return rim,_H,_W

def upSample2(im,H,W):
    return cv2.resize(im,(W,H))


def main(args):
    seqDir = args.seqDir
    ds = args.dsr

    # load image and down sample
    ims = loadImgSeq(seqDir)
    if ds>1:
        ims,H,W = downSample2(ims,ds)

    st = time.time()
    # motion detection
    r = 10
    ims = DG.detectMotion(ims,r)
    en1 = time.time()
    print('Deghost cost %f'%(en1-st))

    # fusion as static
    r1 = 4
    D1,i_mean1,aa1,N1 = SC.scaleFine(ims,r1)

    # the intermediate scale
    h,w,c,n = ims.shape
    nlev = int(np.floor(np.log(min(h,w))/np.log(2))-5)

    D2  = [[] for i in range(nlev)]
    aa2 = [[] for i in range(nlev)]
    N2  = [[] for i in range(nlev)]

    r2 = 4
    for j in range(nlev):
        D2[j],i_mean2,aa2[j],N2[j] = SC.scaleInterm(i_mean1,r2)
        i_mean1 = i_mean2

    # the coarsest scale
    r3 = 4
    fI3,i_mean3,aa3,N3 = SC.scaleCoarse(i_mean2,r3)

    # reconstruct
    # intermediate layers
    for j in range(nlev-1,-1,-1):
        temp = aa2[j]
        fI = np.zeros(temp.shape)
        _h,_w = temp.shape[0],temp.shape[1]
        fI[0:_h:2,0:_w:2] = fI3
        B2 = BX.boxfilter(fI,r2)/N2[j]+D2[j]
        fI3 = B2

    # finest layer
    fI = np.zeros((aa1.shape))
    _h,_w = aa1.shape[0],aa1.shape[1]
    fI[0:_h:2,0:_w:2] = B2
    B1 = BX.boxfilter(fI,r1)/N1
    C_out = np.expand_dims(B1,2).repeat(3,2)+D1
    C_out = np.minimum(1,np.maximum(0,C_out))
    C_out *= 255

    if ds>1:
        C_out = upSample2(C_out,H,W)

    ed = time.time()
    print('fusion cost: %fs'%(ed-st))

    cv2.imwrite('tmp.jpg',C_out)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--seqDir',help='input '\
            'sequence dir')
    parser.add_argument('-d','--dsr',help='down '\
            'sample rate',default=1,type=int)
    args = parser.parse_args()
    main(args)
