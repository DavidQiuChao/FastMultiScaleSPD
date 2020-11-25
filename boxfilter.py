#!/usr/bin/env python
#coding=utf-8

import numpy as np


def boxfilter(imSrc,r):
    h,w = imSrc.shape
    imDst = np.zeros(imSrc.shape)

    imCum = np.cumsum(imSrc,0)
    imDst[0:r+1,:] = imCum[r:2*r+1,:]
    imDst[r+1:h-r,:] = imCum[2*r+1:h,:]-imCum[0:h-2*r-1,:]
    imDst[h-r:h,:] = np.expand_dims(imCum[h-1,:],0).repeat(r,0)\
            -imCum[h-2*r-1:h-r-1,:]

    imCum = np.cumsum(imDst,1)
    imDst[:,0:r+1] = imCum[:,r:2*r+1]
    imDst[:,r+1:w-r] = imCum[:,2*r+1:w]-imCum[:,0:w-2*r-1]
    imDst[:,w-r:w] = np.expand_dims(imCum[:,w-1],1).repeat(r,1)\
            -imCum[:,w-2*r-1:w-r-1]
    return imDst
