#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import numpy as np
import imHistMatch as IHM
import boxfilter as BX



def detectMotion(ims,r):
    h,w,_,n = ims.shape
    tp = np.ones((h,w))
    N = BX.boxfilter(tp,10)
    i_mean = np.zeros((h,w,n))

    C = (0.03**2)/2
    structureThres = 0.8
    consistencyThres = 0.2
    exposureThres=0.02
    ref = int(np.ceil(n/2))
    out = np.zeros(ims.shape)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))

    R_img = ims[...,ref]
    R_mean = BX.boxfilter(R_img[...,0],r)/N\
            +BX.boxfilter(R_img[...,1],r)/N\
            +BX.boxfilter(R_img[...,2],r)/N

    R_mean /= 3

    R_var = BX.boxfilter(R_img[...,0]*R_img[...,0],r)/N\
            +BX.boxfilter(R_img[...,1]*R_img[...,1],r)/N\
            +BX.boxfilter(R_img[...,2]*R_img[...,2],r)/N\
            -R_mean*R_mean*3
    R_var /= 3
    R_std = np.sqrt(np.maximum(R_var,0))

    i_mean[...,ref] = R_mean

    muIdxMap = (R_mean<exposureThres) | (R_mean>(1-exposureThres))

    i_std = np.zeros((h,w,n))
    sRefMap = np.zeros((h,w,n))
    outs = np.zeros((h,w,3,n))

    for i in range(n):
        if i != ref:
            img = ims[...,i]

            i_mean[...,i] = BX.boxfilter(img[...,0],r)/N\
                +BX.boxfilter(img[...,1],r)/N\
                +BX.boxfilter(img[...,2],r)/N
            i_mean[...,i] /= 3

            i_var = BX.boxfilter(img[...,0]*img[...,0],r)/N\
                +BX.boxfilter(img[...,1]*img[...,1],r)/N\
                +BX.boxfilter(img[...,2]*img[...,2],r)/N \
                -i_mean[...,i]*i_mean[...,i]*3
            i_var /= 3
            i_std[...,i] = np.sqrt(np.maximum(i_var,0))

            mean_iR = BX.boxfilter(img[...,0]*R_img[...,0],r)/N\
                    +BX.boxfilter(img[...,1]*R_img[...,1],r)/N\
                    +BX.boxfilter(img[...,2]*R_img[...,2],r)/N
            mean_iR /= 3
            cov_iR = mean_iR-i_mean[...,i]*R_mean

            sMap = (cov_iR+C)/(R_std*i_std[...,i]+C)

            sRefMapt = np.zeros(sMap.shape)
            sRefMapt[sMap > structureThres] = 1
            sRefMapt[muIdxMap] = 1
            sRefMap[...,i] = cv2.morphologyEx(sRefMapt, cv2.MORPH_OPEN, se)

            cMu = IHM.imHistMatch(i_mean[...,i],R_mean)
            diff = abs(cMu - R_mean)
            cMap = np.zeros(diff.shape)
            cMap[diff<=consistencyThres] = 1

            sRefMap[...,i] = sRefMap[...,i]*cMap
            temp = IHM.imHistMatch(ims[...,ref],ims[...,i])
            out[...,i] = ims[...,i]*np.expand_dims(sRefMap[...,i],2).repeat(3,2)\
                +temp*np.expand_dims((1-sRefMap[...,i]),2).repeat(3,2)
        out[...,ref] = ims[...,ref]
    return out
