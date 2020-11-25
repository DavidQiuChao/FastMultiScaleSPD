#!/usr/bin/env python 
#coding=utf-8

import cv2
import numpy as np
import calcWeight as CW
import boxfilter as BX


def scaleFine(ims,r):
    h,w,c,n = ims.shape
    tp = np.ones((h,w))
    N = BX.boxfilter(tp,r)

    tem = np.ones((h,w))
    tem[:,1:w:2] = 0
    tem[1:h:2,:] = 0
    N1 = BX.boxfilter(tem,r)

    p = 4
    WD,Cmax,i_mean2 = CW.detailWeight(ims,p,r)
    WD *= np.expand_dims(Cmax,2).repeat(n,2)

    F_temp2_detail = np.zeros((h,w,c,n))

    # approximate aggregation throught averaging(mean filter) the weight map
    i_meant = np.zeros((int(np.ceil(h/2.0)),int(np.ceil(w/2.0)),n))

    for i in range(n):
        aa = i_mean2[...,i]*tem
        try:
            i_meant[...,i] = aa[0:h:2,0:w:2]
        except:
            i_meant[...,i] = aa[0:h-1:2,0:w-1:2]
        W_D1 = BX.boxfilter(i_mean2[...,i]*WD[...,i],r)/N
        W_D2 = BX.boxfilter(WD[...,i],r)/N

        F_temp2_detail[...,i] = ims[...,i]*np.expand_dims(W_D2,2).repeat(3,2)\
                -np.expand_dims(W_D1,2).repeat(3,2)

    fI3 = np.sum(F_temp2_detail,3)

    return fI3,i_meant,aa,N1


def scaleInterm(ims,r):
    h,w,n = ims.shape
    tp = np.ones((h,w))
    N = BX.boxfilter(tp,r)

    tem = np.ones((h,w))
    tem[:,1:w:2] = 0
    tem[1:h:2,:] = 0
    N1 = BX.boxfilter(tem,r)

    p = 4

    WD,Cmax,i_mean2 = CW.detailWeight(ims,p,r)
    WD *= np.expand_dims(Cmax,2).repeat(n,2)

    F_temp2_detail = np.zeros((h,w,n))

    # approximate aggregation through averaging(mean filter) the weight map

    i_meant = np.zeros((int(np.ceil(h/2.0)),int(np.ceil(w/2.0)),n))

    for i in range(n):
        aa = i_mean2[...,i]*tem
        try:
            i_meant[...,i] = aa[0:h:2,0:w:2]
        except:
            i_meant[...,i] = aa[0:h-1:2,0:w-1:2]
        W_D1 = BX.boxfilter(i_mean2[...,i]*WD[...,i],r)/N
        W_D2 = BX.boxfilter(WD[...,i],r)/N
        F_temp2_detail[...,i] = W_D2*ims[...,i] - W_D1

    fI3 = np.sum(F_temp2_detail,2)
    return fI3,i_meant,aa,N1


def scaleCoarse(ims,r):
    h,w,n = ims.shape
    tp = np.ones((h,w))
    N = BX.boxfilter(tp,r)

    tem = np.ones((h,w))
    tem[:,1:w:2] = 0
    tem[1:h:2,:] = 0
    N1 = BX.boxfilter(tem,r)

    p = 4

    WD,Cmax,i_mean2 = CW.detailWeight(ims,p,r)
    WD *= np.expand_dims(Cmax,2).repeat(n,2)

    brightness = 'our'

    WB = CW.baseWeight(ims,[1,0,0,0,0,1,0],brightness)

    F_temp2_detail = np.zeros((h,w,n))
    F_temp2_base = np.zeros((h,w,n))
    F_temp2 = np.zeros((h,w,n))

    # approximate aggregation through averaging(mean filter) the weight map
    i_meant = np.ones((int(np.ceil(h/2.0)),int(np.ceil(w/2.0)),n))

    for i in range(n):
        aa = i_mean2[...,i]*tem
        try:
            i_meant[...,i] = aa[0:h:2,0:w:2]
        except:
            i_meant[...,i] = aa[0:h-1:2,0:w-1:2]

        W_D1 = BX.boxfilter(i_mean2[...,i]*WD[...,i],r)/N
        W_D2 = BX.boxfilter(WD[...,i],r)/N
        W_B2 = BX.boxfilter(i_mean2[...,i]*WB[...,i],r)/N

        F_temp2_detail[...,i] = W_D2*ims[...,i] - W_D1
        F_temp2_base[...,i] = W_B2
        F_temp2[...,i] = F_temp2_detail[...,i]+F_temp2_base[...,i]

    fI3 = np.sum(F_temp2,2)
    return fI3,i_meant,aa,N1
