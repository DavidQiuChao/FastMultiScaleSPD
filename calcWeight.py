#!/usr/bin/env python 
#coding=utf-8

import cv2
import numpy as np
import boxfilter as BX



def detailWeight(im,p,r):
    R = 2*r + 1
    sps = im.shape
    h,w = sps[0],sps[1]
    tp = np.ones((h,w))
    N1 = BX.boxfilter(tp,r)

    if len(sps) == 3:
        N = sps[2]
    else:
        N = sps[3]

    C = np.zeros((h,w,N))
    i_mean2 = np.zeros((h,w,N))

    for i in range(N):
        if len(sps) == 3:
            img = im[...,i]
            i_mean2[...,i] = BX.boxfilter(img,r)/N1
            i_var2 = BX.boxfilter(img*img,r)/N1\
                    -i_mean2[...,i]*i_mean2[...,i]
            i_var2 = np.sqrt(np.maximum(i_var2,0))

            C[...,i] = i_var2*np.sqrt(R**2)+1e-12
        else:
            img = im[...,i]
            i_mean2[...,i] = BX.boxfilter(img[...,0],r)/N1\
                +BX.boxfilter(img[...,1],r)/N1\
                +BX.boxfilter(img[...,2],r)/N1
            i_mean2[...,i] /= 3

            i_var2 = BX.boxfilter(img[...,0]*img[...,0],r)/N1\
                    +BX.boxfilter(img[...,1]*img[...,1],r)/N1\
                    +BX.boxfilter(img[...,2]*img[...,2],r)/N1\
                    -i_mean2[...,i]*i_mean2[...,i]*3
            i_var2 /= 3

            i_var2 = np.sqrt(np.maximum(i_var2,0))
            C[...,i] = i_var2*np.sqrt(3*R**2)+1e-12

    Cmax = np.max(C,axis=2)

    sMap1 = C**p
    sMap2 = C**(p-1)
    sMap = sMap1 + 1e-12

    normalizer = np.sum(sMap,2)
    sMap = sMap2 / np.expand_dims(normalizer,2).repeat(N,2)
    return sMap,Cmax,i_mean2


def contrast(im):
    h = np.array([
        [0, 1, 0],
        [1,-4, 1],
        [0, 1, 0]
        ])
    H,W,N = im.shape
    C = np.zeros((H,W,N))
    for i in range(N):
        mono = im[...,i]
        C[...,i] = abs(cv2.filter2D(mono,-1,h))
    return C


def fangcha(im):
    countWindowt = np.ones((11,11))
    countWindow = countWindowt / np.sum(countWindowt)

    H,W,N = im.shape[0],im.shape[1],im.shape[2]
    C = np.zeros((H,W,N))
    for i in range(N):
        img = im[...,i]
        i_mean2 = cv2.filter2D(img,-1,contWindow) - i_mean2*i_mean2
        C[...,i] = np.sqrt(np.maximum(i_var2,0))

    return C


def gradient(im):
    H,W,N = im.shape[0],im.shape[1],im.shape[2]
    C = np.zeros((H,W,N))
    for i in range(N):
        img = im[...,i]
        gx = cv2.sobel(img,-1,1,0,ksize=5)
        gy = cv2.sobel(img,-1,0,1,ksize=5)
        C[...,i] = np.sqrt(gx**2+gy**2)
    return C


def saturation(im):
    H,W,N = im.shape[0],im.shape[1],im.shape[2]
    C = np.zeros((H,W,N))
    # saturation is computed as the standard 
    # deviation of the color channels
    for i in range(N):
        R = im[...,0,i]
        G = im[...,1,i]
        B = im[...,2,i]
        mu = (R+G+B)/3
        C[...,i] = np.sqrt(((R-mu)**2+(G-mu)**2+(B-mu)**2)/3)
    return C


def well_exposedness(im):
    H,W,N = im.shape[0],im.shape[1],im.shape[2]
    C = np.zeros((H,W,N))
    sig = 0.2

    for i in range(N):
        R = np.exp((-0.5*im[:,:,0,i]-0.5)**2/sig**2)
        G = np.exp((-0.5*im[:,:,1,i]-0.5)**2/sig**2)
        B = np.exp((-0.5*im[:,:,2,i]-0.5)**2/sig**2)
        C[...,i] = R*G*B
    return C


def  well_exposedness2(im,brightness):
    H,W,N = im.shape[0],im.shape[1],im.shape[2]
    C = np.zeros((H,W,N))

    for i in range(N):
        img = im[...,i]
        M = np.ones((H,W,N))*np.mean(img)
        C[...,i] = meanfun(img,M,brightness)
    return C


def mask(im):
    H,W,N = im.shape[0],im.shape[1],im.shape[2]
    C = np.zeros((H,W,N))
    Ct = np.zeros((H,W))

    for i in range(N):
        img = im[...,i]
        Ct[(img>=0.95)&(img<=0.05)] = 0
        C[...,i] = Ct
    return C


def meanfun(X,X_mean,index,width=None,xielv=None):
    if index =='direct':
        Y = X
        Y[Y>=0.5] = 1-Y[Y>=0.5]
    elif index == 'truncated':
        if not width:
            width = 0.8
        if not xielv:
            xielv = 0.2

        Mexp = 1-np.maximum(0,abs(X-0.5))/0.5
        Mexp = np.minimum(Mexp,width)/width
        m = 0.5
        Y = (np.actan((Mexp-m)*(np.tan(np.pi*m-xielv*m)-np.tan(-np.pi*m+xielv*m)))\
                +np.pi*m-xielv*m)/(np.pi-xielv)
    elif index == 'gussian_lg':
        lSig = 0.5
        gSig = 0.2
        Y = np.exp(-0.5*((X_mean-0.5)**2/gSig**2+(X-0.5)**2/lSig**2))
    elif index == 'gussian_l':
        lSig = 0.2
        Y = np.exp((-(X-0.5)**2)/(2*lSig**2))
    elif index == 'our':
        Z = np.zeros(X.shape)
        Z[X<=0.5] = 1
        Y5 = 2/np.pi*np.arctan(20*(X*Z))
        Y6 = 2/np.pi*np.arctan(20*(1-X)*(1-Z))
        Y = Y5+Y6
    return Y


def baseWeight(ims,m,brightness):
    r = ims.shape[0]
    c = ims.shape[1]
    N = ims.shape[2]

    W = np.ones((r,c,N))

    # compute the measure and combines them into a weight map
    contrast_parm = m[0]
    sat_parm      = m[1]
    wexp_parm     = m[2]
    fangcha_parm  = m[3]
    gradient_parm = m[4]
    wexp2_parm    = m[5]
    mask_parm     = m[6]

    if contrast_parm:
        W *= contrast(ims)**contrast_parm
    if sat_parm:
        W *= saturation(ims)**sat_parm
    if wexp_parm:
        W *= well_exposedness(ims)**wexp_parm
    if fangcha_parm:
        W *= fangcha(ims)**fangcha_parm
    if gradient_parm:
        W *= gradient(ims)**gradient_parm
    if wexp2_parm:
        W *= well_exposedness2(ims,brightness)**wexp2_parm
    if mask_parm:
        W *= mask(ims)**mask_parm


    # normalize weights: make sure that weights sum to one for each pixel
    W += 1e-12
    W /= np.expand_dims(np.sum(W,2),2).repeat(N,2)

    return W
