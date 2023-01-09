# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:33:04 2023

@author: jwill
"""

import numpy as np 
import pywt 
from scipy.optimize import differential_evolution, Bounds
import copy
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy.stats import qmc
#global BE; global BO; global coeffsE; global coeffsO; global E; global O; global StatesE; global StatesO
def nasob4(signal):
    seed = 6
    sig = np.hstack(signal)
    sig_length = np.size(sig)
    components = 2
    wave_levels = int(np.round(np.log2(sig_length)-4)-1)
    Lam = np.zeros([wave_levels, components+1])
    e, o = Split1D(sig)
    coeffsE = pywt.wavedec(e,'db6',level=wave_levels)
    coeffsO = pywt.wavedec(o,'db6', level = wave_levels)
    scoeffsE = copy.deepcopy(coeffsE)
    scoeffsO = copy.deepcopy(coeffsO)
    #print(wave_levels,np.shape(coeffsE))
    BS, valS = SBins(sig)
    BE, BO, val = Bins(coeffsE,coeffsO)
    sBE = copy.deepcopy(BE)
    sBO = copy.deepcopy(BO)
    valG = np.sort(np.hstack(val))
    '''
    Standard for all models 
    '''
    #generate the models for each ben 
    M, StatesE, StatesO = Mod(BE, BO, components)
    fun = 0
    #mrange = 10*np.max(signal)**2
    '''
    #Step 1 Find L(j-1)...(j-N)
    Then see if hmm can improve? 
    For this method we incorporate the hmm states.
    '''
    for j in range(wave_levels-1,-1,-1):
        #args = BE, BO, coeffsE, coeffsO, e, o, j, StatesE, StatesO
        #Create the input to DE below 
        minimizer_kwargs = {"args": (BE, BO, coeffsE, coeffsO, e, o, j, StatesE, StatesO)}
        bounds0 = [[0,2*np.max(np.hstack(valG))]]
        bounds1 = [[0,2*np.max(np.hstack(valG))],[0,2*np.max(val[j])],[0,2*np.max(val[j])]]
        sampler = qmc.LatinHypercube(d=1, )
        sample = sampler.random(n=30)
        init1 = qmc.scale(sample,0,np.max(valG))
        sampler = qmc.LatinHypercube(d=3, )
        sample = sampler.random(n=10)
        init2 = qmc.scale(sample,[0,0,0],[np.max(valG),np.max(val[j]), np.max(val[j])])
        #Execute DE
        deg_result = differential_evolution(GBins, bounds0, args=(BE, BO, coeffsE, coeffsO, e, o, j),init = init1, updating = 'immediate', maxiter=10, x0=np.max(valG[j]), seed = 6)
        deh_result = differential_evolution(HMMBins, bounds1, args=(BE, BO, coeffsE, coeffsO, e, o, j, StatesE, StatesO), init = init2, updating = 'immediate', maxiter = 10, x0=[np.max(valG[j]), np.max(val[j]),np.max(val[j])], seed = 6)
        #Determine if one thresh or three thresholds (second 2 are HMM) are better for CV
        tg = deg_result['x']
        tgfun = deg_result['fun']
        th = deh_result['x']
        thfun = deh_result['fun']
        if tgfun<=thfun:
            tL = tg
            if tg >= np.max(val[j]):
                tL = [10*np.max(valG),10*np.max(valG),10*np.max(valG)]
            tfun = tgfun
            #print("L win")
        else: 
            tL = th
            if th[1] >= np.max(val[j]) and th[2] >= np.max(val[j]):
                tL = [10*np.max(valG), 10*np.max(valG), 10*np.max(valG)]
            elif th[1] >= np.max(val[j]):
                tL[1]=10*np.max(valG);
            elif th[2] >= np.max(val[j]):
                tL[2]=10*np.max(valG);
            tfun = thfun
            #print('hmm win')
        # Check to make sure new func val is better or as good otherwise previous value for Lam 
        if tfun<=fun or j == wave_levels-1:
            Lam[j] = tL
            fun = tfun
            coeffsE, coeffsO, BE, BO = newcoeffs(coeffsE, coeffsO, BE,BO,j, Lam, StatesE, StatesO)
        else:
            Lam[j] = Lam[j-1,0]
            coeffsE, coeffsO, BE, BO = newcoeffs(coeffsE, coeffsO, BE,BO,j, Lam, StatesE, StatesO)
            #print("previous L win")

    
    #Step3 Iterate through each level from current values
    for iterations in range(5):
        for j in range(wave_levels-1,-1,-1):
            minimizer_kwargs = {"args": (BE, BO, coeffsE, coeffsO, e, o, j, StatesE, StatesO)}
            bounds1 = [[0,2*np.max(Lam[j,1])],[0,2*np.max(Lam[j,2])]]
            bounds0 = [[0,2*np.max(Lam[j,0])]]
            sampler = qmc.LatinHypercube(d=1, )
            sample = sampler.random(n=30)
            init1 = qmc.scale(sample,[0],[2*np.max(Lam[j,0])])
            sampler = qmc.LatinHypercube(d=2, )
            sample = sampler.random(n=30)
            init2 = qmc.scale(sample,[0,0],[2*np.max(Lam[j,1]), 2*np.max(Lam[j,2])])
            g = 1
            deg_result = differential_evolution(HMMLBins, bounds0, args=(BE, BO, coeffsE, coeffsO, e, o, j, StatesE, StatesO, g),init = init1, updating = 'immediate', maxiter=10, seed = 6)
            g = 0
            deh_result = differential_evolution(HMMLBins, bounds1, args=(BE, BO, coeffsE, coeffsO, e, o, j, StatesE, StatesO, g), init = init2, updating = 'immediate', maxiter = 10, seed = 6)
            resfun = np.min([deg_result['fun'],deh_result['fun']])
            if resfun < fun:
                print("rep win")
                if deg_result['fun'] < deh_result['fun']:
                    tL = deg_result['x'] 
                    if tL >= np.max(val[j]):
                        tL=3*np.max(valG);  
                    Lam[j,:] = tL
                    fun = tfun
                    coeffsE, coeffsO, BE, BO = newcoeffs(coeffsE, coeffsO, BE,BO,j, Lam, StatesE, StatesO)
                else:
                    tL = deh_result['x'] 
                    fun = tfun
                    if tL[0] >= np.max(val[j]) and tL[1] >= np.max(val[j]):
                        tL = [10*np.max(valG), 10*np.max(valG)]
                    elif tL[0] >= np.max(val[j]):
                        tL[0]=10*np.max(valG);   
                    elif th[1] >= np.max(val[j]):
                        tL[1]=10*np.max(valG);
                    Lam[j,(1,2)] = tL
    #Step 4 Rebuild 
    #plt.plot(pywt.waverec(coeffsE,'db6')); plt.plot(pywt.waverec(coeffsO,'db6'))
    r, coeffs = rebuild(sig, Lam, wave_levels, M, coeffsE, components)
    return r, Lam, coeffs 
    
    # Probably need to set J-1,...1 to this thresh value no just the individual block (maybe next vers)

def HMMLBins(L,BE, BO, coeffsE, coeffsO, E, O, j, SE, SO, g):
    if g==0:
        H1 = L[0]; H2 = L[1]
    else:
        H1 = L
        H2 = L
    BEn = copy.deepcopy(BE); BOn = copy.deepcopy(BO);
    p = j
    for block in range(np.shape(BE[p])[0]):
        if SE[p][block] == 0:
            if np.sum(BEn[p][block]**2)<H1:
                BEn[p][block]=0*BEn[p][block]
        if SO[p][block] == 0:
            if np.sum(BOn[p][block]**2)<H1:
                BOn[p][block]=0*BOn[p][block]
        if SE[p][block] == 1:
            if np.sum(BE[p][block]**2)<H2:
                BEn[p][block]=0*BEn[p][block]
        if SO[p][block] == 1:
            if np.sum(BOn[p][block]**2)<H2:
                BOn[p][block]=0*BOn[p][block]
    flatE = np.hstack(BEn[p])
    flatO = np.hstack(BOn[p])
    diff = np.size(flatE)-np.size(coeffsE[p+1])
    if diff>0:
        flatE = flatE[:(np.size(flatE)-diff)]
        flatO = flatO[:(np.size(flatO)-diff)]
    coeffsE[p+1] = flatE
    coeffsO[p+1] = flatO
    rE = pywt.waverec(coeffsE,'db6'); rO = pywt.waverec(coeffsO,'db6')
    lam = .5*(np.linalg.norm(rE-O)+np.linalg.norm(rO-E))+.5*(np.linalg.norm(rE-rO))
    return lam

def HMMBins(L,BE, BO, coeffsE, coeffsO, E, O, j, SE, SO):
    Lg = L[0]; H1 = L[1]; H2 = L[2]
    BEn = copy.deepcopy(BE); BOn = copy.deepcopy(BO);
    for p in range(j,-1,-1):
        BEn = copy.deepcopy(BE); BOn = copy.deepcopy(BO);
        if p == j: 
            for block in range(np.shape(BE[p])[0]):
                if SE[p][block] == 0:
                    if np.sum(BEn[p][block]**2)<H1:
                        BEn[p][block]=0*BEn[p][block]
                if SO[p][block] == 0:
                    if np.sum(BOn[p][block]**2)<H1:
                        BOn[p][block]=0*BOn[p][block]
                if SE[p][block] == 1:
                    if np.sum(BE[p][block]**2)<H2:
                        BEn[p][block]=0*BEn[p][block]
                if SO[p][block] == 1:
                    if np.sum(BOn[p][block]**2)<H2:
                        BOn[p][block]=0*BOn[p][block]
        elif p!= j:
            for block in range(np.shape(BE[p])[0]):
                if np.sum(BEn[p][block]**2)<Lg:
                    BEn[p][block]=0*BEn[p][block]
                if np.sum(BOn[p][block]**2)<Lg:
                    BOn[p][block]=0*BOn[p][block]
        flatE = np.hstack(BEn[p])
        flatO = np.hstack(BOn[p])
        diff = np.size(flatE)-np.size(coeffsE[p+1])
        if diff>0:
            flatE = flatE[:(np.size(flatE)-diff)]
            flatO = flatO[:(np.size(flatO)-diff)]
        coeffsE[p+1] = flatE
        coeffsO[p+1] = flatO
    rE = pywt.waverec(coeffsE,'db6'); rO = pywt.waverec(coeffsO,'db6')
    lam = .5*(np.linalg.norm(rE-O)+np.linalg.norm(rO-E))+.5*(np.linalg.norm(rE-rO))
    #lam = np.sum((rE-O)**2)+np.sum((rO-E)**2)
    return lam

def GBins(L,BE, BO, coeffsE, coeffsO, E, O, j):
    Lg = L[0]; 
    BEn = copy.deepcopy(BE); BOn = copy.deepcopy(BO);
    for p in range(j,-1,-1):
        BEn = copy.deepcopy(BE); BOn = copy.deepcopy(BO);
        for block in range(np.shape(BE[p])[0]):
            if np.sum(BEn[p][block]**2)<Lg:
                BEn[p][block]=0*BEn[p][block]
            if np.sum(BOn[p][block]**2)<Lg:
                BOn[p][block]=0*BOn[p][block]
        flatE = np.hstack(BEn[p])
        flatO = np.hstack(BOn[p])
        diff = np.size(flatE)-np.size(coeffsE[p+1])
        if diff>0:
            flatE = flatE[:(np.size(flatE)-diff)]
            flatO = flatO[:(np.size(flatO)-diff)]
        coeffsE[p+1] = flatE
        coeffsO[p+1] = flatO
    rE = pywt.waverec(coeffsE,'db6'); rO = pywt.waverec(coeffsO,'db6')
    lam = .5*(np.linalg.norm(rE-O)+np.linalg.norm(rO-E))+.5*(np.linalg.norm(rE-rO))
    #lam = np.sum((rE-O)**2)+np.sum((rO-E)**2)
    return lam


def Mod(BE, BO, components): 
    ModE = []; StatesE = []; StatesO = [];
    for j in range(np.shape(BE)[0]):
        mode = hmm.GaussianHMM(n_components=2, tol=4, verbose=True)
        #mode = hmm.GaussianHMM(n_components=components,n_iter=10000)
        B = np.concatenate([BE[j],BO[j]])
        mode.fit([B[m] for m in range(np.shape(B)[0])])
        tempE = np.asarray(np.hstack(mode.predict([(BE[j][m]) for m in range(np.shape(BE[j])[0])])));
        #print(tempE)
        tempO = np.asarray(np.hstack(mode.predict([(BO[j][m]) for m in range(np.shape(BO[j])[0])])));
        ModE.append(mode)
        StatesE.append(tempE)
        StatesO.append(tempO)
    return ModE, StatesE, StatesO 
    
def rebuild(signal, lam, wave_lengths, Mod, coeffsE, components):
    coeffs = pywt.wavedec(signal,'db6', level = wave_lengths)
    for j in range(1,wave_lengths+1):
            w = int(np.log(np.size(coeffsE[j])))
            L1 = ((1-(np.log10(2)/np.log10(np.size(signal)/2**(j))))**(-1))*lam[j-1,1]
            L2 = ((1-(np.log10(2)/np.log10(np.size(signal)/2**(j))))**(-1))*lam[j-1,2]
            for block in range(0,np.size(coeffs[j]),w):
                end = np.size(coeffs[j])-(block+w)
                #print(end)
                if end < 0:
                    block = block-(w-end)
                #print(np.shape(coeffs[j][block:block+w]))
                state = Mod[j-1].predict([coeffs[j][block:block+w]])
                if state==0:
                    t = np.sum((coeffs[j][block:block+w])**2)
                    if t < L1:
                        coeffs[j][block:block+w]=0*coeffs[j][block:block+w]
                    if lam[j-1,1] == -1 :
                        coeffs[j][block:block+w]=0*coeffs[j][block:block+w]
                if state==1:
                    t = np.sum((coeffs[j][block:block+w])**2)
                    if t < L2:
                        coeffs[j][block:block+w]=0*coeffs[j][block:block+w]
                    if lam[j-1,2] == -1 :
                        coeffs[j][block:block+w]=0*coeffs[j][block:block+w]
    newsig = pywt.waverec(coeffs,'db6').T, coeffs
    for i in range(np.shape(signal)[0]):
        if newsig[0][i]<np.min(signal):
            newsig[0][i] = newsig[0][i]+(np.min(signal)-newsig[0][i])
    return newsig
        
def newcoeffs(coeffsE, coeffsO, BE,BO,j,lam, StatesE, StatesO):
    for block in range(np.shape(BE[j])[0]):
        if StatesE[j][block] == 0 and (np.sum(BE[j][block]**2) <= lam[j,1] or lam[j-1,1]==-1):
            BE[j][block] = 0*BE[j][block]
        if StatesO[j][block] == 0 and (np.sum(BO[j][block]**2) <= lam[j,1] or lam[j-1,1]==-1):
            BO[j][block] = 0*BO[j][block]
        if StatesE[j][block] == 1 and (np.sum(BE[j][block]**2) <= lam[j,2] or lam[j-1,1]==-1):
            BE[j][block] = 0*BE[j][block]
        if StatesO[j][block] == 1 and (np.sum(BO[j][block]**2) <= lam[j,2] or lam[j-1,1]==-1):
            BO[j][block] = 0*BO[j][block]
    flatE = np.hstack(BE[j])
    flatO = np.hstack(BO[j])
    diff = np.size(flatE)-np.size(coeffsE[j+1])
    if diff>0:
        flatE = flatE[:(np.size(flatE)-diff)]
        flatO = flatO[:(np.size(flatO)-diff)]
    coeffsE[j+1] = flatE
    coeffsO[j+1] = flatO
    return coeffsE, coeffsO, BE, BO


def Bins(coeffsE, coeffsO):
    BE = []; BO = []; Mod = []
    Val = []
    for j in range(1,np.shape(coeffsE)[0]):
        valE = []; valO = [];
        j_length = np.size(coeffsE[j])
        w = int((np.log(j_length)))
        BlocksE = []; BlocksO = []
        for i in range(0,j_length,w):
            if (i+w)>j_length:
                start = i-((i+w)-j_length)
                BlocksE.append(np.asarray(coeffsE[j][start:start+w]))
                BlocksO.append(np.asarray(coeffsO[j][start:start+w]))
            else:
                BlocksE.append(np.asarray(coeffsE[j][i:i+w]))
                BlocksO.append(np.asarray(coeffsO[j][i:i+w]))
            valE.append(np.sum((coeffsE[j][i:i+w])**2))
            valO.append(np.sum((coeffsO[j][i:i+w])**2))
        Val.append(np.sort(np.hstack([valE, valO])))
        BE.append(BlocksE)
        BO.append(BlocksO)
    return BE, BO, Val

def Split1D(signal):
    E = signal[::2]
    O = signal[1::2]
    return E, O;

def SBins(sig):
    B = []; Mod = []
    Val = []
    coeffs = pywt.wavedec(sig,'db6')
    for j in range(1,np.shape(coeffs)[0]):
        val = [];
        j_length = np.size(coeffs[j])
        w = int((np.log(j_length)))
        Blocks = []
        for i in range(0,j_length,w):
            if (i+w)>j_length:
                start = i-((i+w)-j_length)
                Blocks.append(np.asarray(coeffs[j][start:start+w]))
            else:
                Blocks.append(np.asarray(coeffs[j][i:i+w]))
            val.append(np.sum((coeffs[j][i:i+w])**2))
        Val.append(np.sort(np.hstack([val])))
        B.append(Blocks)
    return B, val
