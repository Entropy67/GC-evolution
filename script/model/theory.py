"""
theoretical calculation of antigen extraction chance

"""

import numpy as np
from scipy import integrate
import warnings

kT = 300*1.38E-23
PI = 3.1415926

"""
extraction chance under a ramping force for the cusp-harmonic potential
"""


    
def approxWarn():
    warnings.warn("--force too large, approximation fails--", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    approxWarn()

def getStiffness(xb, Eb):
    return 2*Eb*kT*1.0E18 / xb**2

def extRatio_rampongF_cuspHarmonic(E1, E2, xa, xb, r=0, f0=0, fm=30, m=1):
    k1 = 2*E1*kT*1.0E18/xa**2
    k2 = 2*E2*kT*1.0E18/xb**2

    return extRatio_rampingF_cuspHarmonic_withStiffness(E1, E2, k1, k2, r, f0, fm, m)

def meanRuptureForce(E1, E2, xa, xb, r=0, f0=0, fm=30, m=1):
    k1 = 2*E1*kT*1.0E18/xa**2
    k2 = 2*E2*kT*1.0E18/xb**2
    return meanRuptureForce_withStiffness(E1, E2, k1, k2, r, f0, fm, m)
    
def meanRuptureForce_withStiffness(E1, E2, k1, k2, r=0, f0=0, fm=80, m=1.0):
    ## force in pN
    df = 0.001
    combine = lambda f: f*(-Sca2(k1, E1, f0, f+df,r,m)*Sab2(k2, E2, f0, f+df, r, m)+Sca2(k1, E1, f0, f,r,m)*Sab2(k2, E2, f0, f, r, m))/df
    if f0 == fm or r ==0:
        ret1 = 0
    else:
        ret1 = integrate.quad(combine, f0, fm)[0]
    ret2 = fm*Sab2(k2, E2, f0, fm, r, m)*Sca2(k1, E1, f0, fm, r, m)
    return ret1 + ret2
    

def extRatio_rampingF_cuspHarmonic_withStiffness(E1, E2, k1, k2, r=0, f0=0, fm=30, m=1.0):
    '''
    calculate the extraction chance under a ramping force, in the cusp-harmonic potential
    @param
        E1, APC-Ag affinity
        E2, BCR-Ag affinity
        k1: APC-Ag stiffness, 2*E1*kT*1.0E18/xa**2, xa in nm, E1 in kT
        k2: BCR-Ag stiffness, 2*E2*kT*1.0E18/xb**2, xb in nm, E2 in kT
        r: ramping rate, pN/s
        f0: initial force, pN
        fm: cut-off force, pN
        m: ratio of damping constant
    @return:
        extraction chance
    '''
    
    
    combine = lambda f: Sab2(k2, E2, f0, f, r, m)*Sca2(k1, E1, f0, f, r, m)/(r*tauCA(f, E1, k1, m))
    if f0 == fm or r==0:
        ret1 = 0
    else:
        ret1 = integrate.quad(combine, f0, fm)[0]
    ret2 = eta0(k1,k2,E1,E2,fm, m)*Sab2(k2, E2, f0, fm, r, m)*Sca2(k1, E1, f0, fm, r, m)
    return ret1+ret2



def Sab2(k2, E2, fi0, ft0, r0, m=1.0, pot="cusp"):
    '''
    return the survival probabily for the BCR-Ag bond when force reaches ft
    @param:
        k2: BCR-Ag stiffness
        E2: BCR-Ag affinity
        fi0: initial force, in pN
        ft0: force at time t, in pN
        m: ratio of damping constant
        pot: potential function
    @return:
        survival probability at t
    '''
    
    Mg = 1.0   ## M*gma 
    ## bond parameters
    f2 = np.sqrt(2.0*E2*k2*kT)
    #print("force f2=", f2)
    fi = fi0*1.0E-12
    ft = ft0*1.0E-12
    r = r0*1.0E-12
    
    if fi0==ft0 or r0==0:
        return 1.0
    part1 = -(1+m)*np.sqrt(2*(k2**3)*kT/PI)/(4*r*Mg)
    part2 = np.exp(-E2*(1-ft/f2)**2)-np.exp(-E2*(1-fi/f2)**2)
    return np.exp(part1*part2)


def Sca2(k1, E1, fi0, ft0, r0, m=1.0, pot="cusp"):
    '''
    return the survival probabily for the APC-Ag bond when force reaches ft
    @param:
        k1: APC-Ag stiffness
        E1: APC-Ag affinity
        fi0: initial force
        ft0: force at time t
        m: ratio of damping constant
        pot: potential function
    @return:
        survival probability at t
    '''
    Mg = 1.0   ## M*gma 
    ## bond parameters
    f1 = np.sqrt(2.0*E1*k1*kT)
    
    fi = fi0*1.0E-12
    ft = ft0*1.0E-12
    r = r0*1.0E-12
    if fi0==ft0 or r0==0:
        return 1.0
    #part1 = -np.sqrt(2*k1**3*kT/PI)/(4*r*Mg)
    part1 = -np.sqrt(2*(k1**3)*kT/(PI))/(4*r*Mg)
    part2 = np.exp(-E1*(1-ft/f1)**2)-np.exp(-E1*(1-fi/f1)**2)
    return np.exp(part1*part2)


def tauCA(f0, E1, k1, m=1, cutoff=False):
    '''
    return the lifetime of APC-Ag bond under a constant force f0
    @param: 
        f0: force in pN
        E1: APC-Ag bond affinity
        k1: APC-Ag bond stiffness
    @return
        the lifetime of APC-Ag bond
    '''
    Mg = m
    f1 = np.sqrt(2*k1*E1*kT)
    f = f0*1.0E-12
    
    
    part1 = 2*Mg*np.sqrt(PI)/k1
    part2 = np.exp(E1*(1-f/f1)**2)/(np.sqrt(E1)*(1-f/f1))
        
    if cutoff and (f>f1 or E1*(1-f/f1)**2<0.5):
        return None
    return part1*part2

def eta0(k1,k2,E1,E2, f0=0, m=1):
    '''
    return the extraction chance under a constant force
    @param:
        k1: APC-Ag bond stiffness
        k2: BCR-Ag bond stiffness
        E1: APC-Ag bond affinity
        E2: BCR-Ag bond affinity
        f0: force
        m: ratio of damping constant
    @return:
        extraction chance
    '''
    f1 = np.sqrt(2.0*E1*k1*kT)
    f2 = np.sqrt(2.0*E2*k2*kT)
    f = f0*1.0E-12
    if f>f1*(1.0-1.0/np.sqrt(E1)) or f>f2*(1.0-1.0/np.sqrt(E2)):
        approxWarn()
    dE = E1*(1-f/f1)**2-E2*(1-f/f2)**2
    tmp = (1+m)*np.sqrt(k2/k1)*np.exp(dE)*(f2-f)/(f1-f)
    return 1.0/(1.0+tmp)