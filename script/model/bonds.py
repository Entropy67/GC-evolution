## -----------------------------------
## -----------------------------------

"""
Bonds informaiton class
    File name: BrownianMotion
    Author: Hongda Jiang
    Date created: 02/26/2019
    Date last modified: 02/26/2019
    Python Version: 3.6
    Requirst package: Numpy, Matplotlib, Random
    
    
    Log:
        02/25/2019:  changed the unbinding condition to x> x1+x2/2
"""

__author__ = "Hongda Jiang"
__copyright__ = "Copyright 2019, UCLA PnA"
__license__ = "GPL"
__email__ = "hongda@physics.ucla.edu"
__status__ = "Building"

## ----------------------------------
## ----------------------------------




### import modules
import numpy as np
import matplotlib.pyplot as plt
import warnings

PI = 3.1415926
kT = 300*1.38E-23
labelSize = 16
tickSize = 14

def approxWarn():
    warnings.warn("--force too large, approximation fails--", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    approxWarn()

    
def getBonds(e1=10, e2=10, x1=3.0, x2=2.0, pot="cusp", output=True):

    apc = Bond()
    apc.x1 =x1 ## nm
    apc.e1= e1
    apc.potential=pot
    apc.setup()
    if output:
        apc.info()


    bcr = Bond()
    bcr.x1 = x2 ## nm
    bcr.e1 = e2
    bcr.potential=pot
    bcr.setup()
    if output:
        bcr.info()
    return apc, bcr


    

class Bond:
    
    def __init__(self):
        '''
        representation of a bond
        includint its potential landscape information
        '''
        self.e1 = 6 ### unbinding potential well depth in kT
        self.e2 = 2  ### binding potential barrier in kT
        
        ### we consider the bond length is fixed when changing the affinity
        self.x1 = 2.0 ### nm unbinding potential length
        self.x2 = 2.0 ### nm binding potential length
        
        ### to use spring constant as variable uncommen the following lines
        #self.k1 = 0.02 ### unbinding potential well curvature
        #self.k2 = 0.01 ### binding potential well curvature
        
        self.f0 = 0  ## external force applied, in pN
        
        self.m = 1.0 ## mass
        self.gma = 1.0 ## viscocity constant
        
        self.mg = 1.0  ### mass * gamma
        
        
        self.x = 0 ### inital position
        
        self.potential = "cusp"
        #self.continue_force = True
        
        self.setup()
        
        return
    
    def setup(self):
        self.mg = self.m*self.gma
        self.std = np.sqrt(2*self.gma*kT*1.0E18)
        self.f = self.f0*1.0E-12 ### in N
        
        self.t_off, self.t_on, self.koff, self.kon = 0, 0, 0, 0
        if self.potential == "cusp":
            self.de = self.e1-self.e2

            self.E1 = self.e1*kT  ## unbinding energy
            self.E2 = self.e2*kT  ## binding energy
            self.dE = self.E1-self.E2

            

            ## compute k1 and k2 using x1 and x2
            self.k1 = 2*self.E1*1.0E18/self.x1**2  ## unit: nN/nm
            self.k2 = 2*self.E2*1.0E18/self.x2**2  ## unit: nN/nm

            ### calculate x1 x2 using k1 and k2
            # self.x1 = np.sqrt(2*self.E1/self.k1)*1.0E9 ### bond length
            # self.x2 = np.sqrt(2*self.E2/self.k2)*1.0E9 ### bond length2
            self.xb = self.x1 + self.x2/2
            self.xd = self.x1

            self.x_unbound = self.x1+self.x2

            self.t_off = kramer(self.e1, self.k1, self.f, self.gma)
            self.t_on = kramer(self.e2, self.k2, 0, self.gma)
            self.koff = 1.0/self.t_off
            self.kon = 1.0/self.t_on
            
            self.x_break = self.x1
            
        elif self.potential == "harm2":
            
            
            self.E1 = self.e1*kT
            self.E2 = self.e2*kT
            
            
            ## compute k1 and k2 using x1 and x2
            self.k1 = 2*self.E1*1.0E18/self.x1**2  ## unit: nN/nm
            self.k2 = 2*self.E2*1.0E18/self.x2**2  ## unit: nN/nm
            
            
            self.xb = self.x1
            self.xc = self.x1 + self.x2
            self.Eb = self.e1*kT  ## unbinding energy
            
            
            
            self.wa = 4*self.Eb*1.0E18/(self.xb**2)
            self.wb = self.wa
            self.wc = self.k2*self.wb/(self.wb-self.k2)
            #print("k1, k2: ", self.k1, self.k2)
            #print("wa, wb, wc: ", self.wa, self.wb, self.wc)
            #
            self.xt1 = self.wb*self.xb/(self.wa+self.wb)
            self.xt2 = (self.wc*self.xc + self.wb*self.xb)/(self.wb + self.wc)
            #print("xt1, xt2: ",self.xt1, self.xt2)
            
            self.x_break = self.xb-self.f/self.wb
            
        elif self.potential == "linear-cubic":
            self.xb = self.x1
            self.Eb = self.e1*kT
            
            self.k1 = 6*self.Eb*1.0E18/self.xb**2
            
            self.f_drift = 3*self.Eb/(2*self.xb*1.0E-9)
            
            self.x_break = self.x1
            #self.x_break = 0.5*self.xb*(1+np.sqrt(1-self.f/self.f_drift))
            
            
        return
    
    def update_boundary(self, f):
        if self.potential == "harm2":
            self.x_break = self.xb-f/self.wb
        if self.potential == "linear-cubic":
            self.x_break = 0.5*self.xb*(1+np.sqrt(1-f/self.f_drift))
        return
    
    def noiseOff(self):
        self.std = 0
        return
    
    
    def broken(self, x):
        ### bond breaks when x>x1+x2/2
        
        if x<self.x_break:
            return False
        else:
            return True
        
    def reflect(self, x):
        ### reflective boundary condition
        if x<self.x1:
            return x
        else:
            return 2*self.x1-x
    
    def info(self):
        print("bond length: x1={0:.4f}, x2={1:.4f}".format(self.x1, self.x2))
        print("bond stiffness: k1={0:.4f}, k2={1:.4f}".format(self.k1, self.k2))
        print("energy barrier: e1={0:.1f}, e2={1:.1f}, de={2:.1f}".format(self.e1, self.e2, self.de))
        print("wait time : t_on={0:.3f}, t_off={1:.3f}".format(self.t_on, self.t_off))
        print("reaction rate: k_on={0:.4e}, k_off={1:.3e}".format(self.kon, self.koff))
        pass
    
    

        
    
class Bond2(Bond):
    
    def __init__(self):
        super().__init__()
        self.setup()
        
        self.continue_force = False
        pass
    
    def load(self, bond):
        self.e1 = bond.e1
        self.e2 = bond.e2
        self.k1 = bond.k1
        self.k2 = bond.k2
        self.f0 = bond.f0
        self.mg = bond.mg
        self.x1 = bond.x1
        self.x2 = bond.x2
        self.potential = bond.potential
        
        self.setup()
        pass
    
    def init(self):
        self.x = 0
        return
        
        
    def plotPotential(self, ax=None, fmt='-k',label=None):
        if ax==None:
            fig, ax = plt.subplots()
            plt.xlabel("$x-x_0$, nm", fontsize=labelSize)
            plt.ylabel("U, kT", fontsize=labelSize)
        #plt.ylim(-1, 6)
        if self.potential =="cusp":
            xm = self.x1+2*self.x2
        elif self.potential=="harm2":
            xm = self.x1+2*self.x2
        elif self.potential == "linear-cubic":
            xm = 1.5*self.x1
        else:
            xm = self.x1
        x = np.linspace(-0.5*self.x1, xm, 100)
        y = np.array([self.pot(xi) for xi in x])
        ax.plot(x, y, fmt, lw=2.0)
        return ax
        
    def pot(self, x):
        unit = 1.0E-18/kT
        if self.potential == "cusp":
            if x<self.x1:
                return 0.5*self.k1*x**2*unit-self.f*x*1.0E-9/kT ## convert to kT scale
            else:
                #return -100000
                return 0.5*self.k2*(x-self.x1-self.x2)**2*1.0E-18/kT+self.de-self.f*self.x1*1.0E-9/kT
        elif self.potential == "harm2":
            if x<self.xt1:
                part1 = 0.5*self.wa*x**2*unit
            elif x<self.xt2:
                part1 = -0.5*self.wb*(x-self.xb)**2*unit+self.Eb/kT
            else:
                part1 = 0.5*self.wc*(x-self.xc)**2*unit + self.E1/kT-self.E2/kT
            part2 = -self.f*x*1.0E-9/kT
            if self.broken(x):
                if self.continue_force:
                    pass
                else:
                    part2 = -self.f*self.x_break*1.0E-9/kT
            return part1 + part2
            
        elif self.potential =="linear-cubic":
            
            return 0.75*self.Eb*((2*x-self.xb)/self.xb)/kT-0.25*self.Eb*((2*x-self.xb)/self.xb)**3/kT-self.f*x*1.0E-9/kT+0.5*self.Eb/kT
        return 0
        
    
    def move(self, f, dt):
        xi = np.random.normal(0, self.std)
        dx = dt*(self.force(x)+f+xi/np.sqrt(dt))/self.mg
        self.x += dx
        return
    
    def state(self):
        if self.x<self.x1:
            return 1  ## bound state
        else:
            return 0  ## unbound state
        
    def force(self, x):
        if x<self.x1:
            return -self.k1*x
        else:
            return -self.k2*(x-self.x1-self.x2)
        
        
def kramer(e, k, f, m=1.0):
    ## f in N
    if e<0:
        raise Exception("e<0!, e="+str(e))
    tau0 = 2*m*np.sqrt(PI)/(k*np.sqrt(e))
    fs = np.sqrt(2.0*e*k*kT)
    dE=e*(1-f/fs)**2
    if 1-f/fs<0:
        approxWarn()
        #raise Exception("====== Error: energy barrier vanishes! ======")
    return tau0*np.exp(dE)/(1-f/fs)

        