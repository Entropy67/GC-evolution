"""
    12/09/2019, simulate single antigen extraction
"""



import numpy as np
from . import bonds as bd

PI = 3.1415926
kT = 300*1.38E-23


prm_dict = {
    
    #### simulation
    "tm": 1e7,
    "dt": 1.0,
    "record_time":1,
    "time_unit": 0.01,
    "potential": "cusp",
    
    #### force
    "scheme": "c",
    "r": 0.001,
    "f0": 10,
    "beta": 1.0,
    "tL": 100,
    "tH": 100,
    "fH": 10,
    "fL": 0,
    "tS": 1000,
    
    
    #### bonds
    "xb1": 1.5,
    "xb2": 2.0,
    "Eb1": 10,
    "Eb2": 10,
}

def printPrm(prm):
    print("prm info:")
    for item, amount in prm.items():  # dct.iteritems() in Python 2
        print("{}:\t{}".format(item, amount))


class Force_prm:
    
    def __init__(self, scheme="c"):
        self.scheme=scheme
        ### const, c
        self.f0 = 0
        
        ### ramping, r
        self.r = 0
        
        ### nonlinear ramping, nr
        self.beta = 1.0
        
        
        ## pulse, p
        self.tL = 100
        self.tH = 100
        self.fL = 0
        self.fH = 0
        
        ### sigmoid ramping, sr
        self.tS = 100
        
        ##
        self.f = 0
        
    def loadprm(self, prmdict):
        
        self.scheme=prmdict["scheme"]
        self.r = prmdict["r"]
        self.f0 = prmdict["f0"]
        self.beta = prmdict["beta"]
        self.tL = prmdict["tL"]
        self.tH = prmdict["tH"]
        self.fL = prmdict["fL"]
        self.fH = prmdict["fH"]
        self.tS = prmdict["tS"]
        pass
        
    def get_f(self, t):
        ### f is in pN
        if self.scheme=="r":
            ### ramping force
            self.f = self.r*t ### in pN
            
        elif self.scheme=="c":
            ## const force
            self.f = self.f0
            
        elif self.scheme=="p":
            ### periodic pulse
            t_eff = int(t)%int(self.tL + self.tH)
            if t_eff > self.tH:
                self.f = self.fL
            else:
                self.f = self.fH
                
        elif self.scheme=="nr":
            ## nonlinear ramping
            self.f = self.r*(t**self.beta)
            
        elif self.scheme=="sr":
            ## sigmoid ramping
            self.f = self.f0*t/(t+self.tS)
                
        return self.f


    
class System:
    
    def __init__(self, prm=prm_dict):
        
        self.prm = prm.copy()
        self.force_gen = Force_prm()
        
        self.noise = True
        self.output = False
        
        
        self.numRun = 20
        self.numSample = 200
        
        self.setup()
        
        pass
    
    def loadprm(self, prm):
        self.prm = prm
        self.setup()
        pass
    
    def setup(self):
        self.potential = self.prm["potential"]
        self.record_time = self.prm["record_time"]
        
        self.tm = self.prm["tm"]
        self.dt = self.prm["dt"]
        self.time_unit = self.prm["time_unit"]
        
        self.force_gen.loadprm(self.prm)
        
        self.bd1, self.bd2 = bd.getBonds(
            self.prm["Eb1"], 
            self.prm["Eb2"], 
            self.prm["xb1"], 
            self.prm["xb2"], 
            self.prm["potential"],
            output=False)
        
        
        self.bd1.setup()
        self.bd2.setup()
        
        if not self.noise:
            self.bd1.noiseOff()
            self.bd2.noiseOff()
            
        self.sqrtdt = np.sqrt(self.dt)
        self.gma_eff = self.bd1.gma*self.bd2.gma/(self.bd1.gma+self.bd2.gma)
        pass
    
    
    def _updateForce(self, t):
        ## return force in nN
        return self.force_gen.get_f(t)*1.0E-3
    
    
    def init(self):
        self.t = 0
        self.x1, self.x2 = 0, 0
        pass
    
    def run(self):
        return self._run(self.output) 
    
        
    def _run(self, output=True):
        count = np.zeros(3, dtype=int)
        for j in range(self.numSample):
            if output:
                printProgress(j, self.numSample)
            flag, p = self.run1(init=True)
            if flag:
                count[p] += 1
            else:
                print("simulation is not finished! please addjust tm")
        self.eta = count[2]/sum(count)
        return self.eta
        
        
        
    def run1(self, init=True):
        '''
        single run
        output: flag: break or not
                p = 0 : apc-ag-bcr
                    1 : apc-ag bcr
                    2 : apc ag-bcr
                t: tend
                f: fend
        
        '''
        if init:
            self.init()
        flag = False ## mark break or not
        step = 0
        while (self.t<self.tm):
            f = self._updateForce(self.t) ### f in nN
            self._step(f)  ### two bond movement
            flag, p = self._breakOrNot()
            if flag:
                break
            self.t += self.dt
            step += 1
        return flag, p
    
    
    
    def _step(self, f):
        ### input: f in nN
        xi1 = np.random.normal(0, self.bd1.std)
        xi2 = np.random.normal(0, self.bd2.std)
        
        fx1, fx2 = self._drift_force()
        
        #fx2 = -f
        
        dx1 = self.dt*(fx1-fx2+xi1/self.sqrtdt)/self.bd1.gma
        dx2 = self.dt*(fx2/self.gma_eff-fx1/self.bd1.gma+f/self.bd2.gma+xi2/(self.sqrtdt*self.bd2.gma)-xi1/(self.sqrtdt*self.bd1.gma))
        
        self.x1 += dx1
        self.x2 += dx2

        return

    def _drift_force(self):
        '''
        return the drift force exerting on the two bonds
        return 
            fx1: potential force generated by APC-Ag bond
            fx2: potential drift force generated by BCR-Ag bond
        '''
        unit = 1.0E18
        fx1, fx2 = 0, 0
        flag=False
        if self.potential == "cusp":
            fx1 = -self.bd1.k1*self.x1
            fx2 = -self.bd2.k1*self.x2
            flag=True
        elif self.potential =="linear-cubic":
            fx1 = (-1.5*self.bd1.Eb/(self.bd1.xb)+1.5*self.bd1.Eb*((2*self.x1-self.bd1.xb)/self.bd1.xb)**2/(self.bd1.xb))*unit
            fx2 = (-1.5*self.bd2.Eb/(self.bd2.xb)+1.5*self.bd2.Eb*((2*self.x2-self.bd2.xb)/self.bd2.xb)**2/(self.bd2.xb))*unit
            flag=True
        if not flag:
            raise Exception("no simulation, check the potential")
        return fx1, fx2
    
    def _breakOrNot(self):
        if self.bd1.broken(self.x1):
            return True, 2

        elif self.bd2.broken(self.x2):
            return True, 1
        else:
            return False, 0
    
def get_most_likely(arr, bins=30):
    """
    return the most likely value in arr
    """
    n, b = np.histogram(arr, bins=bins)
    return b[np.where(n == n.max())][0]
    
    
def printProgress(n, N):
    percent = int(100.0*n/N)
    toPrint = "progress: "
    for i in range(percent//5):
        toPrint += '|'
    toPrint += "{:d}%".format(percent)
    print(toPrint, end='\r')
    return