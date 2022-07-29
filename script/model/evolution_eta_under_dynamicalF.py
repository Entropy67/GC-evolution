import numpy as np
import matplotlib.pyplot as plt
#import random as rd
from time import gmtime, strftime
from collections import defaultdict
from scipy import integrate
import warnings
import os
from mpmath import mp
from termcolor import colored
from . import singleBond as sb
from . import theory as theory
import json
import heapq
from datetime import datetime
from copy import deepcopy
PI = 3.1415926
kT = 300*1.38E-23    # temperature

USE_DIVISION = False ## use w as number of division instead of children num

TWO_MUTATION = False ## TWO mutation happen simultenously

FEEDBACK_XB = False ### whether the feedback antibody bond length determines the tether bond length

INFINITE_PLASMA = True ### infinite plasma cell capacity

def info():
    print("use division or not: ", USE_DIVISION)
    print("xb mutation and Gb mutation happen simulteneously? : ", TWO_MUTATION)
    print("antibody bond length xb feedback to tether bond length xa ? : ", FEEDBACK_XB)
    print("the plasma cell capacity is infinite ? : ", INFINITE_PLASMA)
    return
    
    
def approxWarn():
    warnings.warn("--force too large, approximation fails--", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    approxWarn()
    
    
"""
Parameter choosing:
    one GC cycle: 12 hour [Christopher et al. Science (2007)]
    GC reaction: 4 weeks, or 4*7*24h = 672h
    Total num of cycles: 60 cycles
    Binding constant change: 1000 fold, or 7kT
    Assume naive B cell affinity: 12kT
    Maturated B cell affinity: 21kT
    
    Initial GC size: 1000 naive B cells
    GC capacity: 2000, [Jacob et al., 1991, Wang et al. 2015]
    
    11/14/2021
    change the rule of Ab update
    change to linear-cubic potential
    
    11/15/2021
    use increasing force: different generations use different forces. 
    
    11/20/2021
    change the max replication rate to 2^6
    change the death rate
    
    03/25/2022
    add dynamical force
    
"""

week_per_cycle = 0.5/7

prm_list = {
    
    "feedback": False, ## ab renewable or not
    "output": True, ### write out info and warn
    "randomAb":True,
    "useWeek":False,
    "useSim":True,
    "debug":False,
    "useBell":False,
    "goodDiff":False,
    "useBinomial":False,
    "dump_eta": False,
    "no_selection": False,
    "dynamical_force": False, ## use dynamical force to extract antigen
    
    "death": "random",
    
    "update_rule": "topK", #### "all, ", "random", "topK"
    
    "potential": "linear-cubic", ## "cusp" or "linear-cubic" or "bell"
    
    "cag":100, ## antigen concentration
    
    "Eb": 14, ## naive B cell affinity
    "Ea": 14, ## tether affinity
    
    "xb1": 1.5,   ## APC-Ag bond stiffness
    "xb2": 2.0,   ## BCR-Ag bond stiffness
    
    ### plasma cell property
    "Td": 0, ### delay betwin diff to Ab feedback
    "Tp": 1000, ## plasma cell lifetime, 0 do not die
    
    "cutoff": 200, ## cycle cutoff to calculate the average maturation rate 
    
    "w0": 8,  ### max number of divisions in each cycle
    "pd": 0.05, ### B cell differentiation rate
    "pa": 0.7,  ### 1 - B cell  death rate
    "pm": 0.5, #### B cell mutation rate
    
    "pm_xb": 0.5, ### probability of bond length mutation
    
    "Nc": 2000,  ### GC carrying capacity
    "Npc": 5000, ### plasma cell capacity
    
    "N0": 1000, ## initial B cell number
    "Nab": 100, ### Ab pool size
    
    "dE": 0.1, ### kT, mutation step
    "dxb": 0.1, ### nm, mutation in bond length
    
    "f":0,  # const force
    "df": 0, ## force increasing step size per GC cycle
    "eta0": 0.5, 
    "tm": 300,
    "r": 0.0001, ### ramping rate when extracting antigens
    
    "Eb_min": 2.0, ### minimal barrier height that analytical expression apply, the higher the more accurate,
    "eta_file": "fixedEb",
}




################################
## get statistical property
class manyRun:
    def __init__(self, prm=prm_list):
        ## control penal
        self.prm = deepcopy(prm)
        
        self.output = True
        self.fix_success = False
        self.save_traj = True
        
        self.traj_qty_name = ["e",
                              "eStd",
                             
                              "tau", 
                              "tauStd", 
                             
                              "xb", 
                              "xbStd", 
                             
                              "eab", 
                              "eabStd",
                             
                              "xb_ab", 
                              "xb_abStd",
                         
                              "tau_ab", 
                              "tau_abStd",
                         
                              "ep", 
                              "epStd", 
                              
                              
                              "Ec",
                         
                              "tau_p", 
                              "tau_pStd",
                         
                              "n", 
                              "np", 
                              #"nab",
                         
                              "w", 
                              "wStd",
                         
                              "eta", 
                              "etaStd"
                             ]
        
        self.saved_qty = ["e", ### affinity at tmax
                          "eStd", ### affinity std
                          "xb", ### bond length at tmax
                          "xbStd", ### bond length std
                          "tau",  ### intrinsic lifetime
                          "tauStd",  ### lifetime std
                          "eab",  ### ab pool mean affinity
                          "eabStd", ### ab pool affinity std
                          "tau_ab", ### ab pool intrinsic lifetime
                          "n",  ### b cell pop size
                          "np", ### plasma cell pop size
                          "w",  ### fitness average
                          "wStd", ### fitness std
                          "eta",  ### average eta
                          "etaStd",
                          "tau0", ### initial lifetime] 
                          "mat_rate", ### maturation rate
                          "surv_prob", ### survival probability
                          "tend", ### ending time 
                          "Ec", 
                          "de",
                          "Nb"
                         ]
        self.trajdata = {}
        
        self.dataset = {}
        self.eta_record = {}
        self.mean_dataset = {}
        self.setup()
        self.sample_rate = 5
        self.num_run = 20
        return
    
    def set_prm(self, qty, value):
        self.prm[qty] = value
        self.setup()
        return 
    
    def setup(self):
        self.tm = self.prm["tm"]
        self.f=self.prm["f"]
        self.eta_record = {}
        self.gc = GC(f0=self.f, prm=self.prm)
        self.gc.output=False
        self.surv_prob = None
        if self.save_traj:
            for qty in self.traj_qty_name:
                self.trajdata[qty] = []
            
        for qty in self.saved_qty:
            self.dataset[qty] = []
        pass
    

    
    
    def dump_all(self, filename=""):
        dump(self.dataset, filename + "_dataset")
        dump(self.trajdata, filename + "_trajdata")
        return
        
    
    def load_eta_record(self):
        self.gc.load_eta_record()
        self.eta_record = self.gc.eta_record
        return
    
    def dump_eta_record(self):
        self.gc.eta_record = self.eta_record
        self.gc.dump_eta_record()
        return
    
    def run(self, output=False):
        
        self.setup()
        
        fix_success=self.fix_success
        success_count = 0
        dead_count = 0
        tot_count = 0

        mat_rate, var_tmp, eta_tmp = 0, 0, 0
        if self.output and output:
            print("manyRun: check parameter: f=%f" % self.prm["f"], ", xa= %f" % self.prm["xb1"])
        while success_count < self.num_run:
            if output and self.output:
                print(">>>> starting GC {0:d}, last run: mr= {1:.3f}, std={2:.3f}, eta={3:.3f}, tm={4:.1f}, success={5:d}".format(tot_count, mat_rate, var_tmp, eta_tmp, self.gc.t, success_count))
            
            self.gc.setup()
            self.gc.eta_record = self.eta_record
            self.gc.output = (output and self.output)
            success = self.gc.run(tm=self.tm)
            self.eta_record = self.gc.eta_record
            tot_count += 1
            
            if self.save_traj:
                self.append_traj()
            self.append_qty()
            
            self.dataset["tend"].append(self.gc.t)
            if not success:
                dead_count += 1
                if tot_count>2000:
                    if success_count == 0:
                        print("WARNING:---- All GC died!")
                        self.surv_prob = 0
                        return False
                    else:
                        break
            else:  
                success_count += 1
                var_tmp = np.mean(self.gc.dataset["eStd"][-20:])
                mat_rate = np.mean(diff(self.gc.dataset["e"])[-20:])
                eta_tmp = np.mean(self.gc.dataset["eta"][-20:])
            
            if (not fix_success) and tot_count>self.num_run:
                break
        self.surv_prob = success_count/(success_count+dead_count)
        self.dataset["surv_prob"] = [self.surv_prob]
        if output and self.output:
            print(colored("finished! success={0:d}, dead={1:d}, percen={2:.3f}\t".format(success_count, dead_count, self.surv_prob), "grey"))
        if success_count == 0:
            print("WARNING: ----- All GC died!")
            return False
        else:
            if self.output:
                print("many run: summary: surv_prob = {2:.4f}\t, Eb = {3:.3f}{5:.3f}\t, tau={4:.3f}".format(success_count, dead_count, self.surv_prob, np.mean(self.dataset["e"]), np.std(self.dataset["e"]), np.log10(np.mean(self.dataset["tau"])/self.gc.dataset["tau"][0])))
        
        #self.get_mean()
        return True
    
    def append_qty(self):
        for qty in self.saved_qty:
            if qty in self.gc.dataset.keys() and self.gc.alive:
                self.dataset[qty].append(self.gc.dataset[qty][-1])
        
    def append_traj(self):
        for qty in self.traj_qty_name:
            traj_tmp = self.gc.dataset[qty]
            self.trajdata[qty].append(traj_tmp[::self.sample_rate])
            ### include the last one
            if len(traj_tmp) % self.sample_rate != 1:
                self.trajdata[qty][-1].append(traj_tmp[-1])
        return
    
    def append(self, qty, value):
        self.dataset[qty].append(value)
        pass
    
    def get_data(self):
        return self.dataset
    
    def get_mean_traj(self, qty):
        return np.mean(self.trajdata[qty], axis=0)
    
    def plotQty(self, qty, cr="r", label="", ax=None, filling=False, multiDim=True, **keyargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(4,3), dpi=100)
            ax.set_xlabel("time", fontsize=12)
            ax.set_ylabel(qty, fontsize=12)
        
        if multiDim:
            mean = np.mean(self.trajdata[qty], axis=0)
            std = np.std(self.trajdata[qty], axis=0)
            for i in range(len(mean)):
                if mean[i] is None:
                    continue
                start_id = i
                break
        else:
            mean = self.trajdata[qty]
            
        if self.prm["useWeek"]:
            tlist = convert_to_week(range(len(mean)))
        else:
            tlist = range(len(mean))
        ax.plot(tlist[start_id:], mean[start_id:],  color=cr, label=label,  **keyargs)
        
        if filling:
            ax.fill_between(tlist[start_id:], mean[start_id:]-std[start_id:], mean[start_id:]+std[start_id:], color=cr, alpha=.3)
        return ax
    
    def plot_combined(self, axes=None, cr='r', show_ab=True, filling=True):
        qty_name = ["B cell population",  "Affinity", "eta", "affinity gap", "variance", "sensitivity"]
        if axes is None:
            fig, axes = plt.subplots(figsize=(6, 8), dpi=100, ncols=2, nrows=3)
            plt.subplots_adjust(wspace=0.3)
            i=0
            for ax in axes:
                for axi in ax:
                    axi.set_ylabel(qty_name[i])
                    i += 1

        ax1= self.plotQty("n", ax=axes[0, 0], cr=cr, filling=filling)
        ax1.set_ylim(0, 1200)
        ax2= self.plotQty("e", ax=axes[0, 1], cr=cr, filling=filling)
        if show_ab:
            self.plotQty("eab", ax=axes[0, 1], filling=filling, cr=cr, linestyle='--')
        ax3=self.plotQty("eta", ax=axes[1, 0], cr=cr, filling=filling)
        ax3.set_ylim(0, 1)
        ax4=self.plotQty("de", ax=axes[1, 1], cr=cr, filling=filling)
        ax4.set_ylim(0, 10)
        
        ax5=self.plotQty("e_var", ax=axes[2, 0], cr=cr, filling=filling)
        ax5.set_ylim(0, 4)
        
        ax5=self.plotQty("sens", ax=axes[2, 1], cr=cr, filling=filling)
        return axes
    
    
def approx_de(mr):
    var = mr.mean_dataset["e_var"]
    eta = mr.mean_dataset["eta"]
    Eb = mr.mean_dataset["e"]
    n = mr.mean_dataset["n"]
    w = mr.mean_dataset["w"]
    
    w0 = mr.prm["w0"]
    nc = mr.prm["Nc"]
    pa = mr.prm["pa"]
    pd = mr.prm["pd"]
    eta0 = mr.prm["eta0"]
    fx = mr.prm["f"]
    Ef = fx*mr.prm["xb2"]/(2*Eb*4.012)
    
    wEbar = w0*(1-n/nc)*eta/(eta0+eta)+1
    
    
    dw = (pa+pd-pa*pd)*eta0*(1-eta)/(eta0+eta)*(1-1.5/Eb - Ef**2-(Ef/Eb)/(1-Ef)) + (wEbar/w-1)*Eb
    return dw, var*dw

### B cell classes
class PlasmaCell:
    def __init__(self, GC, E0, xb, t0=0):
        """
        a plasma B cell can generate antibodies which enter the feedback antibody pool and act as part of the tether bond
        the lifetime is T
        
        args:
            GC: germinal center class
            E0: BCR-Ag binding energy, barrier height
            xb: BCR-Ag bond length
            t0: the time point when a plasma cell is born
        """
        self.gc = GC
        self.alive = True
        self.E = E0
        self.xb = xb
        self.t0 = t0
        
        self.tau = lifetime_linear_cubic(self.E, self.xb) ### force-free bond lifetime
        self.T = GC.Tp   #40
        
    
    def goDie(self):
        """
        the plasma cell go die
        """
        self.alive = False


class Bcell:  
    def __init__(self, gc, E0, xb):
        self.gc = gc
        self.alive = True
        self.mature = False ## the B cell can extract antigen and compete for survival if True
        self.E = E0  ## binding energy, in kBT
        self.xb = xb ### BCR-Ag bond length, in nm
        self.eta = 0 ### average antigen extraction chance
        self.w = 0 ### fitness
        self.setup()
        return
        
    def setup(self):
        self.dE = self.gc.dE  ## binding energy change per mutation, in kBT
        self.dxb = self.gc.dxb ## bong length change per mutation, in nm
        self.pd = self.gc.pd # prob of differentiation to plasma cell, 0.02
        self.pa = self.gc.pa  ## apoptosis probability
        self.pm = self.gc.pm ### mutation probability
        self.pm_xb = self.gc.pm_xb
        self.tau = lifetime_linear_cubic(self.E, self.xb) ### force-free bond lifetime
        pass
        
    def divide(self, num_children=2):
        ## proliferate num_children daughter cells and go die afterwards
        sons = []
        
        for i in range(num_children):
            soni = Bcell(self.gc, self.E, self.xb)
            soni.mutate()
            sons.append(soni)
        self.goDie()
        return sons
    
    def comp_divide(self):
        '''
        divide according to fitness self.w
        
        ### num of sons = 2^ num of divisions
        num of suns = self.w
        '''
        if self.w == 0:
            return []
        
        num_division =int( min(np.random.poisson(self.w), self.gc.w0)) ### w0 is the maximal birth rate
        if USE_DIVISION:
            if num_division==0:
                self.goDie()
                return []

            sons = [self]
            for _ in range(num_division):
                new_sons = []
                for si in sons:
                    new_sons += si.divide()
                sons = new_sons
        else:
            sons = self.divide(num_division)
        return sons
    
    def differentiation(self, eta_mean):
        if self.eta>eta_mean and np.random.uniform()<self.pd:
            self.goDie()
            self.gc.addPlasma(self.E, self.xb)
        
    def mutate(self):
        ## this B cell can mutate to a different affinity
        
        if TWO_MUTATION: #### affinity and xb change simulatenously
            if np.random.uniform()<self.pm:
                self.E += np.random.normal(0, self.dE)
                xb_tmp = self.xb + np.random.normal(0, self.dxb)

                while xb_tmp < 0.1 or xb_tmp > 10:
                    ### restrict the range of xb
                    xb_tmp = self.xb + np.random.normal(0, self.dxb)
                self.xb = xb_tmp
        else:
            if np.random.uniform() < self.pm:
                 self.E += np.random.normal(0, self.dE)
                    
            if np.random.uniform() < self.pm_xb:
                ### change in bond length
                xb_tmp = self.xb + np.random.normal(0, self.dxb)

                while xb_tmp < 0.1 or xb_tmp > 10:
                    ### restrict the range of xb
                    xb_tmp = self.xb + np.random.normal(0, self.dxb)
                self.xb = xb_tmp
        return
    
        
    def goDie(self):
        self.alive = False
    
    def extract(self, Ea, xa):
        ## get extraction chance
        if FEEDBACK_XB:
            ## the tether bond length is determined by xb of the feedback antibody
            self.eta = self.gc.Ag_extract(Ea, self.E, xa,  self.xb)
        else:
            ## the tether bond length is fixed at gc.xb1
            self.eta = self.gc.Ag_extract(Ea, self.E, self.gc.xb1,  self.xb) ### fixing the tether stiffness during evolution
        self.mature=True
        
        if self.eta is None:
            self.eta = 0
        ## convert to fitness
        if self.gc.no_selection:
            self.w = self.gc.Tcell_selection(self.E-Ea)
        else:
            self.w = self.gc.Tcell_selection(self.eta)  
        return self.eta
    
    def apoptosis(self, threshold=-1):
        if self.eta == 0:
            self.goDie()
            return
        
        
        if self.gc.random_death:
            if np.random.uniform()<1-self.pa:
                self.goDie()
        else:
            if self.E < threshold:
                self.goDie()
        return
            
    
    
class GC:
    def __init__(self, f0=10.0, prm=prm_list):
            ## current number of alived B cells
        self.f = f0    ## pulling force applied by B cells
        self.prm = deepcopy(prm)
        
        self.qty_name = ["e", ### GC B cell affinity mean
                         "eStd",  ### GC B cell affinity std
                         "eVar", ### GC B cell affinity variance
                         "eMp", ### most prob affinity
                         "tau", ### GC B cell force-free bond lifetime
                         "tauVar", ### lifetime variance
                         
                         "tauF", ### GC B cell lifetime under force
                         
                         "tau0", ### initial lifetime
                         
                         "xb", ### GC B cell bond length
                         "xbStd", ### GC B cell bond length std
                         
                         "ep",  ### plasma cell affinity, mean
                         "epStd", ### plasma cell affinity, std
                         "tau_p", ### plasma cell bond lifetime
                         "tau_pStd", ### plams cell lifetime std
                         
                         "eab",  ### Ab pool mean affinity
                         "eabStd", ### Ab pool affinity std
                         "eabVar",
                         "xb_ab", ### Ab pool mean xb
                         "xb_abStd", ### Ab pool xb std
                         
                         "Ec", ### thresholding feedback affinity
                         
                         "tau_ab",
                         "tau_abStd",
                         
                         "n",  ### B cell pop size
                         "np", ###  plasma cell pop size
                         "nab", ### antibody number
                         "w",  ### B cell fitness, mean
                         "wStd", ### B cell fitness, std
                         "eta",  ### B cell extraction chance, mean
                         "etaStd", ### extraction chance, std
                         "cov", ### covariance between fitness and E
                         "cov_lmd", ## cov / mean fit
                         "cov2", ### covariance between fitness and E^2
                         "de", ### affinity gap between Ab and B cell]
                         "deltaE", ### affinity diff
                         "sens", 
                         "sens_var", ## sens * var
                         "finished", ## finished or not, single value,
                         "mat_rate",
                         "f", ### force
                         "Nb", ### bottle neck pop size
                        ]
            
        self.dataset = {}
                          
        self.eta_record = {} 

        self.no_selection = False
        
        self.setup()
        pass
    
    def load_prm(self, prm):
        ### load parameters
        self.Td = int(prm["Td"])  ### feedback delay time
        self.Tp = int(prm["Tp"])  ### plasma cell lifetime
        self.xb1 = prm["xb1"]     ### APC-Ag bond length
        self.xb2 = prm["xb2"]     ### BCR-Ag bond length
        self.w0 = prm["w0"]       ### max proliferation rate
        
        self.N0 = prm["N0"]       ### initial B cell pop size
        self.cag = prm["cag"]     ### antigen density
        self.dE = prm["dE"]       ### mutation step size for barrier height
        self.dxb = prm["dxb"]     ### mutation step size for bond length
        
        self.pa = prm["pa"]       ### probability for apoptosis
        self.pd = prm["pd"]       ### probability for death
        self.pm = prm["pm"]       ### probability for mutation
        self.pm_xb = prm["pm_xb"]
        
        self.Ea = prm["Ea"]       ### tether affinity, no feedback only
        self.Eb = prm["Eb"]       ### naive B cell barrier height
        
        self.df = prm["df"]       ### force increament per GC cycle
        
        self.f = prm["f"]         ### force 
        
        self.eta0 =prm["eta0"]    ### half saturation in fitness when kernel=2
        self.Nc = prm["Nc"]       ### GC B cell pop capacity
        self.Npc = prm["Npc"]
        
        self.Nab = prm["Nab"]     ### feedback Ab pool size
        
        self.cutoff = prm["cutoff"]
        
        ## for feedback only
        self.feedback = prm["feedback"]     ### bool, if feedback or not
        
        self.randomAb = prm["randomAb"]     ### randomize tether affinity
        
        self.useSim = prm["useSim"]          ### use numerical simulation to obtain eta
        if self.useSim:
            self.sto = sb.System()
            
        self.update_rule = prm["update_rule"]
        
        self.output = prm["output"]         ### write out infomration and warnnings
        
        self.useWeek = prm["useWeek"]       ### False: use GC cycle as time, True: use weeks
        
        self.debug= prm["debug"]
        
        self.useBell = prm["useBell"]       ### use Bell formula
        
        self.goodDiff = prm["goodDiff"]     ### only chose B cell with fitness above the average for differentiation. 
        
        self.useBinomial = prm["useBinomial"]   ### use n_ag instead of eta to compute fitness
        
        self.no_selection = prm["no_selection"]   ### pure diffusion
        
        self.potential = prm["potential"]     ### type of interacing potential
        
        self.dump_eta = prm["dump_eta"]       ### save eta or not
        
        self.random_death = (prm["death"] == "random")   ### random death          
        
        self.eta_record_file = "eta_record/"+self.potential +"/" + prm["eta_file"]
        
        
        self.extraction_under_dynamical_force = prm["dynamical_force"]
        self.ramping_rate = prm["r"]
        return
    
    def load_eta_record(self):
        try:
            with open(self.eta_record_file, "r") as myfile:
                while True:
                    line = myfile.readline()
                    if not line:
                        break
                    s = line.split('\t')
                    if self.feedback:
                        key = (int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]))
                        val = float(s[5])
                    else:
                        key = (int(s[0]), int(s[1]), int(s[2]))
                        val = float(s[3])
                    self.eta_record[key] = val
        except:
            #self.eta_record = {}
            pass
        return
    
    def dump_eta_record(self):
        with open(self.eta_record_file, "w") as myfile:
            for key, val in self.eta_record.items():
                if self.feedback:
                    myfile.write("{0:d}\t{1:d}\t{2:d}\t{3:.4f}\n".format(key[0], key[1], key[2], key[3], key[4], val))
                else:
                    myfile.write("{0:d}\t{1:d}\t{2:.4f}\n".format(key[0], key[1], key[2], val))
        return
    
    def read_eta_from_record(self, Ea, Eb, xa, xb, f):
        if not self._used_sim_already and self.output and False:
            print("using simulation!") 
        key = ( my_round(Ea), my_round(Eb), my_round(xa), my_round(xb), my_round(f) )
        if key in self.eta_record:
            eta = self.eta_record[key]
        else:
            eta = self.run_sim(Ea, Eb, xa=xa, xb=xb, f=f)
            self.eta_record[key] = eta
            self._used_sim_already=True
        return eta
    
    def run_sim(self, Ea, Eb,  dE=1, xa=1.5, xb=2.0, f=None):
        '''
        run Brownian simulations to get eta
        
        '''
        
        if not self.useSim:
            return None
        self.sto.prm["Eb1"] = Ea
        self.sto.prm["Eb2"] = Eb
        if f is None:
            self.sto.prm["f0"] = self.f
        else:
            self.sto.prm["f0"] = f
        self.sto.prm["xb1"] = xa
        self.sto.prm["xb2"] = xb
        
        self.sto.prm["potential"] = self.potential
        if dE>3:
            self.sto.prm["dt"] = 2
            self.sto.numSample=100
        elif dE<=0.1:
            self.sto.prm["dt"] = 0.2
            self.sto.numSample=400
        else:
            self.sto.prm["dt"] = 1
            self.sto.numSample=200
            
        self.sto.setup()
        print("@", end="")
        return self.sto.run()
    
        
    def setup(self):
        self.load_prm(self.prm)
        
        self.N = 0 
        self.NP = 0 ### num of plasma cell

        
        self.Ec = self.Eb
        
        self.t = 0
        self.tm = 0 
        self.finished = False   ## True if affinity reaches 10^10
        self._used_sim_already= False
        self._warned = False
        
        self.agents = [] ### GC B cell list
        self.plasma = [] ### plasma cell list
        self.Ab_pool = [] #[np.ones(self.Nab)*self.Eb]  ### feedback Ab pool
        self.my_Ab_pool = [] ### current Ab pool
        self.Ab_pool_taul_list = []
        self.tether_aff = self.Eb ### tether affinity
        
        self.division_record = [] ### record number of divisions for all B cells
        for qty in self.qty_name:
            self.dataset[qty] = []

        self.etaList = [] ### current eta list
        
        self.fit_mean, self.fit_max = 0, 0
        
        for i in range(self.N0):
            bcell = Bcell(self, self.Eb, self.xb2)
            self.addAliveBcell(bcell)
        self.checkN()
        self.N_ave = self.N
        self.N_min = self.N
        
        self.N_last = self.N ### pop size of last cycle, to smooth the pop curve when it is close to the capacity
        
        self.lightZoneStep() ### iniitialize extraction
        self.collectData()
        self.dataset["tau0"].append(lifetime_linear_cubic(self.Eb, self.xb2))
        
        self.alive = True
        return
    
    def step(self):
        self.N_ave = self.N/2
        
        if self.debug:
            self._print("*"*20)
            self._print(["current time:", self.t])
            
        self.lightZoneStep()
        
        ## remove dead B cells.
        self.clean()
        if self.debug:
            self._print(["light_zone_step DONE!, current pop=", self.N])

        self.N_ave += self.N/2
        
        ## collecting data
        self.collectData()
        if self.debug:
            self._print(["collect_data DONE!, mean_fit={0:.4f}, max_fit={1:.4f}".format(self.fit_mean, self.fit_max)])
        
        ## in the dark zone
        ## B cells divide and mutate
        self.darkZoneStep()
        
        ## remove dead B cells.
        self.clean()
        
        if self.debug:
            self._print(["dar_zone_step DONE!, current pop=", self.N])
        #self.N_ave = self.N
        ## plasma decay
        self.plasmaStep()

        self.t += 1
        
        self.update_force()
        return
    
    
    def update_force(self):
        
        self.f += self.df
        return
    
    ############# main fuctions ####################
    def Ag_extract(self, Ea, Eb, xa, xb):
        flag = True
        if self.useBell:
            eta0, flag, barrier_height = extRatio_linear_cubic(xa, xb, Ea, Eb, 0, output=False, Eb_min=self.prm["Eb_min"])
            tau0 = 1/eta0 - 1
            eta = 1.0/(1.0+tau0*np.exp(self.f*(xb-xa)/4.012))

        else:
            if self.potential=="cusp":
                if self.extraction_under_dynamical_force:
                    flag = True
                    key = (my_round(Ea), my_round(Eb), my_round(self.ramping_rate * 1e6))
                    if key in self.eta_record:
                        eta = self.eta_record[key]
                    else:
                        eta = theory.extRatio_rampongF_cuspHarmonic(E1=Ea, E2=Eb, xa=xa, xb=xb, r=self.ramping_rate, f0=0, fm=60, m=1)
                        self.eta_record[key] = eta
                else:
                    eta, flag = extRatio(xa, xb, Ea, Eb, self.f, output=False)
            elif self.potential=="linear-cubic":
                eta, flag, barrier_height = extRatio_linear_cubic(xa, xb, Ea, Eb, self.f, output=False, Eb_min=self.prm["Eb_min"])
            if not flag and not self._warned and self.output and False:
                print("*GC warning: low barrier! Ea={0:.1f}, Eb={1:.1f}, dE={2:.2f}, xa={5:.2f}, xb={4:.2f}, f={3:.3f}".format(Ea, Eb, barrier_height, self.f, xb, xa))
                self._warned=True

            if not flag and self.useSim:
                if self.output and not self._used_sim_already:
                    if eta is None and self.output and False:
                        print("*start sim @t={0:.0f}, F={1:.2f}, eta_theory=None".format(self.t, self.f), end=", ")
                        pass
                    elif self.output and False:
                        print("*start sim @t={0:.0f}, F={1:.2f}, eta_theory={2:.3f}".format(self.t, self.f, eta), end=", ")
                eta = self.read_eta_from_record(Ea, Eb, xa, xb, self.f)
        if self.useBinomial:
            return np.random.binomial(self.cag, eta)
        else:
            return eta
        
        
    def interpolation(key):
        if key in self.eta_record:
            return self.eta_record[key]
        
        return 0
        
        
    def Tcell_selection(self, eta):
        if self.no_selection:
            ## weak selection
            if eta<3:
                return self.w0*np.exp(0.1*eta)*(1-self.N/self.Nc)/np.exp(0.3)
            else:
                return self.w0*(1-self.N/self.Nc)
        
        if self.useBinomial:
            return self.w0*eta/(self.cag*self.eta0+eta) * (1-self.N/self.Nc)
        else:
            return self.w0*eta/(self.eta0+eta) * (1-self.N/self.Nc)
    #################################################
    
    def collectData(self):
        
        # check if N = number of alive B cell or not
        self.checkN()
        
        
        ## record number of B cells ## averaging over the entire circle
        self.append("n", (self.N_ave + self.N_last)/2)
        
        self.N_min = min(self.N_min, (self.N_ave + self.N_last)/2)
        self.append("Nb", self.N_min)
        
        self.N_last = self.N_ave
        
        
        
        ## record affinity distribution
        elist = self.getCurrentAffinityList()
        eb = self._get_mean(elist)
        eb_var = self._get_var(elist)
        self.append("e", eb)
        self.append("eStd", np.sqrt(eb_var))
        self.append("eVar", eb_var)
        self.append("eMp", self._get_most_prob(elist))
        
        if len(self.dataset["e"]) > self.cutoff:
            self.append("mat_rate", np.mean(diff(self.dataset["e"])[-self.cutoff:]))
        else:
            if len(self.dataset["e"]) >1:
                self.append("mat_rate", np.mean(diff(self.dataset["e"])))
            else:
                self.append("mat_rate", 0)
        ### record bond length distribution
        xblist = self.getCurrentXbList()
        self.append("xb", self._get_mean(xblist))
        self.append("xbStd", self._get_std(xblist))
        
        ## record lifetime distribution
        taulist = self.getCurrentLifetimeList()
        tau_mean = self._get_mean(taulist)
        tau_var = self._get_var(taulist)
        self.append("tau", tau_mean)
        self.append("tauVar", tau_var)
        self.append("tauStd", np.sqrt(tau_var))
        
        ### record lifetime under force
        self.append("tauF",  lifetime_linear_cubic(eb, self._get_mean(xblist), f0=self.f))
            
        ## record number of plasma cells
        self.append("np", self.NP)
        
        ## record plasma cell affinity
        eplist = self.getCurrentPlasmaAffinityList()
        self.append("ep", self._get_mean(eplist))
        self.append("epStd", self._get_std(eplist))
        
        ### record plasma cell lifetime distribution
        tau_p_list = self.getCurrentPlasmaLifetimeList()
        self.append("tau_p", self._get_mean(tau_p_list))
        self.append("tau_pStd", np.sqrt(self._get_var(tau_p_list)))
        
        
        ### record Ab pool affinity
        Ab_pool_Eb = [Ab[0] for Ab in self.my_Ab_pool]
        eab = self._get_mean(Ab_pool_Eb)
        eab_std = self._get_std(Ab_pool_Eb)
        self.append("eab", eab)
        self.append("eabStd", eab_std)
        self.append("eabVar", eab_std**2)
        self.append("nab", len(self.my_Ab_pool))
        self.append("Ec", min(Ab_pool_Eb))
        
        Ab_pool_xb = [Ab[1] for Ab in self.my_Ab_pool]
        self.append("xb_ab", self._get_mean(Ab_pool_xb))
        self.append("xb_abStd", self._get_std(Ab_pool_xb))
        
        Ab_pool_tau = [lifetime_linear_cubic(Ab[0], Ab[1]) for Ab in self.my_Ab_pool]
        self.append("tau_ab", self._get_mean(Ab_pool_tau))
        self.append("tau_abStd", self._get_std(Ab_pool_tau))
        
        ### record extraction list
        etaList = self.getCurrentEtaList()
        eta = np.mean(etaList)
        self.append("eta", eta)
        self.append("etaStd", np.std(etaList))
        ## record extract chance, fitness and affinity
        
        fitList = self.getCurrentFitList()
        fit = self._get_mean(fitList)
        self.fit_mean = fit
        self.fit_max = self._get_max(fitList)
        self.append("w", fit)
        self.append("wStd", self._get_std(fitList))
        
        if self.N>0:
            self.append("de", eb-eab)
        else:
            self.append("de", np.nan)
        
        cov = my_cov(elist, fitList)
        self.append("cov", cov)
        self.append("cov_lmd", cov/fit)
        
        e2list = [ei**2 for ei in elist]
        self.append("cov2", my_cov(e2list, fitList))
        
        if self.tm == 0 or self.N<1:
            deltaE = 0
        else:
            deltaE = eb - self.dataset["e"][self.tm-1]
        self.append("deltaE", deltaE)
        
        self.append("f", self.f)
        
        
        
        
        #sens = np.nan
        #if self.N>0:
        #    sens = self._get_sens3(eab, eb)
        #self.append("sens", sens)
        #self.append("sens_var", sens*eb_var)
        #self.tm +=1
        return
    
    
    def darkZoneStep(self): 
               
        
            
        ### apoptosis
        threshold = -1
        if not self.random_death:
            threshold = self.quick_select(self.getCurrentAffinityList())
            if self.debug:
                min_aff, max_aff = min(self.getCurrentAffinityList()), max(self.getCurrentAffinityList())
                print("aff threshold:", threshold, ", min aff=", min_aff, ", mmax_aff=", max_aff)
            
        for bcell in self.agents:
            if bcell.alive:
                bcell.apoptosis(threshold)
                
        ### differentiate
        if self.goodDiff:
            mean_eta = np.mean(self.etaList)
        else:
            mean_eta = -1
        for bcell in self.agents:
            if bcell.alive:
                bcell.differentiation(mean_eta)
                
                
        ### proliferation and mutation
        daughter_cell_list = []
        self.division_record = []
        
        for bcell in self.agents:
            if bcell.alive:
                ret = bcell.comp_divide()
                if ret:
                    daughter_cell_list += ret
                self.division_record.append([bcell.eta, len(ret)])
        for bcell in daughter_cell_list:
            self.addAliveBcell(bcell)   
            
        
        return
    
    def update_Ab_pool(self):
        if not self.feedback:
            print("warning! updating tether pool even without feedback")
            return
        if self.update_rule == "all":
            plasma_aff = self.getCurrentPlasmaAffinityList(include_xb=True)+[(self.Eb, self.xb2)]
            self.Ab_pool.append(plasma_aff)
            
        elif self.update_rule == "random":
            ###### this part is problematic
            
            plasma_aff = self.getCurrentPlasmaAffinityList(include_xb=True)+[(self.Eb, self.xb2)]
            rd.shuffle(plasma_aff)
            new_Ab = np.zeros(self.Nab)
            for i in range(self.Nab):
                new_Ab[i] = max(self.Ab_pool[-1][i], plasma_aff[i % len(plasma_aff)])
            self.Ab_pool.append(new_Ab)
        elif self.update_rule == "topK":
            self.Ab_pool.append(self.getCurrentTopPlasmaAffinityList(self.Nab))
        else:
            raise Exception("updating rule not found")
        return
    
    def lightZoneStep(self, test=False):
        
        if self.feedback and not test:
            ### update the feedback Ab pool
            #self.Ab_pool.append(self.getCurrentPlasmaAffinityList())
            self.update_Ab_pool()
        
        ## intial eta list
        if self.feedback and self.t>self.Td:
            if self.randomAb:
                self.my_Ab_pool = self.Ab_pool[self.t-self.Td] 
            ## random shuffle feedback Ab pool
                np.random.shuffle(self.my_Ab_pool)
            else:
                self.my_Ab_pool = [np.mean(self.Ab_pool[self.t-self.Td], axis=0).tolist()]
        elif self.feedback:
            self.my_Ab_pool = [(self.Eb, self.xb2)]
        else: ## no feedback
            self.my_Ab_pool = [(self.Ea, self.xb1)]
        Ab_num = len(self.my_Ab_pool)
        
        
        ## B cells extract antigens 
        order = 0
        for bcell in self.agents:
            if bcell.alive:
                eta = bcell.extract(*self.my_Ab_pool[order % Ab_num])
                order += 1
        self.etaList = self.getCurrentEtaList()
        return
        
    def checkN(self):
        assert len(self.agents)==self.N, \
            "at t={0:d} population wrong: len(agents)={1:d} N={2:d}".format(self.t, len(self.agents), self.N)
        
    def clean(self):
        ## remove dead B cells form germinal center
        newAgents = []
        for bcell in self.agents:
            if bcell.alive:
                newAgents.append(bcell) 
        self.agents = newAgents
        self.N = len(self.agents)
        self.checkN()
        return
    
    def addAliveBcell(self, bcell):
        if bcell.alive:
            self.agents.append(bcell)
            self.N += 1
            
        return
    
    def removeDeadBcell(self, bcell):
        if not bcell.alive:
            self.agents.remove(bcell)
            self.N -= 1
        return
        
    def getCurrentPlasmaAffinityList(self, include_xb=False):
        Elist = []
        for plasma in self.plasma:
            if plasma.alive:
                if include_xb:
                    Elist.append( (plasma.E, plasma.xb) )
                else:
                    Elist.append(plasma.E)
        return Elist
    
    def getCurrentPlasmaLifetimeList(self):
        taulist = []
        for plasma in self.plasma:
            if plasma.alive:
                taulist.append(plasma.tau)
        return taulist
    
    def getCurrentFeedbackAbList(self):
        return [Ab[0] for Ab in self.my_Ab_pool]
    
    def getCurrentTopPlasmaAffinityList(self, k=1):
        tau0 = lifetime_linear_cubic( self.Eb, self.xb2)
        Elist = [(tau0, self.Eb, self.xb2)]
        for plasma in self.plasma:
            if plasma.alive: # and plasma.tau > tau0:
                if len(Elist) < k:
                    heapq.heappush(Elist, (plasma.tau, plasma.E, plasma.xb))
                else:
                    heapq.heappushpop(Elist, (plasma.tau, plasma.E, plasma.xb)) ### first push and then pop. Always maintain the largest K values
        
        self.Ec = Elist[0][1]
        self.Ab_pool_taul_list = [ei[0] for ei in Elist]
        return [(ei[1], ei[2]) for ei in Elist]
    
    
        
    def getCurrentAffinityList(self):
        Elist = []
        for bcell in self.agents:
            if bcell.alive:
                Elist.append(bcell.E)
        return Elist
    
    def getCurrentLifetimeList(self):
        taulist = []
        for bcell in self.agents:
            if bcell.alive:
                taulist.append(bcell.tau)
        return taulist
    
    def getCurrentXbList(self):
        xblist = []
        for bcell in self.agents:
            if bcell.alive:
                xblist.append(bcell.xb)
        return xblist

    def getCurrentEtaList(self):
        etaList = []
        for bcell in self.agents:
            if bcell.alive and bcell.mature:
                ### skip the new born B cell
                etaList.append(bcell.eta)
                #raise Exception("some B cells have zero eta")
        return etaList
    
    def getCurrentFitList(self):
        fitList= []
        for bcell in self.agents:
            if bcell.alive and bcell.mature:
                fitList.append((bcell.w*self.pa*(1-self.pd)))
        return fitList
    
    def _get_sens(self, eta, Eb):
        deta = self.eta0*(1-eta)/(self.eta0+eta)
        Ef = self.f*self.xb2/(2*Eb*4.012)
        dtau = 1#(1-1.5/Eb - Ef**2-(Ef/Eb)/(1-Ef))
        dw =deta*dtau
        return dw
    
    def _get_sens2(self, w0, w, Eb0, Eb):
        return (w-w0)/(w*(Eb-Eb0))
    
    def _get_sens3(self, ea, eb):
        de = 0.01
        eta1, flag1, _ = extRatio_linear_cubic(e10=ea, e20=eb+de, xb1=self.xb1, xb2=self.xb2, f0=self.f)
        eta2, flag2, _= extRatio_linear_cubic(e10=ea, e20=eb-de, xb1=self.xb1, xb2=self.xb2, f0=self.f)
        
        if not (flag1 and flag2):
            return np.nan
        
        eta = (eta1+eta2)/2
        eta0 = self.prm["eta0"]
        deta_dGb = (eta1-eta2)/(2*de)
        return eta0/(eta*(eta0+eta))*deta_dGb
        
    def _get_mean(self, alist):
        if len(alist)>0:
            return np.mean(alist)
        else:
            return np.nan
    def _get_std(self, alist):
        if len(alist)>0:
            return np.std(alist)
        else:
            return np.nan
    def _get_max(self, alist):
        if len(alist)>0:
            return max(alist)
        else:
            return np.nan
        
    def _get_var(self, alist):
        if len(alist)>0:
            return np.var(alist)
        else:
            return np.nan
        
    def _get_most_prob(self, array):
        hist, bins = np.histogram(array, bins=30)
        index = np.argmax(hist)
        return (bins[index] +bins[index+1])/2
    
    def quick_select(self, alist):
        if not alist:
            return -1
        
        k = int(len(alist)*self.pa)
        if k>= len(alist):
            return min(alist)
        return alist[self._quick_select(alist, 0, len(alist)-1, k)]
    
    def _quick_select(self, alist, left, right, k):
        if left >= right:
            return k
        
        p = self._partition(alist, left, right)
        if p==k:
            return p
        elif p>k:
            return self._quick_select(alist, left, p-1, k)
        else:
            return self._quick_select(alist, p+1, right, k)
        
    
    def _partition(self, alist, left, right):
        pivot = alist[right]
        a = left
        for i in range(left, right):
            if alist[i]>=pivot:
                alist[i], alist[a] = alist[a], alist[i]
                a +=1
                
        alist[right], alist[a] = alist[a], alist[right]
        return a
        
    def _print(self, s):
        if type(s) == type(""):
            print(s)
        else:
            for si in s:
                print(si, end=", ")
            print("")
        return
        
    def append(self, qty, value):
        
        if qty in self.qty_name:
            
            self.dataset[qty].append(value)
        else:
            self.qty_name.append(qty)
            self.dataset[qty] = [value]
    
    def addPlasma(self, E, xb):
        if INFINITE_PLASMA or self.NP < self.Npc:
            self.plasma.append(PlasmaCell(self, E, xb, self.t))
            self.NP += 1
        else:
            ### if plasma pool is full, randomly delete one plasma cell and add the new one
            rand_idx = np.random.randint(self.NP)
            self.plasma[rand_idx] = PlasmaCell(self, E, xb, self.t)
        return
    
    def plasmaStep(self):
        for cell in self.plasma:
            if cell.alive:
                if self.t - cell.t0 > cell.T and cell.T>0:
                    cell.goDie()
        newPlasmaList = []
        for cell in self.plasma:
            if cell.alive:
                newPlasmaList.append(cell)
            else:
                self.NP -= 1
        self.plasma = newPlasmaList
        return
    
    def _death_process(self):
        ### apoptosis
        for bcell in self.agents:
            if bcell.alive:
                bcell.apoptosis()
        return
    
    def _birth_process(self):
        ### proliferation and mutation
        daughter_cell_list = []
        
        for bcell in self.agents:
            if bcell.alive:
                ret = bcell.comp_divide()
                if ret:
                    daughter_cell_list += ret
        
        for bcell in daughter_cell_list:
            self.addAliveBcell(bcell)
        return
    
        
    def run(self, tm=10):
        if self.output:
            pass
            #write("Info: start simulation")
        if self.dump_eta:    
            self.load_eta_record()
        while self.t < tm and self.N>0:
            self.step()
            if self.output:
                
                sb.printProgress(self.t, tm)
            if self.N > 2E4:
                write("warning: population too large %d"%self.N)
        if self.N == 0:
            self.alive = False
        if self.dump_eta and len(self.eta_record)>0:   
            self.dump_eta_record()
        if self.output:
            if self.N == 0:
                write("Info: GC dead")
            else:
                write("Info: finished!")
        if self.N == 0:
            self.dataset["finished"] = False
            return False
        else:
            self.dataset["finished"] = True
            return True
    
    
    def plotQty(self, qty, cr="r", label="", ax=None, filling=False,  **keyargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(4,3), dpi=100)
            ax.set_xlabel("week", fontsize=12)
            
            ax.set_ylabel(qty, fontsize=12)
            
        if filling:
            mean = np.asarray(self.dataset[qty])
            std = np.asarray(self.dataset[qty+"Std"])
            for i in range(len(mean)):
                if mean[i] is None:
                    continue
                start_id = i
                break
            if self.useWeek:
                tlist = convert_to_week(range(len(mean)))
            else:
                tlist = range(len(mean))
            ax.plot(tlist[start_id:], mean[start_id:],  color=cr, label=label,  **keyargs)
            ax.fill_between(tlist[start_id:], mean[start_id:]-std[start_id:], mean[start_id:]+std[start_id:], color=cr, alpha=.3)
        else:
            if self.useWeek:
                tlist = convert_to_week(range(len(self.dataset[qty])))
            else:
                tlist = range(self.tm)
            ax.plot(tlist, self.dataset[qty], color=cr, label=label)
        return ax

    def plotAff2(self, ax=None, cr='r', fmt='x', label=""):
        if not ax:
            ax = getAx("plasma affinity", "GC affinity")
        ax.plot(self.dataset["eab"], self.dataset["e"], color=cr, marker=fmt, label=label)
        return ax
    
    def plot_combined(self, full=False):
        if full:
            fig, axes = plt.subplots(figsize=(5, 6), dpi=100, ncols=2, nrows=3)
        else:
            fig, axes = plt.subplots(figsize=(5, 4), dpi=100, ncols=2, nrows=2)
        plt.subplots_adjust(wspace=0.2, hspace=0.0)
        qty_name = ["B cell population N",  "affinity, [kT]", r"extraction chance $\eta$",  "variance [kT$^2$]", r"fitness, $\lambda$", r"sensitivity $\alpha$"]
        

        i=0
        if self.useWeek:
            xtick = [0, 1,2,3,4]
            for ax in axes:
                for axi in ax:
                    axi.set_xlim(0, 4)
                    axi.set_ylabel(qty_name[i])
                    axi.set_xticks(xtick)
                    axi.set_xticklabels(xtick, fontsize=0)
                    i += 1
        else:
            for ax in axes:
                for axi in ax:
                    axi.set_ylabel(qty_name[i])
                    if self.t>199 and self.t<301:
                        axi.set_xticks([0, 100, 200, 300])
                    i += 1
            xtick = [0, 100, 200, 300]
        ax1=self.plotQty("n", ax=axes[0, 0])
        ax1.set_ylim(0, 1200)
        ax2=self.plotQty("e", ax=axes[0, 1], filling=True)
        self.plotQty("eab", ax=axes[0, 1], filling=True, cr='C0')
        
        
        ax3=self.plotQty("eta", ax=axes[1, 0], filling=True)
        #ax3.set_ylim(0,1)
        
        ax4= self.plotQty("e_var", ax=axes[1, 1], filling=False)
        #ax4=self.plotQty("w", ax=axes[1, 1], filling=True)
        #ax4.set_ylim(0, 2)
        if full:
            self.plotQty("w", ax=axes[2, 0], filling=False)
            self.plotQty("sens", ax=axes[2, 1], filling=False)
        if full:
            bottom_ax = 2
        else:
            bottom_ax = 1
        if self.useWeek:
            axes[bottom_ax, 0].set_xlabel("week")
            axes[bottom_ax, 0].set_xticks(xtick)
            axes[bottom_ax, 0].set_xticklabels(xtick, fontsize=10)
            axes[bottom_ax, 1].set_xlabel("week")
            axes[bottom_ax, 1].set_xticks(xtick)
            axes[bottom_ax, 1].set_xticklabels(xtick, fontsize=10)
        else:
            axes[bottom_ax, 0].set_xlabel("GC cycle")
            axes[bottom_ax, 1].set_xlabel("GC cycle")
        #plt.grid()
        plt.tight_layout()
        #plt.show()
        return axes
    
    
    def plot_combined_vert(self, cr='r'):
        fig, axes = plt.subplots(figsize=(2, 6), dpi=100,  nrows=4)
        #plt.subplots_adjust( hspace=0.2)
        qty_name = ["B cell population",  "affinity", "extraction chance", "birth rate", "cov", "de"]

        i=0
        xtick = [0, 1,2,3,4]
        for axi in axes:
            axi.set_xlim(0, 4)
            axi.set_ylabel(qty_name[i])
            axi.set_xticks(xtick)
            axi.set_xticklabels(xtick, fontsize=0)
            i += 1
        self.plotQty("n", ax=axes[0], cr=cr)
        #gc0.plotQty("np", ax=axes[0, 0], cr="C1")
        #gc0.plotQty("nab", ax=axes[0, 0], cr="C4")
        self.plotQty("e", ax=axes[1], filling=True, cr=cr)
        self.plotQty("eab", ax=axes[1], filling=True, cr='C0')
        
        self.plotQty("eta", ax=axes[2], filling=True, cr=cr)
        self.plotQty("w", ax=axes[3], filling=True, cr=cr)

        bottom_ax = 3
        
        axes[bottom_ax].set_xlabel("GC cycle")
        axes[bottom_ax].set_xticks(xtick)
        axes[bottom_ax].set_xticklabels(xtick, fontsize=10)
        #plt.grid()
        #plt.tight_layout()
        #plt.show()
        return axes
    
    
    
############################################## scanner   ###############

class Scan_prm0:
    
    def __init__(self):
        self.dataset = {}
        self.qty_name = []
        self.dataset_deleted = {}
        pass
    
    def setup(self):
        for qty in self.qty_name:
            self.dataset[qty] = []
            self.dataset_deleted[qty] = []
        self.dataset_deleted["index"] = []
    
    def copy(self, other_obj):
        self.dataset = other_obj.dataset.copy()
        self._unzipDataset()
        return
    
    def dump(self, filename, unique=True):
        """
        save data to file
        """
        #self._zip2Dataset()
        if unique:
            ### save to json
            v = 0
            while os.path.exists(filename+'.json'):
                print("file exists! ")
                filename += gen_pi(v)
                v += 1
        
        with open(filename+'.json', 'w') as fp:
            json.dump(self.dataset, fp)
        return
    
    def load(self, filename):
        """
        load data from filename
        """
        ### load the json
        with open(filename+'.json', 'r') as fp:
            self.dataset = json.load(fp)
        
        #self._unzipDataset()
        
        return
    
    def recover(self, index):
        ind = self.dataset_deleted["index"].index(index)
        for qty in self.qty_name:
            try:
                self.dataset[qty].append(self.dataset_deleted[qty][ind])
            except:
                pass
        return
    
    def remove(self, elements):
        try:
            for ei in elements:
                self._remove_index(ei)
        except:
            self._remove_index(elements)
        return
    
    def _remove_index(self, index):
        for qty in self.qty_name:
            if type(self.dataset[qty]) is type([1,2]):
                if len(self.dataset[qty])>0:
                    value = self.dataset[qty].pop(index)
                    self.dataset_deleted[qty].append(value)
                    self.dataset_deleted["index"].append(index)
        return
    
    def print2(self, qty):
        print(qty)
        print(["{0:.3f}".format(i) for i in self.dataset[qty]])
        return
    
    
    def get_data_type(self):
        for qty in self.qty_name:
            try:
                print(qty,", ", type(self.dataset[qty][0]))
            except:
                print(qty,", ", type(self.dataset[qty]))
                
                
                
class evo_scan(Scan_prm0):
    
    def __init__(self, ea):
        super().__init__()
        self.qty_name = ["prm_name", "prms", "mat_rate",  "var", "eta",  "dE", "Eb", "cov", "surv_prob", "fit", "N", "eab", "eab_var", "sens", "tau", "xb"]
        self.qty_label = ["parameter name",
                          "parameter value",
                          "maturation rate, kT/cycle",
                          "affinity variance",
                          "extraction chance, $\eta$",
                          "BCR tether affinity difference, kT",
                          "affinity-fitness covariance",
                          "GC survival percentage", 
                          "fitness",
                          "GC B cell population size",
                          "Ab pool mean affinity, kT",
                          "Ab pool affinity variance"]
        self.dataset = {}
        self.ea = ea
        self.sample_num = 20
        self.cutoff = 20
        self.save = False
        self.setup()
        
    def setup(self):
        super().setup()
        self.ea.output=False
        self.f = self.ea.f
        self.eta0 = self.ea.prm["eta0"]
        pass
    
    def auto_dump(self):
        curr_time = datetime.now().strftime("%m-%d-%y_%H-%M")
        if self.save:
            curr_time +=  "exp"
        else:
            curr_time += "test"
        self.dump("data/auto_saved/"+curr_time)
        self.save_prm("data/auto_saved/"+curr_time)
        print("save to: ", curr_time)
        return
        
    def run(self, prm_name, prm_list, filename):
        
        
        print(prm_name, "\ts\tvar\teta\tde\tEb\tsurve_prob")
        self.prm_name = prm_name
        self.dataset["prm_name"] = prm_name
        for prm in prm_list:
            self.ea.prm[prm_name] = float(prm)
            self.ea.setup()
            flag = self.ea.run(self.sample_num)
            if not flag:
                print("{0:.3f}\t All GC died".format(prm))
                self.collect_data(prm, flag)
                continue

            mr = np.mean(self.get_mat_rate())
            var = np.mean(self.get_mean_end(self.ea.dataset["e_var"]))
            eta = np.mean(self.get_mean_end(self.ea.dataset["eta"]))
            de = np.mean(self.get_mean_end(self.ea.dataset["de"]))
            Eb = np.mean(self.get_mean_end(self.ea.dataset["e"], 1))
            tau = np.mean(self.get_mean_end(self.ea.dataset["tau"], 1))
            self.collect_data(prm)
            print("{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.3f}\t{6:.3f}\t{7:.3f}".format(prm, mr, var, eta, de, Eb, self.ea.surv_prob, tau))
            self.dump("data/auto_saved/"+filename, unique=False)
        self.auto_dump()
        return
    
    def remove_by_prm(self, prm):
        index = self.dataset["prms"].index(prm)
        self.remove(index)
        return
    
    def sort_by_prm(self):
        n = len(self.dataset["prms"])
        for i in range(n):
            for j in range(i+1, n):
                if self.dataset["prms"][i]<self.dataset["prms"][j]:
                    self._switch(i, j)
                    
        
    def _switch(self, i, j):
        for qty in self.qty_name:
            if qty!="prm_name":
                tmp = self.dataset[qty][i]
                self.dataset[qty][i]=self.dataset[qty][j]
                self.dataset[qty][j]=tmp
        return
    
    def get_mean_end(self, alist, i0=0):
        retlist = []
        if i0==0:
            i0 = self.cutoff
        for i, ai in enumerate(alist):
            retlist.append(float(np.mean(ai[-i0:])))
        return retlist
    
    def get_var_end(self, alist):
        retlist = []
        for i, ai in enumerate(alist):
            retlist.append(float(np.std(ai[-self.cutoff:])))
        return retlist
    
    def get_mat_rate(self, end=-1):
        mr_tmp = []
        for el in self.ea.dataset["e"]:
            mr_tmp.append(float(np.mean(diff(el)[-self.cutoff:end])))
        return mr_tmp
        
    def collect_data(self, prm, finished=True):
        if finished:
            #self.dataset["mat_rate"].append(self.get_mat_rate())

            self.dataset["var"].append(self.get_mean_end(self.ea.dataset["e_var"]))
            self.dataset["eab"].append(self.get_mean_end(self.ea.dataset["eab"], 1))
            self.dataset["eab_var"].append(self.get_mean_end(self.ea.dataset["eab_var"]))

            self.dataset["eta"].append(self.get_mean_end(self.ea.dataset["eta"]))

            self.dataset["dE"].append(self.get_mean_end(self.ea.dataset["de"]))

            self.dataset["Eb"].append(self.get_mean_end(self.ea.dataset["e"], 1))
            self.dataset["tau"].append(self.get_mean_end(self.ea.dataset["tau"], 1))
            self.dataset["xb"].append(self.get_mean_end(self.ea.dataset["xb"], 1))

            self.dataset["cov"].append(self.get_mean_end(self.ea.dataset["cov"]))

            self.dataset["fit"].append(self.get_mean_end(self.ea.dataset["w"]))

            self.dataset["N"].append(self.get_mean_end(self.ea.dataset["n"]))
            

            #self.dataset["sens"].append(self.get_mean_end(self.ea.dataset["sens"]))

            
        else:
            #for qty in ["mat_rate", "var", "eab", "eab_var", "eta", "dE", "Eb", "cov", "fit", "N", "sens" ]:
            #    self.dataset[qty].append(np.nan)
            self.dataset["mat_rate"].append(np.nan)

            self.dataset["var"].append(self.get_mean_end(self.ea.dataset["e_var"], 1))
            self.dataset["eab"].append(self.get_mean_end(self.ea.dataset["eab"], 3))
            self.dataset["eab_var"].append(self.get_mean_end(self.ea.dataset["eab_var"], 3))

            self.dataset["eta"].append(self.get_mean_end(self.ea.dataset["eta"], 3))

            self.dataset["dE"].append(self.get_mean_end(self.ea.dataset["de"], 3))

            self.dataset["Eb"].append(self.get_mean_end(self.ea.dataset["e"], 3))

            self.dataset["cov"].append(self.get_mean_end(self.ea.dataset["cov"], 3))

            self.dataset["fit"].append(self.get_mean_end(self.ea.dataset["w"], 3))

            self.dataset["N"].append(self.get_mean_end(self.ea.dataset["n"], 3))

            self.dataset["sens"].append(self.get_mean_end(self.ea.dataset["sens"], 3))

            
            
        self.dataset["prms"].append(float(prm))
        self.dataset["surv_prob"].append(self.ea.surv_prob)
        return
    
    def get_mean(self, qty):
        mean = []
        for xi in self.dataset[qty]:
            mean.append(np.nanmean(xi))
        return np.asarray(mean)
    
    def get_std(self, qty):
        std = []
        for xi in self.dataset[qty]:
            std.append(np.nanstd(xi))
        return std
    
    def get_deltaE(self):
        dEList = []
        for i in range(len(self.dataset["prms"])):
            dEi = []
            for j in range(len(self.dataset["cov"][i])):
                covi = self.dataset["cov"][i][j]
                fiti = self.dataset["fit"][i][j]
                dEi.append(covi/fiti)
            dEList.append(np.mean(dEi))
        return dEList
    
    def get_deltaE_mean(self):
        """
        taking ensemble average after calculating the sens
        """
        dElist = []
        for i in range(len(self.dataset["prms"])):
            dEi = []
            for j in range(len(self.dataset["var"][i])):
                var = self.dataset["var"][i][j]
                sens = self.dataset["sens"][i][j]
                dEi.append(var*sens)
            dElist.append(np.mean(dEi))
        return dElist
    
    def get_mean_sensitivity(self):
        """
        taking ensemble average before calcuating the sens
        """
        var = self.get_mean("var")
        eta = self.get_mean("eta")
        Eb = self.get_mean("Eb")
        pa = self.ea.prm["pa"]
        pd = self.ea.prm["pd"]
        eta0 = self.ea.prm["eta0"]
        xb2 = self.ea.prm["xb2"]
        if self.dataset["prm_name"]== "f":
            fx = np.asarray(self.dataset["prms"])
        else:
            fx = self.ea.prm["f"]
        Ef = fx*xb2/(2*Eb*4.012)
        dw = (pa+pd-pa*pd)*eta0*(1-eta)/(eta0+eta)*(1-1.5/Eb - Ef**2-(Ef/Eb)/(1-Ef))
        return dw, var*dw
    
        
    def plotMR(self, ax=None, filename="", ylim=(0, 0.05), plot_price=True, plot_price_app=False):
        if not ax:
            fig, ax = plt.subplots(figsize=(4,3.7), dpi=100)
            ax.set_xlabel("Ab delay time, Td", fontsize=13)
            ax.set_ylabel("maturation rate", fontsize=13)
            ax.set_ylim(ylim)
        plt.errorbar(self.dataset["prms"], self.dataset["mat_rate"], yerr=self.dataset["mat_rate_std"],
                    fmt="-o",capsize=4, label="simulation"
                    )
        if plot_price:
            var = np.asarray(self.dataset["std"])
            eta = np.asarray(self.dataset["eta"])
            Eb = np.asarray(self.dataset["Eb"])
            
            if self.dataset["prm_name"] =="f":
                self.f = np.asarray(self.dataset["prms"])
            if plot_price_app:
                plt.plot(self.dataset["prms"], var*(0.5/(0.5+eta))*(1-eta)*(1-self.f*2.0/(2*4.012*Eb)-(1/(2*Eb))/(1-self.f*2.0/(2*4.012*Eb))), '-o', fillstyle="none",  color="r", label="price Eq.")
            else:
                if "cov" in self.qty_name:
                    cov = np.asarray(self.dataset["cov"])
                    plt.plot(self.dataset["prms"], cov/(eta/(0.5+eta)), '-o', fillstyle="none",  color="r", label="price Eq.")
        plt.tight_layout()
        plt.legend(frameon=False, fontsize=13)

        if filename:
            plt.savefig("figs/aff_dis/"+filename+".pdf", format='pdf')
        return ax

    def plotQtyMean(self, qty, ax=None,cr='r', errorbar = True, filename="", ylim=(0, 1), fmt='-o'):
        if not ax:
            fig, ax = plt.subplots(figsize=(3,2.7), dpi=100)
            ax.set_xlabel(self.dataset["prm_name"], fontsize=13)
            ax.set_ylabel(qty, fontsize=13)
            ax.set_ylim(ylim)
        mean = self.get_mean(qty)
        std = self.get_std(qty)
        if errorbar:
            ax.errorbar(self.dataset["prms"], mean, yerr=std,
                        fmt=fmt,capsize=4, fillstyle='none', color=cr
                        )
        else:
            ax.plot(self.dataset["prms"], mean, fmt, fillstyle="none", color=cr)
        if filename:
            plt.tight_layout()
            plt.savefig("figs/aff_dis/"+filename+".pdf", format='pdf')
        return ax
    
    def save_prm(self, filename):
        v = 0
        prm = self.ea.prm
        while os.path.exists(filename):
            print("file exists! ")
            filename += gen_pi(v)
            v += 1

        with open(filename+"_prm.json", "w") as myfile:
            json.dump(prm, myfile)
        return
    
    def load_prm(self, filename):
        with open(filename+'.json', 'r') as fp:
            prm = json.load(fp)
        print_dict(prm)
        return prm
    
    

    

def convert_to_week(tm):
    try:
        return tm*week_per_cycle
    except:
        tmp = []
        for ti in tm:
            tmp.append(ti*week_per_cycle)
        return tmp

def convert_to_affinity(ei):
    return np.exp(np.asarray(ei))/np.exp(14)
#################################################################


def write(s, path=""):
    if path == "":
        print(s)
    else:
        t = "\n" + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ":  " + s
        with open(path, 'a') as log:
            log.write(t)
    return

    
def diff(a, step=1):
    return [(a[i+step]-a[i])/step for i in range(len(a)-step)]

def my_cov(a, b):
    if len(a) != len(b):
        write("Error, cov array length doesn't match")
        raise ValueError()
    if len(a)>0:
        return np.inner(a, b)/len(a)-np.mean(a)*np.mean(b)
    else:
        return None

def get_mean(a):
    b = []
    lengthlist = [len(ai) for ai in a]
    if len(lengthlist)==0:
        return 0
    length = min(lengthlist)
    n = len(a)
    for j in range(length):
        s = 0
        for i in range(n):
            s += a[i][j]
        b.append(s/n)
    return np.asarray(b)

def get_std(a):
    b = []
    lengthlist = [len(ai) for ai in a]
    if len(lengthlist)==0:
        return 0
    length = min(lengthlist)
    n = len(a)
    for j in range(length):
        s = 0
        b.append(np.std([a[k][j] for k in range(n)]))
    return np.asarray(b)

def setAxWidth(ax, frame_width):
    ax.spines['top'].set_linewidth(frame_width)
    ax.spines['right'].set_linewidth(frame_width)
    ax.spines['bottom'].set_linewidth(frame_width)
    ax.spines['left'].set_linewidth(frame_width)
    return
            
     
def getAx(xlabel="time", ylabel="quantity", fontsize=15, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    setAxWidth(ax, 1.5)
    return ax


def extRatio_linear_cubic(xb1, xb2, e10, e20, f0, output=True, Eb_min=3, pre_factor=True):
    """
    xb1: APC-Ag bond length, nm
    xb2: BCR-Ag bond length, nm
    e10: APC-Ag affinity, kT
    e20: BCR-Ag affinity, kT
    f: force, pN
    """
    kT = 4.012
    ## Eq.30, Eq.31, eq.32 in si
    fa = 3*e10*kT/(2*xb1) ## pN
    fb = 3*e20*kT/(2*xb2) ## pN
    if f0>fa or f0>fb:
        return np.nan, False, -1
    
    ka = 4*(fa/xb1)*np.sqrt(1-f0/fa)*0.001 ## nN/nm
    kb = 4*(fb/xb2)*np.sqrt(1-f0/fb)*0.001 ## nN/nm
    
    ga=gb=1
    
    ## barrier height
    Eb1 = e10*(1-f0/fa)**(3/2)
    Eb2 = e20*(1-f0/fb)**(3/2)
    
    ## prefactor
    tau0_a = (ga*gb/(2*ka*kb))*((kb-ka)/ga + kb/gb + np.sqrt(((kb-ka)/ga + kb/gb)**2+4*ka*kb/(ga*gb)))
    tau0_b = (ga*gb/(2*ka*kb))*((-kb+ka)/ga - kb/gb + np.sqrt(((-kb+ka)/ga - kb/gb)**2+4*ka*kb/(ga*gb)))
    
    if not pre_factor:
        tau0_a = tau0_b = 1
    if Eb1<Eb_min or Eb2<Eb_min:
        return 1/(1+tau0_a*np.exp(Eb1-Eb2)/tau0_b), False, min(Eb1, Eb2)
    return 1/(1+tau0_a*np.exp(Eb1-Eb2)/tau0_b), True, min(Eb1, Eb2)


def lifetime_linear_cubic(eb, xb, f0=0, output=False, Eb_min=None, pre_factor=True):
    kT = 4.012
    const = 2*3.1415
    if f0==0:
        fb = 3*eb*kT/(2*xb)
        k = 4*(fb/xb)*0.001 ## nN/nm
        if not pre_factor:
            const = 1
            k=1
        return const*np.exp(eb)/k
    else:
        fb = 3*eb*kT/(2*xb)
        if Eb_min is not None and f0>fb:
            return np.nan
        Ef = eb*(1-f0/fb)**(3/2)
        
        if output:
            print("     fb=", fb, ", f=", f0)
            print("     Ef=", Ef, "kT")
        if Eb_min is not None and Ef<Eb_min:
            return np.nan
        k = 4*(fb/xb)*0.001 * np.sqrt(1 - f0/fb)
        if not pre_factor:
            const = 1
            k=1
        return const*np.exp(Ef) / k

def extRatio(xb1, xb2, e10, e20, f0, m0=1.0, output=True):
    ## parameters
    m = m0
    Mg = 1.0   ## M*gma
    T = 300   # temperature
    kB = 1.38E-23   # Boltzmann constant
    
    ## bond parameters
    E1 = e10*kB*T  ## CR-Ag bond binding energy, in kT
    
    k10 = 2*E1/(xb1*1.0E-9)**2
    k1 = k10       ## CR-Ag bond stiffness, nN/nm
    xb = np.sqrt(2*E1/k1)*1.0E9   ## CR-Ag bond maximum deformation
    
    
    E2 = e20*kB*T  ## BCR-Ag bond binding energy, in kT
    k20 = 2*E2/(xb2*1.0E-9)**2
    k2 = k20       ## BCR-Ag bond stiffness, nN/nm
    zb = np.sqrt(2*E2/k2)*1.0E9  ## BCR-Ag bond range, nN/nm
    
    
    f = f0*1.0E-12   ## convert F from pN to N
    
    
    f1 = np.sqrt(2.0*E1*k1)
    f2 = np.sqrt(2.0*E2*k2)
    
    Eb1 = E1*(1-f/f1)**2   ## energy barrier
    Eb2 = E2*(1-f/f2)**2   ## 

    p = np.exp((Eb1-Eb2)/(kB*T))
    q = (1+m)*np.sqrt(k2/k1)*(f2-f)/(f1-f)

    if f<f1*(1.0-2.0/np.sqrt(E1/(kB*T))) and f<f2*(1.0-2.0/np.sqrt(E2/(kB*T))):
        return 1/(1+q*p), True
    elif f>1*f1 or f>1*f2:
        #write("---- Error: energy barrier vanishes ----")
        if output:
            print("*", end="")
        return 1/(1+q*p), False
    else:
        #write("---- Warrining: energy barrier too small ----")
        if output:
            print("!", end="")
        return 1/(1+q*p), False
    return None, False

## Artifitial linear dependence
def extRatio_linear(e10, e20, dE, DE):
    y = (e20-e10-dE)/DE+0.5
    if y<0:
        return 0
    elif y>1:
        return 1
    else:
        return y
    return


## time dependent force
def eta0(k1,k2,E1,E2, f0=0, m=1):
    ## extraction without external force
    f1 = np.sqrt(2.0*E1*k1*kT)
    f2 = np.sqrt(2.0*E2*k2*kT)
    f = f0*1.0E-12
    if f>f1*(1.0-1.0/np.sqrt(E1)) or f>f2*(1.0-1.0/np.sqrt(E2)):
        approxWarn()
    dE = E1*(1-f/f1)**2-E2*(1-f/f2)**2
    tmp = (1+m)*np.sqrt(k2/k1)*np.exp(dE)*(f2-f)/(f1-f)
    return 1.0/(1.0+tmp)


def tauCA(f0=0, E1=12, k1=0.2):
    ## bond lifetime of APC-Ag bond
    Mg = 1.0
    f1 = np.sqrt(2*k1*E1*kT)
    f = f0*1.0E-12
    
    part1 = 2*Mg*np.sqrt(PI)/k1
    part2 = np.exp(E1*(1-f/f1)**2)/(np.sqrt(E1)*(1-f/f1))
    return part1*part2
    
def Sab2(k2, E2, fi0, ft0, r0, m=1.0):
    ## survival probability for Ag-BCR bond
    Mg = 1.0   ## M*gma 
    ## bond parameters
    f2 = np.sqrt(2.0*E2*k2*kT)
    
    fi = fi0*1.0E-12
    ft = ft0*1.0E-12
    r = r0*1.0E-12
    if fi0==ft0 or r0==0:
        return 1.0
    part1 = -(1+m)*np.sqrt(2*k2**3*kT/PI)/(4*r*Mg)
    part2 = np.exp(-E2*(1-ft/f2)**2)-np.exp(-E2*(1-fi/f2)**2)
    return np.exp(part1*part2)

def Sca2(k1, E1, fi0, ft0, r0, m=1.0):
    ## survical probability for CR-Ag bond
    Mg = 1.0   ## M*gma 
    ## bond parameters
    f1 = np.sqrt(2.0*E1*k1*kT)
    
    fi = fi0*1.0E-12
    ft = ft0*1.0E-12
    r = r0*1.0E-12
    if fi0==ft0 or r0==0:
        return 1.0
    part1 = -np.sqrt(2*k1**3*kT/PI)/(4*r*Mg)
    part2 = np.exp(-E1*(1-ft/f1)**2)-np.exp(-E1*(1-fi/f1)**2)
    return np.exp(part1*part2)

def extRatioF(k10, k20, E10, E20, f0, fm=50, r=0.001):
    ## E10 and E20 are in kT unit
    ## return extraction ratio under increasing force
    
    ## parameters
    Mg = 1.0   ## M*gma
    
    ## bond parameters
    k2 = k20   ## bond stiffness, nN/nm
    k1 = k10
    E2 = E20
    E1 = E10
    combine = lambda f: Sab2(k2, E2, f0, f, r)*Sca2(k1, E1, f0, f, r)/(r*tauCA(f, E1, k1))
    if f0 == fm or r==0:
        ret1 = 0
    else:
        ret1 = integrate.quad(combine, f0, fm)[0]
    ret2 = eta0(k1,k2,E1,E2,fm)*Sab2(k2, E2, f0, fm, r)*Sca2(k1, E1, f0, fm, r)
    return ret1+ret2



### extraction ratio of periodic force
def tau(k1, k2, E1, E2, f0, m0=1.0):
    ## get mean first passage times
    m = m0    ## m/M
    Mg = 1.0   ## M*gma

    f = 1.0e-12*f0  ## convert F from pN to nN
    dt = 1.0
    
    f1 = np.sqrt(2.0*k1*E1*kT)
    f2 = np.sqrt(2.0*k2*E2*kT)
    
    if f>f1*(1.0-1.0/np.sqrt(E1)) or f>f2*(1.0-1.0/np.sqrt(E2)):
        approxWarn()
        
    ## calculate tau_ca
    part1 = 2*m*Mg*np.sqrt(PI)/k1
    part2 = np.exp(E1*(1-f/f1)**2)/(np.sqrt(E1)*(1-f/f1))
    tau_ca = part1*part2
    
    ## calculate tau_ab
    #part1 = (m*Mg)*np.sqrt(3.1416*ky)/(2*(1+m)*kx*np.sqrt(kx+ky))
    part1 = 2*(m*Mg)*np.sqrt(PI)/(k2*(1+m))
    part2 = np.exp(E2*(1-f/f2)**2)/(np.sqrt(E2)*(1-f/f2))
    tau_ab = part1*part2
    p=1.0/(1.0+tau_ca/tau_ab)
    return [tau_ca, tau_ab, p]


def extRatioT(k10, k20, E10, E20, f0, tH=100, tL=100):
    k1 = k10
    k2 = k20
    E1 = E10
    E2 = E20
    f = f0*1.0E-12
    
    tHca, tHab, pH = tau(k1,k2,E1,E2,f0)
    tLca, tLab, pL = tau(k1,k2,E1,E2,0)
    alpha = (1.0-np.exp(-tH*(1.0/tHca+1.0/tHab)))/(1.0-np.exp(-(tH/tHca+tH/tHab+tL/tLca+tL/tLab)))
    return alpha*pH+(1-alpha)*pL


def extRatio_bell(xb1, xb2, E1, E2, f):
    ## E1, E2 in kbT
    ## f in pN
    ## xb1 xb2 in nm
    
    
    
    tau1 = np.exp(E1-f*xb1*1.0E-21/(kT))
    tau2 = np.exp(E2-f*xb2*1.0E-21/(kT))
    return 1.0/(1.0+tau1/tau2)

def my_round(x, tol=0.1):
    return int(round(x/tol))

from termcolor import colored

def print_dict(dct, ncol=1):
    print("Items held:")
    if ncol==1:
        for item, amount in dct.items():  # dct.iteritems() in Python 2
            print("{}:\t {}".format(item, amount))
    else:
        idx = 0
        end = ["\t\t"]*(ncol-1) + ["\n"]
        max_length = max([len(k) for k in dct.keys()])
        fmt = "%"+str(max_length)+"s"
        for item, val in dct.items():
            print(fmt % item, end=": \t")
            print(colored("{}".format(val), "magenta"), end=end[idx % ncol])
            idx += 1
            
            
    return
        
def gen_pi(n):
    if n==0:
        return '_3'
    elif n==1:
        return '.1'
    else:
        return str(mp.pi)[n+1]
    
def save_prm(prm, filename):
    v = 0
    while os.path.exists(filename):
        print("file exists! ")
        filename += gen_pi(v)
        v += 1
        
    with open(filename+".txt", "w") as myfile:
        for item, amount in prm.items():  # dct.iteritems() in Python 2
            myfile.write("{}: {}".format(item, amount))
    return


def dump(self, dataset, filename, unique=True, mod='w'):
    """
    save data to file
    """

    if unique:
        ### create uniq filename
        v = 0
        while os.path.exists(filename+'.json'):
            print("file exists! ")
            filename += gen_pi(v)
            v += 1

    with open(filename+'.json', mod) as fp:
        json.dump(dataset, fp)
    return filename