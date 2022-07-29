
import logging
import termcolor
import json
import os
import copy
from datetime import datetime
from . import data as data
from .utilities import *

class Controller:
    '''
    control class
    '''
    def __init__(self, prm={}):
        self.config = {
            "output": True,
            "write_log": False,
            "save": False,
            "test": False,
            "dir":"."
        }
        self.saved_qty = []
        self.summary_qty = []
        
        self.prm = copy.deepcopy(prm)
        self.tag = "controller"
        self.color = "grey"
        
    def build(self):
        self.dataset = data.MyData(self.saved_qty)
        self.save = self.get_config("save")
        self.output = self.get_config("output")
        self.test = self.get_config("test")
        self.write_log = self.get_config("write_log")
        self.check_file = self.get_config("dir")+"/check"
        self.data_file = self.get_config("dir")+"/data"
        self.log_file = self.get_config("dir") + "/log"
        self.temp_data_file = self.data_file+"_temp"
        
        self.summary_file = self.get_config("dir")+"/summary"
        
        
        self.info = ""
        pass
        
    def run(self):
        pass
    
    def set_prm(self, prm_name="", prm_value=None):
        pass
    
    def set_summary_qty(self, qtys):
        self.summary_qty = qtys
        return
    
    
    def clean(self):
        del self.dataset
        pass
    
    
    def get_info(self):
        self.print2("="*12+"printing info:"+"="*12)
        self.print2("config info:")
        self.print2(convert_dict_to_string(self.config))
        self.print2("prm info:")
        self.print2(convert_dict_to_string(self.prm))
        
        
    def close(self):
        if self.save:
            try:
                filename = self.dataset.dump(self.data_file)
                self.print2("dump to "+filename)
            except:
                self.print2("no data_file found, auto dump instead")
                self.auto_dump()
        self.clean()
        self.delete_check()
        self.print2("controller has been closed.....")
        return
        
    def set_config(self, name, value):
        if name in self.config:
            self.config[name] = value
        else:
            self.print2(name+" is not in the config dict")
        self.build()
        return
    
    def load_config(self, config):
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
        self.build()
        
        
    def update_config(self, config):
        for key, value in config.items():
            self.config[key] = value
        self.build()
        
    def get_data(self):
        return self.dataset
        
    def get_config(self, name):
        if name in self.config:
            return self.config[name]
        else:
            self.print2(name+ " not found in config")
        return
    
    def save_prm(self, filename=""):
        v = 0
        prm  = self.prm
        while os.path.exists(filename+"_prm_json"):
            self.print2("!! saving parameters, file exists! ")
            filename += gen_pi(v)
            v += 1

        with open(filename+"_prm.json", "w") as myfile:
            json.dump(prm, myfile)
            
        self.print2("saved prm to file: "+filename)
        return
    
    
    def auto_dump(self, filename=""):
        curr_time = filename+datetime.now().strftime("%m-%d-%y_%H-%M")
        self.dataset.dump("data/auto_saved/"+curr_time)
        self.save_prm("data/auto_saved/"+curr_time)
        self.print2("save to:" + curr_time)
        return curr_time
    
    def save_to_temp(self):
        temp_file = self.dataset.dump(filename=self.temp_data_file, unique=False)
        self.print2("temp save to: "+temp_file)
        return
    
    def check(self, title=""):
        if os.path.exists(self.check_file):
            my_check = load_prm(self.check_file)
            if my_check["stop"]:
                self.print2(title+"--stop request received. Stop calculation")
                return False
            return True
        else:
            my_check = {
                "stop": False
            }
            with open(self.check_file, "w") as myfile:
                json.dump(my_check, myfile)
            self.print2("create check file:"+self.check_file)
            return True
        
    def delete_check(self):
        
        if os.path.exists(self.check_file):
            os.remove(self.check_file)
        self.print2("check file has been deleted")
        return
        
        
    def print2(self, info, end="\n", tag=True):
        if tag:
            info =  "["+self.tag+"]:: "+info
        if self.output:
            print(termcolor.colored(info, self.color), end=end)
        if self.write_log:
            self.info += info
            if end  != "\n":
                self.info += end
            else:
                logging.info(self.info)
                self.info =""
                
    def summary(self, info):
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'a') as myfile:
                myfile.write(info + "\n")
        else:
            with open(self.summary_file, 'w') as myfile:
                myfile.write(info + "\n")
        return
                
                
                
def load_prm(filename):
    ### load the json
    with open(filename, 'r') as fp:
        prm = json.load(fp)
    return prm

def gen_pi(n):
    if n==0:
        return '_3'
    elif n==1:
        return '.1'
    else:
        return str(mp.pi)[n+1]