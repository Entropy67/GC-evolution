##### data structure

import numpy as np
import json
import os
import os.path

from mpmath import mp
from datetime import datetime
import copy

#from AgExtract.utilities import *


class MyData:
    
    '''
    data structure,
    append, modify, save the data
    '''
    
    def __init__(self, qty_name=[]):
        self.qty_name = qty_name
        self.build()
        self.data_type = "list" ## list or array
        self.file_name_list = set() ## record the loaded filenames
        pass
    
    def build(self):
        self.dataset = {}
        self.length = 0
        for qty in self.qty_name:
            self.dataset[qty] = []
            
        self.dataset_deleted = {}
        for qty in self.qty_name:
            self.dataset_deleted[qty] = []
        
        self.dataset_deleted["index"] = []
        pass
        
    def clean(self):
        if self.dataset:
            for qty in self.qty_name:
                if qty in self.dataset: del self.dataset[qty]
                if qty in self.dataset_deleted: del self.dataset_deleted[qty]
        if "index" in self.dataset_deleted:
            del self.dataset_deleted["index"]
        
        self.dataset = {}
        self.dataset_deleted = {}
        return
    
    def copy(self, other_obj):
        self.dataset = other_obj.dataset.copy()
        self.qty_name = other_obj.qty_name
        return
    
    def get(self, qty, array=True):
        if qty in self.qty_name:
            data = self.dataset[qty]
        else:
            max_length = 0
            for key, value in self.dataset.items():
                if isinstance(value, list) and len(value)>max_length:
                    max_length = len(value)
            data = [np.nan]*max_length
        if array:
            return np.asarray(data)
        else:
            return data
    
    def get_mean(self,qty):
        if qty in self.qty_name:
            if len(self.dataset[qty]) == 0:
                return 0
            return np.nanmean(self.dataset[qty])
        else:
            return np.nan
    
    def get_std(self, qty):
        if qty in self.qty_name:
            if len(self.dataset[qty]) == 0:
                return 0
            return np.nanstd(self.dataset[qty])
        else:
            return np.nan
        
    def set_value(self, qty, value):
        self.dataset[qty] = value;
        return
    
    def append(self, qtys, values):
        if isinstance(qtys, list):
            for qty, v in zip(qtys, values):
                self._append(qty, v)
        else:
            self._append(qtys, values)
        return

    def dump(self, filename, unique=True, mod='w', drop_zeros=False):
        """
        save data to file
        """
        self.to_list()
        if unique:
            ### create uniq filename
            v = 0
            while os.path.exists(filename+'.json'):
                print("file exists! ")
                filename += gen_pi(v)
                v += 1
        
        tmp = {}
        if drop_zeros:
            for k, v in self.dataset.items():
                if isinstance(v, list):
                    if len(v)>0:
                        tmp[k] = v
                    else:
                        continue
                else:
                    tmp[k] = v
                    
        with open(filename+'.json', mod) as fp:
            if drop_zeros:
                fp.write(json.dumps(tmp, indent=4))
            else:
                fp.write(json.dumps(self.dataset, indent=4))
        return filename
    
    
    def dump_to_txt(self, filename, unique=False, mod='w', drop_zeros=True, title=None):
        self.to_list()
        if unique:
            ### create uniq filename
            v = 0
            while os.path.exists(filename+'.json'):
                print("file exists! ")
                filename += gen_pi(v)
                v += 1
        
        tmp = {}
        if drop_zeros:
            for k, v in self.dataset.items():
                if isinstance(v, list):
                    if len(v)>0:
                        tmp[k] = v
                    else:
                        continue
                else:
                    tmp[k] = v
                    
        else:
            tmp = self.dataset
                    
        with open(filename+'.txt', mod) as fp:
            fp.write("\n\n")
            fp.write("##"*30)
            if title is not None:
                fp.write(" "*15+title+ " "*15)
            fp.write("##"*30)
            fp.write("\n")
            for k, v in tmp.items():
                fp.write(str(k) + ":" + str(v)+"\n")
            fp.write("##"*30)
            fp.write("\n")
        return filename
        
    
    def load(self, filename, append=True):
        """
        load data from filename
        """
        ### load the json
        try:
            with open(filename, 'r') as fp:
                data_temp = json.load(fp)
            if append:
                self.eat(data_temp)
            else:
                self.qty_name = []
                self.dataset = data_temp
                for key in self.dataset.keys():
                    self.qty_name.append(copy.deepcopy(key))
                    
            if filename not in self.file_name_list:
                self.file_name_list.add(filename)
        except:
            print(filename+" not found!")
        return
    
    def reload(self, output=True):
        self.clean()
        
        for fn in self.file_name_list:
            if output:
                print("loading file: ", fn)
            self.load(fn)
        return
    
    def load_all(self, directory):
        """
        load all data in a directory starting with "data"
        """
        for file in os.listdir(os.fsencode(directory)):
            filename = os.fsdecode(file)
            if filename.startswith("data"): 
                self.load(directory+filename, True)
        
        
    def eat(self, other_data):
        #### eat other's dataset
        for key, value in other_data.items():
            if key in self.dataset:
                self.dataset[key] += value
            else:
                self.dataset[key] = value
                self.qty_name.append(key)
        return
    
    def get_data_type(self):
        for qty in self.qty_name:
            try:
                print(qty, "[0][0].tpye=", type(self.dataset[qty][0][0]))
            except:
                try:
                    print(qty, "[0].tpye=", type(self.dataset[qty][0]))
                except:
                    print(qty)
                    
    def to_list(self):
        for qty in self.qty_name:
            try:
                for i, mi in enumerate(self.dataset[qty]):
                     self.dataset[qty][i] = np.array(mi).tolist()
            except:
                self.dataset[qty] = np.array(self.dataset[qty]).tolist()
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
        if isinstance(elements, list):
            for ei in elements:
                self._remove_index(ei)
        else:
            self._remove_index(elements)
        return
    
    def _remove_index(self, index):
        for qty in self.qty_name:
            if isinstance(self.dataset[qty], list):
                if len(self.dataset[qty])>1:
                    value = self.dataset[qty].pop(index)
                    if qty in self.dataset_deleted:
                        self.dataset_deleted[qty].append(value)
                    else:
                        self.dataset_deleted[qty] = [value]
                    self.dataset_deleted["index"].append(index)
        return
    
    def _append(self, qty, value):
        '''
        append value to qty in dataset
        '''
        if qty in self.dataset:
            self.dataset[qty].append(value)
        else:
            self.qty_name.append(qty)
            self.dataset[qty] = [value]
        return
    
    def sort_by_qty(self, qty, reverse=False):
        n = len(self.dataset[qty])
        for i in range(n):
            for j in range(i+1, n):
                if reverse:
                    if self.dataset[qty][i]<self.dataset[qty][j]:
                        self._switch(i, j)
                else:
                    if self.dataset[qty][i]>self.dataset[qty][j]:
                        self._switch(i, j)
        return
                    
        
    def _switch(self, i, j):
        for qty in self.qty_name:
            if qty!="prm_name" and qty in self.dataset:
                length = len(self.dataset[qty])
                if length>0 and length>i and length>j:
                    try:
                        tmp = self.dataset[qty][i]
                        self.dataset[qty][i]=self.dataset[qty][j]
                        self.dataset[qty][j]=tmp
                    except:
                        print("error in _switch, (i, j)=(", i, j, "), qty=", qty)
        return
    
    def print2(self, info):
        print(info)
        return
    
def load_data(folder, fname, temp=True, data_name=[]):
    all_data = []
    for fn in fname:
        all_data.append(MyData())
        
        for dn in data_name:
            try:
                all_data[-1].load(filename=folder+fn+"/" + dn)
            except:
                print(folder, " no data file!")
            
        if temp:
            try:
                all_data[-1].load(filename=folder+fn+"/data_temp.json")
            except:
                print(folder, " no temp data file!")
                continue
    return all_data