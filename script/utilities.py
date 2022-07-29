
'''

useful functions


'''

import numpy as np
import json
from mpmath import mp
import os


def load_prm(filename):
    ### load the json
    with open(filename, 'r') as fp:
        prm = json.load(fp)
    return prm


def convert_dict_to_string(my_dict):
    info = "{\n"
    for key, value in my_dict.items():
        info += "\t"
        info += key
        info += ": " + str(value)
        info += "\n"
    info += "\n}"
    return info


def gen_pi(n):
    if n==0:
        return '_3'
    elif n==1:
        return '.1'
    else:
        return str(mp.pi)[n+1]
    
def printProgress(n, N):
    percent = int(100.0*n/N)
    toPrint = "progress: "
    for i in range(percent//5):
        toPrint += '|'
    toPrint += "{:d}%    ".format(percent)
    print(toPrint, end='\r')
    return


def dump(dataset, filename, unique=True, mod='w', multiple_dict=False, sep=False):
    """
    save data to file
    """

    if unique:
        ### create uniq filename
        v = 0
        while os.path.exists(filename+'.txt'):
            print("file exists! ")
            filename += gen_pi(v)
            v += 1
            
    if not multiple_dict: ### only one dict
        with open(filename+'.txt', mod) as fp:
            fp.write(json.dumps(dataset, indent=4))
            if sep:
                fp.write("\n\n")
                fp.write("*"*20)
                fp.write("\n\n")
    else:
        data_saved = []
        if mod=='a' and os.path.exists(filename+'.txt'):
            with open(filename +'.txt', 'r') as fp:
                data_saved = json.load(fp)
        data_saved.append(dataset)
        with open(filename+'.txt', mod) as fp:
            fp.write(json.dumps(data_saved, indent=4))
    return filename
