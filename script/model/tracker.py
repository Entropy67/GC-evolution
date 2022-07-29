#### tracker class implementation
import numpy as np

class Tracker:
    
    def __init__(self, size, qty_name, gc):
        
        self.size = size
        self.dataset = {}
        self.qty_name = qty_name
        self.gc = gc
        self.setup()
        
        
    def load_agent(self):
        self.agents = self.gc.get_agents()
        return
    
    def setup(self):
        self.load_agent()
        for qty in self.qty_name:
            self.dataset[qty] = []
        return
    
    def refresh(self, qty):
        self.load_agent()
        res = [[] for _ in range(self.size)] 
        for bcell in self.agents:
            if bcell.alive:
                res[bcell.id].append(getattr(bcell, qty))
        self.dataset[qty].append(res)
        return
    
    def refresh_all(self):
        for qty in self.qty_name:
            self.refresh(qty)
        return
    
    
    def get_mean(self, qty, idx):
        res = []
        for val in self.dataset[qty]:
            ### val = [[1, 2], [], [3, 4], ...]
            if len(val[idx]) == 0:
                res.append(np.nan)
            else:
                res.append(np.mean(val[idx]))
        return res
    
            
