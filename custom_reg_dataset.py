import os
import torch
import numpy as np
from torch.utils.data import Dataset

class RegisterDataset(Dataset):
    def __init__(self, data, label):
          
        x=data.values
        y=label.values
          
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.y = self.y.type(torch.LongTensor)
    
    def __len__(self):
        return len(self.y)
     
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]