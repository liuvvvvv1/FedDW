import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
import queue


class Client:
    def __init__(self,config,idx):
        self.config=config
        self.idx=idx
        self.num_data=idx.shape[0]
        #by moon
        self.pre_nets=queue.Queue(maxsize=self.config.max_pre_net_num)
        #by pFL
        self.model=None
        #by FedALA
        self.weights=None
        self.start_phase=True



        pass
    pass




class Server:
    def __init__(self,config):
        self.config=config
        pass
    pass

