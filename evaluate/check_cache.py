import os
import pickle
import numpy as np
import torch

rsFile = './cache/RandStates_sd198-20_wrn_baseline++_unified_withinput_s1_q5_w5'
_randStates = torch.load(rsFile)
print(len(_randStates))
print(_randStates[0])
