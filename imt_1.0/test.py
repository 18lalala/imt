import torch
import torch.nn as nn
import numpy as np

a = np.loadtxt('pose_aa.txt')
print(np.max(a[:,3]))