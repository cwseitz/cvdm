import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Sim/4x/Sim-2/eval_data/'
set_metrics_100 = np.load(path + 'N100-set.npz')['metrics']
print(set_metrics_100[:,:,:,0])



