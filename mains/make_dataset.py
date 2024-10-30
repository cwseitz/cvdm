from make.dataset import *
import numpy as np
import os

shape_lr = 64
shape_hr = 64
savepath = '/research3/shared/cwseitz/Data/CVDM/4x/Sim/eval_data/'
os.makedirs(savepath,exist_ok=True)
#os.makedirs(savepath+f'lr_{shape_lr}',exist_ok=True)
#os.makedirs(savepath+f'hr_{shape_hr}',exist_ok=True)

radius = 100.0
nspots = 1000
nsamples = 500
sigma_kde = 1.0
args = [radius,nspots]

kwargs = {
'N0':100,
'B0':0,
'eta':1.0,
'sigma':1.0,
"gain": 1.0,
"offset": 100.0,
"var": 100.0
}

prefix = f'Sim_CMOS-{shape_lr}-'
prefix += str(nspots)
prefix += f'-{kwargs["N0"]}'
prefix += f'-{kwargs["B0"]}'
prefix += f'-{str(kwargs["sigma"]).replace(".","p")}'
prefix += f'-{str(sigma_kde).replace(".","p")}'
print(prefix)

generator = Disc2D(shape_lr,shape_lr)
dataset = TrainDataset(nsamples)

X,Z,S,thetas = dataset.make_dataset(generator,args,kwargs,
                             show=False,upsample=4,
                             sigma_kde=sigma_kde)
                             
imsave(savepath+'lr.tif',X)
imsave(savepath+f'hr.tif',Z)
for n,theta in enumerate(thetas):
    np.savez(savepath+f'coords-{n}.npz',theta=theta)

#for n in range(nsamples):
#    imsave(savepath+f'lr_{shape_lr}/'+prefix+f'_lr-{n}.tif',X[n])
#    imsave(savepath+f'hr_{shape_hr}/'+prefix+f'_hr-{n}.tif',Z[n])

