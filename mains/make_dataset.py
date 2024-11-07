from cvdm.make import *
from skimage.restoration import rolling_ball
from cvdm.utils.zoom import custom_zoom
import numpy as np
import os


savepath = '/N/slate/cwseitz/cvdm/Sim/4x/Sim-5/eval_data/'
os.makedirs(savepath,exist_ok=True)
os.makedirs(savepath+'coords',exist_ok=True)

size_lr = 64
nsamples = 6
sigma_kde = 1.0

kwargs = {
'B0':0,
'N0_min':100.0,
'N0_max':1000.0,
'eta':1.0,
'sigma':1.0,
"gain": 1.0,
"offset": 100.0,
"var": 100.0
}

generator = Uniform2D(size_lr)
dataset = TrainDataset(nsamples)

X,Z,S,thetas = dataset.make_dataset(generator,kwargs,show=False,upsample=4,
sigma_kde=sigma_kde,N_min=500.0,N_max=501.0)

for n,theta in enumerate(thetas):
    np.savez(savepath+f'coords/coords-{n}.npz',theta=theta)

imsave(savepath+'lr-1x.tif',X)
imsave(savepath+f'hr.tif',Z)
X = X-kwargs['offset']
imsave(savepath+'lr-1x-sub.tif',X)
X[X < 0.0] = 0
imsave(savepath+'lr-1x-thresh.tif',X)                         

X2x = []
for n in range(nsamples):
    print(f'Zooming frame {n}')
    X2x.append(custom_zoom(X[n]))
X2x = np.array(X2x)
X2x = X2x.astype(np.float32)
imsave(savepath+'lr-2x.tif',X2x)

X4x = []
for n in range(nsamples):
    print(f'Zooming frame {n}')
    X4x.append(custom_zoom(X2x[n]))
X4x = np.array(X4x)
X4x= X4x.astype(np.float32)
imsave(savepath+'lr.tif',X4x)
print(X4x.shape,Z.shape)



