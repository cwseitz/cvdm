import numpy as np
import matplotlib.pyplot as plt
from .base import *

class Uniform2D(Generator):
    def __init__(self,size):
        self.size = size
        super().__init__(size,size)
    def forward(self,nspots,sigma=0.92,texp=1.0,N0_min=500.0,N0_max=1000.0,
                eta=1.0,gain=1.0,B0=None,nframes=1,offset=100.0,var=5.0,show=False):
        density = Uniform(self.size)
        theta = np.zeros((4,nspots))
        x,y = density.sample(nspots)
        N0 = np.random.uniform(N0_min,N0_max,nspots)
        theta[0,:] = x; theta[1,:] = y
        theta[2,:] = sigma; theta[3,:] = N0
        adu,spikes = self.sample_frames(theta,nframes,texp,eta,B0,gain,offset,var,show=show)
        return adu,spikes,theta


