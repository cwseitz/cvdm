import numpy as np
import matplotlib.pyplot as plt
from cvdm.psf import crlb2d

class CRB2D:
    def __init__(self):
        pass   
    def forward(self,N0space,B0=0):
        cam_params = [1.0,1.0,1.0,0.0,100.0]
        crlb_N0 = np.zeros((len(N0space),3))
        for i,N0 in enumerate(N0space):
            theta0 = np.array([5,5,N0])
            crlb_N0[i] = crlb2d(theta0,cam_params,B0=B0)
        return crlb_N0
        

pixel_size=100
fig,ax=plt.subplots(figsize=(3.2,3.2))
N0space = np.linspace(100,1000,500)
crb = CRB2D()
crlb_N0 = crb.forward(N0space,B0=0)
ax.plot(N0space,crlb_N0[:,0]*pixel_size,
          color='gray',linestyle='--',label=r'$\sigma_{CRLB}$')

ax.set_xscale('log')
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Intensity (photons)')
ax.set_ylabel('Localization uncertainty (nm)')
ax.legend(fontsize=8,frameon=False)
plt.tight_layout()
plt.show()
