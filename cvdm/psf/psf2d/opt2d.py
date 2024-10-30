import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from .psf2d import *
from .ill2d import isologlike2d
from .metropolis import *
import sys
import os

class IsoLogLikelihood:
    def __init__(self,func):
        self.func = func
    def __call__(self,theta,adu):
        return self.func(theta,adu)

class MLE2D_MCMC:
    def __init__(self,theta0,adu,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
                                    
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        if self.theta_gt is not None:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        if self.theta_gt is not None:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        if self.theta_gt is not None:
            ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        ax[3].plot(thetat[:,3])
        if self.theta_gt is not None:
            ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[3].set_xlabel('Iteration')
        ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
        
    def metropolis(self,theta0,iters=3000,tburn=300,prop_cov=0.05,beta=0.02,diag=False):
        mean = np.zeros((3,)); cov = prop_cov*np.eye(3)
        loglike = IsoLogLikelihood(isologlike2d)
        sampler = Metropolis2D(mean,cov,loglike)
        samples = sampler.run(self.adu,theta0,iters=iters,beta=beta,diag=diag)        
        return samples

class MLE2D:
    def __init__(self,theta0,adu,config,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.config = config
       self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]
                            
    def show(self,theta0,theta):
        fig,ax = plt.subplots(figsize=(4,4))
        ax.imshow(self.adu,cmap='gray')
        ax.scatter(theta0[1],theta0[0],color='red',label='init')
        ax.scatter(theta[1],theta[0],color='blue',label='fit')
        ax.scatter(self.theta_gt[1],self.theta_gt[0],color='green',label='true')
        ax.invert_yaxis()
        ax.legend()
        plt.tight_layout()
        
    def plot(self,thetat,iters):
        fig,ax = plt.subplots(1,4,figsize=(10,2))
        ax[0].plot(thetat[:,0])
        if self.theta_gt is not None:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].plot(thetat[:,1])
        if self.theta_gt is not None:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].plot(thetat[:,2])
        if self.theta_gt is not None:
            ax[2].hlines(y=self.theta_gt[2],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$\sigma$')
        #ax[3].plot(thetat[:,3])
        #if self.theta_gt is not None:
        #    ax[3].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        #ax[3].set_xlabel('Iteration')
        #ax[3].set_ylabel(r'$N_{0}$')
        plt.tight_layout()
                
    def optimize(self,max_iters=1000,lr=None,plot_fit=False,tol=1e-8):
        if plot_fit:
           thetat = []
        if lr is None:
           lr = np.array([0.001,0.001,0])
        loglike = []
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        niters = 0
        converged = False
        while niters < max_iters:
            niters += 1
            loglike.append(isologlike2d(theta,self.adu,self.cmos_params))
            jac = jaciso2d(theta,self.adu,self.cmos_params)
            theta = theta - lr*jac
            if plot_fit:
                thetat.append(theta)
            dd = lr[:-1]*jac[:-1]
            if np.all(np.abs(dd) < tol):
                converged = True
                break
        if plot_fit:
            self.plot(np.array(thetat),niters)
            #self.show(theta,self.theta0)
            plt.show()


        return theta, loglike, converged

class MLE2D_BFGS:
    def __init__(self,theta0,adu,config,theta_gt=None):
       self.theta0 = theta0
       self.theta_gt = theta_gt
       self.adu = adu
       self.cmos_params = [config['eta'],config['texp'],
                            np.load(config['gain'])['arr_0'],
                            np.load(config['offset'])['arr_0'],
                            np.load(config['var'])['arr_0']]

    def show(self,theta0,theta):
        fig,ax = plt.subplots(figsize=(4,4))
        ax.scatter(theta0[1],theta0[0],color='red',label='init')
        ax.scatter(theta[1],theta[0],color='blue',label='fit')
        if self.theta_gt:
            ax.scatter(self.theta_gt[1],self.theta_gt[0],color='green',label='true')
        ax.invert_yaxis()
        ax.imshow(self.adu,cmap='gray')
        ax.legend()
        plt.tight_layout()

    def plot(self,thetat,iters,conv,jac):
        fig,ax = plt.subplots(1,3,figsize=(8,2))
        ax[0].set_title(f'j_x0={np.round(jac[0],2)}')
        ax[0].plot(thetat[:,0])
        if self.theta_gt is not None:
            ax[0].hlines(y=self.theta_gt[0],xmin=0,xmax=iters,color='red')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('x')
        ax[1].set_title(f'j_y0={np.round(jac[1],2)}')
        ax[1].plot(thetat[:,1])
        if self.theta_gt is not None:
            ax[1].hlines(y=self.theta_gt[1],xmin=0,xmax=iters,color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('y')
        ax[2].set_title(f'j_N0={np.round(jac[2],2)}')
        ax[2].plot(thetat[:,2])
        if self.theta_gt is not None:
            ax[2].hlines(y=self.theta_gt[3],xmin=0,xmax=iters,color='red')
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel(r'$N_{0}$')
        plt.suptitle(f'converged={conv}')
        plt.tight_layout()

    def optimize(self, max_iters=1000, lr=None, plot_fit=False):
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        thetat = []; loglikes = []

        def objective_function(theta):
            loglike_value = isologlike2d(theta,self.adu,self.cmos_params)
            loglikes.append(loglike_value)
            return loglike_value

        def jacobian_function(theta):
            jac = jaciso2d(theta, self.adu, self.cmos_params)
            return jac

        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        result = minimize(
            objective_function,
            theta,
            method='BFGS',
            jac=jacobian_function,
            options={'maxiter': max_iters, 'disp': False},
            callback=lambda xk: thetat.append(xk.copy()),
        )
        sys.stderr = original_stderr
        theta_mle = result.x
        converged = result.success
        niters = result.nit
        if converged:
            err = np.sqrt(np.diag(result.hess_inv))
        else:
            err = [np.inf,np.inf,np.inf]

        if plot_fit:
            self.show(self.theta0,theta_mle)
            self.plot(np.array(thetat), niters, converged, result.jac)
            plt.show()
            
        return theta_mle,loglikes,converged,err
        

