'''

2d localization problem

Notation follows N. Boyd's ADCG paper 

'''
import os 
import numpy as Np
import autograd as Agrad
import autograd.numpy as np 
import scipy.optimize
from scipy.stats import norm as Norm
# --- locahbay
from locahbay import util as UT 
# -- plotting -- 
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

Ngrid = 20
xpix = np.linspace(0., 1., Ngrid) # default pixel gridding 
ypix = np.linspace(0., 1., Ngrid) 
_xxpix, _yypix = np.meshgrid(xpix, ypix) 
xxpix = _xxpix.flatten()
yypix = _yypix.flatten()
xypix = np.array([xxpix, yypix]).T

# PSF details
sig_psf = 0.02 # psf width
cov_psf = sig_psf**2 * np.identity(2) 
cinv_psf = np.linalg.inv(cov_psf) 

def psi(theta): 
    ''' measurement model (2d gaussian of width sigma PSF) written out to x,y grid
    '''
    return np.exp(-0.5 * np.array([dxy @ (cinv_psf @ dxy.T) for dxy in (xypix - theta[None,:])]))


theta_xgrid = np.linspace(0., 1., Ngrid) 
theta_ygrid = np.linspace(0., 1., Ngrid) 
_theta_xxgrid, _theta_yygrid = np.meshgrid(theta_xgrid, theta_ygrid) 
theta_xygrid = np.array([_theta_xxgrid.flatten(), _theta_yygrid.flatten()]).T
grid_psi = np.stack([psi(tt) for tt in theta_xygrid])  


def Psi(ws, thetas): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    _thetas = np.atleast_2d(thetas)
    return np.sum(np.array([w * psi(tt) for (w,tt) in zip(ws, _thetas)]),0)


def obs2d(N_source=5, sig_noise=0.2, seed=1):  
    ''' generate and write out 2d observations on a xypix. Takes positions 
    and "intensities" (weights) and convolves them with a PSF and adds noise 
    '''
    np.random.seed(seed)
    thetas = np.array([np.random.uniform(0, 1, N_source), np.random.uniform(0, 1, N_source)]).T # x_true, y_true positions 
    weights = np.repeat(10., N_source) #np.random.rand(N_source)*2 # weights --- in SMLM intensities 
    return thetas, weights, Psi(weights, thetas) + sig_noise * np.random.randn(len(xypix)) 


def test_obs2d(N_source=5, sig_noise=0.2, seed=1):  
    thetas_true, w_true, y = obs2d(N_source=N_source, sig_noise=sig_noise, seed=seed)

    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.imshow(y.reshape(_xxpix.shape))
    fig.savefig('test2d.png') 
    return None 


def ell(ws, thetas, yobs): 
    ''' loss function 
    '''
    if len(thetas.shape) == 1 and thetas.shape[0] > 2: 
        thetas = thetas.reshape((int(thetas.shape[0]/2), 2))
    #print('ell Psi', Psi(ws, np.atleast_2d(thetas)))
    #print(thetas)
    #print(thetas[None,:])
    #print([tt for tt in thetas[None,:]])
    #print([(w, tt) for w, tt in zip(ws, thetas[None,:])])
    return ((Psi(ws, thetas) - yobs)**2).sum() 


def gradell(ws, thetas, yobs):  
    ''' gradient of the loss fucntion 
    '''
    return (Psi(ws, thetas) - yobs)/((Psi(ws, thetas) - yobs)**2).sum() 


def lmo(v): 
    ''' step 1 of ADCG: "linear maximization oracle". This function does the following 
    optimization 
    
    argmin < psi(theta), v > 

    where for ADCG, v = the gradient of loss. For simplicity, we grid up theta to 
    theta_grid and calculate grid_psi minimize the inner product 
    '''
    ip = (grid_psi @ v) 
    return theta_xygrid[ip.argmin()] 


def coordinate_descent(thetas, yobs, lossFn, iter=35, min_drop=1e-5, **lossfn_kwargs):  
    ''' step 2 of ADCG (nonconvex optimization using block coordinate descent algorithm).
    compute weights, prune support, locally improve support
    '''
    def min_ws(): 
        # non-negative least square solver to find the weights that minimize loss 
        return scipy.optimize.nnls(np.stack([psi(tt) for tt in thetas]).T, yobs)[0]

    def min_thetas(): 
        res =  scipy.optimize.minimize(
                Agrad.value_and_grad(lambda tts: lossFn(ws, tts, yobs, **lossfn_kwargs)), thetas, 
                jac=True, method='L-BFGS-B', bounds=[(0.0, 1.0)]*2*thetas.shape[0])
        return res['x'], res['fun']

    old_f_val = np.Inf
    for i in range(iter): 
        thetas = np.atleast_2d(thetas)

        ws = min_ws() # get weights that minimize loss

        thetas, f_val = min_thetas() # keeping weights fixed, minimize loss 
    
        if len(thetas.shape) == 1 and thetas.shape[0] > 2: 
            thetas = thetas.reshape((int(thetas.shape[0]/2), 2))

        if old_f_val - f_val < min_drop: # if loss function doesn't improve by much
            break 
        old_f_val = f_val.copy()
    return ws, thetas 


def adcg(yobs, lossFn, gradlossFn, local_update, max_iters, **lossfn_kwargs): 
    ''' Alternative Descent Conditional Gradient 
    '''
    thetas, ws = np.zeros(0), np.zeros(0) 
    output = np.zeros(len(xypix)) 

    history = [] 
    for i in range(max_iters): 
        residual = output - yobs
        loss = lossFn(ws, thetas, yobs, **lossfn_kwargs) 
        print('iter=%i, loss=%f' % (i, loss)) 
        history.append((loss, ws, thetas))
    
        # get gradient of loss function 
        grad = gradlossFn(ws, thetas, yobs, **lossfn_kwargs) 
        # compute new support
        theta = lmo(grad)
        # update signal parameters  
        if i == 0: _thetas = np.append(thetas, theta)
        else: _thetas = np.append(np.atleast_2d(thetas), np.atleast_2d(theta), axis=0)

        ws, thetas = local_update(_thetas, yobs, lossFn, **lossfn_kwargs)

        # calculate output 
        output = Psi(ws, thetas)
    return history 


def select_k(history): 
    drop = np.array([history[i][0] - history[i+1][0] for i in range(len(history)-1)])

    if np.sum(drop < 0.1) == 0: 
        print(drop)
        raise ValueError('need more iterations') 

    k_hat = np.argmax(drop < 0.1)
    return history[k_hat]


def eg2d_ADCG(N_source=3, sig_noise=0.2, seed=1, max_iter=20): 
    ''' ADCG on simple 2D problem 
    '''
    thetas_true, weights_true, yobs = obs2d(N_source=N_source, sig_noise=sig_noise, seed=seed)
    ytrue = Psi(weights_true, thetas_true) 
    print(thetas_true, weights_true)

    fdensity = float(N_source) / float(theta_xygrid.shape[0]) # for now lets assume we know the density perfectly. 

    hist = adcg(yobs, ell, gradell, coordinate_descent, max_iter)

    loss, ws, thetas = select_k(hist)
    output_adcg = Psi(ws, thetas) 

    # plot data 
    fig = plt.figure(figsize=(15,5))
    sub = fig.add_subplot(141)
    sub.imshow(yobs.reshape(_xxpix.shape))
    sub.set_title(r'$y_{\rm obs}$', fontsize=20) 

    sub = fig.add_subplot(142)
    sub.imshow(ytrue.reshape(_xxpix.shape))
    sub.set_title(r'$y_{\rm true}$', fontsize=20) 

    sub = fig.add_subplot(143) 
    sub.imshow(output_adcg.reshape(_xxpix.shape))
    sub.set_title(r'$y_{\rm adcg}$', fontsize=20) 
    
    sub = fig.add_subplot(144) 
    sub.imshow((ytrue - output_adcg).reshape(_xxpix.shape))
    sub.set_title(r'$y_{\rm true} - y_{\rm adcg}$', fontsize=20) 

    ffig = os.path.join(UT.fig_dir(), 
            'obs2d.psf%s.nois%s.Nsource%i.seed%i.%s.adcg.png' % (str(sig_psf), str(sig_noise), N_source, seed, 'ell'))
    fig.savefig(ffig, 
            bbox_inches='tight')
    return None 


if __name__=="__main__": 
    #test_obs2d(N_source=5, sig_noise=0.01, seed=1)
    eg2d_ADCG(N_source=3, sig_noise=0.2, seed=2, max_iter=30)
