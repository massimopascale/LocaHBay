'''

simple 1d localization problem

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


xpix = np.linspace(0., 1., 200) # default pixel gridding 
sig_psf = 0.02 # psf width

def psi(theta): 
    ''' measurement model, which in our case is just a 1d gaussian of width 
    sigma (PSF) written out to xpix 
    '''
    return np.exp(-((xpix - theta)/sig_psf)**2)

theta_grid = np.linspace(0., 1., len(xpix)) 
grid_psi = np.stack([psi(tt) for tt in theta_grid])  

def Psi(ws, thetas): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    return np.sum(np.array([w * psi(tt) for (w,tt) in zip(ws, thetas)]),0)


def obs1d(N_source=5, sig_noise=0.2, seed=1):  
    ''' generate and write out 1d observations on a xpix. Takes positions 
    and "intensities" (weights) and convolves them with a PSF and adds noise 
    '''
    np.random.seed(seed)
    thetas = np.random.rand(N_source) # x_true positions 
    weights = np.random.rand(N_source)*2 # weights --- in SMLM intensities 
    return  thetas, weights, Psi(weights, thetas) + sig_noise * np.random.randn(len(xpix)) 
    

def ell(ws, thetas, yobs): 
    ''' loss function 
    '''
    return ((Psi(ws, thetas) - yobs)**2).sum() 


def ell_noise(ws, thetas, yobs, sig_noise=0.2): 
    ''' loss function  = likelihood 
    '''
    return ((Psi(ws, thetas) - yobs)**2/sig_noise**2).sum() 


def ell_sparseprior(ws, thetas, yobs, **kwargs): 
    ''' loss function with a sparse prior 
    '''
    fdensity = kwargs['fdensity']
    def lnlike(ws):     
        return -0.5 * np.sum((Psi(ws, theta_grid) - yobs)**2/kwargs['sig_noise']**2) 

    def prior_i(wi): 
        ''' log of Poisson prior for an indivudial pixel
        '''
        pri = 0.
        if 0. < wi <= 2.: 
            pri += fdensity # p1 term 
        if wi > 0: 
            pri += (1.-fdensity) * gaussian(np.log(wi), loc=-4., scale=0.75)/wi
        return pri

    def lnprior(ws): 
        return np.sum([np.log(prior_i(w)) for w in ws])

    return -1.*(lnlike(ws) + lnprior(ws)) 


def lmo(v): 
    ''' "linear maximization oracle"
    This function does the following optimization 
    
    argmin < psi(theta), v > 

    where for ADCG, v = the gradient of loss. Since this is a 1D problem, 
    we grid up theta to theta_grid and calculate grid_psi minimize the 
    inner product 
    '''
    ip = np.matmul(grid_psi, v) 
    return theta_grid[ip.argmin()] 


def coordinate_descent(thetas, yobs, lossFn, iter=35, min_drop=1e-5, **lossfn_kwargs):  
    ''' block coordinate descent algorithm. 
    compute weights, prune support, locally improve support
    '''
    def min_ws(): 
        # non-negative least square solver to find the weights that minimize loss 
        return scipy.optimize.nnls(np.stack([psi(tt) for tt in thetas]).T, yobs)[0]
    def min_thetas(): 
        res =  scipy.optimize.minimize(
                Agrad.value_and_grad(lambda tts: lossFn(ws, tts, yobs, **lossfn_kwargs)), thetas, 
                jac=True, method='L-BFGS-B', bounds=[(0.0, 1.0)]*len(thetas))
        return res['x'], res['fun']
    
    old_f_val = np.Inf
    for i in range(iter): 
        ws = min_ws() # get weights that minimize loss
        thetas, f_val = min_thetas() # keeping weights fixed, minimize loss 
        if old_f_val - f_val < min_drop: # if loss function doesn't improve by much
            break 
        old_f_val = f_val 
    return ws, thetas 


def adcg(yobs, lossFn, local_update, max_iters, **lossfn_kwargs): 
    ''' Alternative Descent Conditional Gradient method implementation 
    carefully following Nick Boyd's tutorial. Mainly involves two optimization 
    problems
    '''
    thetas, ws = np.zeros(0), np.zeros(0) 
    output = np.zeros(len(xpix)) 
    history = [] 
    for i in range(max_iters): 
        residual = output - yobs
        #loss = (residual**2).sum() 
        loss = lossFn(ws, thetas, yobs, **lossfn_kwargs) 
        print('iter=%i, loss=%f' % (i, loss)) 
        history.append((loss, ws, thetas))

        theta = lmo(residual) 
        ws, thetas = local_update(np.append(thetas, theta), yobs, lossFn, **lossfn_kwargs)
        output = Psi(ws, thetas)
    return history 


def select_k(history): 
    drop = np.array([history[i][0]-history[i+1][0] for i in range(len(history)-1)])
    k_hat = np.argmax(drop<0.1)
    return history[k_hat]


def ADCG_1d(loss_str='l2', N_source=3, sig_noise=0.2, seed=1): 
    '''
    '''
    thetas_true, weights_true, yobs = obs1d(N_source=N_source, sig_noise=sig_noise, seed=seed)
    fdensity = float(N_source) / float(len(theta_grid)) # for now lets assume we know the density perfectly. 

    if loss_str is 'l2': 
        hist = adcg(yobs, ell, coordinate_descent, 30)
    elif loss_str == 'l2_noise':
        hist = adcg(yobs, ell_noise, coordinate_descent, 30)
    elif loss_str == 'l2_noise_sparse': 
        hist = adcg(yobs, ell_sparseprior, coordinate_descent, 30, fdensity=fdensity, sig_noise=sig_noise)
    loss, ws, thetas = select_k(hist)
    output = Psi(ws, thetas) 
    print ws 

    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.errorbar(xpix, yobs, yerr=sig_noise, elinewidth=0.2, fmt='xC0', zorder=1)
    for x, w in zip(thetas_true, weights_true):
        sub.vlines(x, 0, w, color='k', linewidth=1.5, linestyle='--')

    if len(ws) > 0:
        for x, w in zip(thetas, ws):
            _plt = sub.vlines(x, 0, w, color='C1')
        sub.legend([_plt], ['inf. intensities'], loc='upper right', fontsize=15) 
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(-0.1, 1.1) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    ffig = os.path.join(UT.fig_dir(), 
            'obs1d.psf%s.nois%s.Nsource%i.seed%i.%s.adcg.png' % (str(sig_psf), str(sig_noise), N_source, seed, loss_str))
    fig.savefig(ffig, 
            bbox_inches='tight')
    return None 


def gaussian(x, loc=None, scale=None): 
    ''' N(x; loc, scale) 
    '''
    y = (x - loc)/scale
    return np.exp(-0.5 * y**2)/np.sqrt(2.*np.pi)/scale


def SpareBayes_1d(N_source=3, sig_noise=0.2, seed=1, prior_loc=-3):
    '''
    '''
    thetas_true, weights_true, yobs = obs1d(N_source=N_source, sig_noise=sig_noise, seed=seed)
    fdensity = float(N_source) / float(len(theta_grid)) # for now lets assume we know the density perfectly. 

    weights_true_grid = np.repeat(1e-10, len(theta_grid))
    for ww, tt in zip(weights_true, thetas_true):
        weights_true_grid[(np.abs(tt - theta_grid)).argmin()] = ww

    def prior_i(wi, prior_scale=0.2): 
        ''' log of Poisson prior for an indivudial pixel
        '''
        pri = 0.
        if 0. < wi <= 2.: 
            pri += fdensity # p1 term 
        if wi > 0: 
            pri += (1.-fdensity) * gaussian(np.log(wi), loc=prior_loc, scale=prior_scale)/wi
        return pri

    def lnprior(ws, prior_scale=0.2): 
        return np.sum([np.log(prior_i(w, prior_scale=prior_scale)) for w in ws])

    def lnlike(ws):     
        return -0.5 * np.sum((Psi(ws, theta_grid) - yobs)**2/sig_noise**2) 

    def lnpost(ws, prior_scale=0.2): 
        return lnlike(ws) + lnprior(ws, prior_scale=prior_scale)

    tt0 = 2.*np.ones(len(theta_grid)) 
    post_inf = np.inf 
    for prior_scale in np.logspace(-3, 0, 10)[::-1]: 
        res = scipy.optimize.minimize(
                Agrad.value_and_grad(lambda tt: -1.*lnpost(tt, prior_scale=prior_scale)), 
                tt0, # theta initial 
                jac=True, 
                method='L-BFGS-B', 
                bounds=[(1e-5, 1e1)]*len(tt0))
        print('sig=%f, loss=%f' % (prior_scale, res['fun']))
        if res['fun'] > post_inf: break 
        tt0 = res['x']
        post_inf = res['fun'] 
    ws_inf = tt0 
    print 'true', lnpost(weights_true_grid)
    print 'zero', lnpost(np.repeat(1e-5, len(tt0)))

    #loss, ws, thetas = select_k(hist)
    output = Psi(ws_inf, theta_grid) 

    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    #sub.scatter(xpix, yobs, marker='x', s=10) 
    sub.errorbar(xpix, yobs, yerr=sig_noise, elinewidth=0.2, fmt='xC0', zorder=1)
    for x, w in zip(thetas_true, weights_true):
        sub.vlines(x, 0, w, color='k', linewidth=1.5, linestyle='--')
    sub.plot(xpix, output, c='C1', lw=0.5, ls=':') 
    sub.plot(theta_grid, ws_inf, c='C1', label='inf. intensities')
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(-0.1, 1.1) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    ffig = os.path.join(UT.fig_dir(), 
            'obs1d.psf%s.nois%s.Nsource%i.seed%i.sparsebayes.png' % (str(sig_psf), str(sig_noise), N_source, seed))
    fig.savefig(ffig, bbox_inches='tight')
    return None 


if __name__=="__main__": 
    #ADCG_1d(loss_str='l2_noise')
    #ADCG_1d(loss_str='l2_noise_sparse', N_source=5, sig_noise=0.2, seed=1)
    for seed in range(10): 
        ADCG_1d(loss_str='l2', N_source=5, sig_noise=0.3, seed=seed)
        SpareBayes_1d(N_source=5, sig_noise=0.3, prior_loc=-3, seed=seed)
