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
sig_nois = 0.1 # noise level 

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


def obs1d(N_source=5, seed=1):  
    ''' generate and write out 1d observations on a xpix. Takes positions 
    and "intensities" (weights) and convolves them with a PSF and adds noise 
    '''
    np.random.seed(seed)
    thetas = np.random.rand(N_source) # x_true positions 
    weights = np.random.rand(N_source)+1 # weights --- in SMLM intensities 
    return  Psi(weights, thetas) + sig_nois * np.random.randn(len(xpix)) 
    

def ell(ws, thetas, yobs): 
    ''' loss function 
    '''
    return ((Psi(ws, thetas) - yobs)**2).sum() 


def ell_noise(ws, thetas, yobs): 
    ''' loss function  = likelihood 
    '''
    return ((Psi(ws, thetas) - yobs)**2/sig_nois**2).sum() 


def ell_sparseprior(ws, thetas, yobs, **kwargs): 
    ''' loss function with a sparse prior 
    '''
    f_density = kwargs['f_density']
    lnlike = ((Psi(ws, thetas) - yobs)**2/sig_nois**2).sum() 

    def p1(_I): # the intensity function is uniform between 1 and 2 
        _I = np.atleast_1d(_I)
        _p1 = np.zeros_like(_I) 
        _p1[(_I >= 1.) & (_I <= 2.)] = 1.
        return _p1

    #def p2(_I): 
    #    _p2 = lambda ii: p1(_I - ii) * p1(ii) 
    #    return sp.integrate.quad(_p2, -np.inf, np.inf)[0]

    def lnPrior(ws): 
        wlim = (ws > 0.)
        _p1 = np.array([p1(w) for w in ws[wlim]])
        #p2tt = np.array([p2(ttt) for ttt in tt])
        # ignore second order term for now 
        pri = np.sum(np.log((1.-f_density) * np.clip(Norm.pdf(np.log(ws[wlim]), loc=-10, scale=1), 1e-5, None)/ws[wlim] + f_density * _p1)) 
        if np.isfinite(pri): 
            return -pri
        else: 
            raise ValueError
    lnprior = lnPrior(ws) 
    print 'like', lnlike
    return lnlike + lnprior


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


def ADCG_1d(loss_str='l2', N_source=3, seed=1, **lossfn_kwargs): 
    '''
    '''
    np.random.seed(seed)
    thetas_true = np.random.rand(N_source) # x_true positions 
    weights_true = np.random.rand(N_source)+1 # weights --- in SMLM intensities 
    yobs =  Psi(weights_true, thetas_true) + sig_nois * np.random.randn(len(xpix)) 

    if loss_str is 'l2': 
        hist = adcg(yobs, ell, coordinate_descent, 10)
    elif loss_str == 'l2_noise':
        hist = adcg(yobs, ell_noise, coordinate_descent, 30)
    elif loss_str == 'l2_noise_sparse': 
        hist = adcg(yobs, ell_sparseprior, coordinate_descent, 30, **lossfn_kwargs)
    loss, ws, thetas = select_k(hist)
    output = Psi(ws, thetas) 

    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(xpix, yobs, marker='x', s=10) 
    for i in range(len(thetas_true)):
        sub.plot(xpix, weights_true[i] * psi(thetas_true[i]), c='k', ls=':', lw=0.5)

    sub.plot(xpix, output, c='k') 
    for i in range(len(thetas)): 
        sub.plot([thetas[i], thetas[i]], [0., 2.5], c='k', ls='--') 
        #sub.plot(xpix, ws[i] * psi(thetas[i]), c='k', ls=':', lw=0.5)
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(0., 1.) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    fobs = os.path.join(UT.dat_dir(), 'obs1d.psf%s.nois%s.Nsource%i.seed%i.npy' % 
            (str(sig_psf), str(sig_nois), N_source, seed))
    fig.savefig(os.path.join(UT.fig_dir(), os.path.basename(fobs).replace('.npy', '.%s.adcg.png' % loss_str)), 
            bbox_inches='tight')
    return None 


def _Norm(x, loc=None, scale=None): 
    ''' N(x; loc, scale) 
    '''
    y = (x - loc)/scale
    return np.exp(-y**2./2.)/np.sqrt(2.*np.pi)/scale


def Post_sparseprior(ws, yobs, f_density=0., silent=True): 
    ''' Spare Bayes posterior with a sparse prior 
    '''
    lnlike = ((Psi(ws, theta_grid) - yobs)**2/sig_nois**2).sum() 

    def p1(_I): # the intensity function is uniform between 1 and 2 
        _I = np.atleast_1d(_I)
        _p1 = np.zeros_like(_I) 
        _p1[(_I >= 0.) & (_I <= 2.)] = 0.5
        return _p1

    #def p2(_I): 
    #    _p2 = lambda ii: p1(_I - ii) * p1(ii) 
    #    return sp.integrate.quad(_p2, -np.inf, np.inf)[0]

    def lnPrior(ws): 
        wlim = (ws > 0.)
        _p1 = p1(ws) 
        #_p1 = np.array([p1(w) for w in ws[wlim]])
        #p2tt = np.array([p2(ttt) for ttt in tt])
        # ignore second order term for now 
        #pri = np.sum(np.log((1.-f_density) * np.clip(_Norm(np.log(ws[wlim]), loc=-10, scale=1-3), 1e-5, None)/ws[wlim] + f_density * _p1)) 
        pri0 = (1.-f_density) * np.clip(_Norm(np.log(ws), loc=np.log(1e-10), scale=1e-5), 0, None)/ws
        pri1 = f_density * _p1
        pri = np.sum(np.log(pri0 + pri1)) 
        #print 'w', ws 
        #print '0', pri0
        if not silent: print 'p1', pri1
        #print (pri0+pri1).min() 
        if np.isfinite(pri): 
            return -pri
        else: 
            raise ValueError
    lnprior = lnPrior(ws) 
    #print lnlike + lnprior
    if not silent: print 'like', lnlike
    #print 'prior', lnprior
    return lnlike + lnprior


def SpareBayes_1d(N_source=3, seed=1, f_density=0.015): 
    '''
    '''
    np.random.seed(seed)
    thetas_true = np.random.rand(N_source) # x_true positions 
    weights_true = 2.*np.random.rand(N_source) # weights --- in SMLM intensities 1 to 2
    yobs =  Psi(weights_true, thetas_true) + sig_nois * np.random.randn(len(xpix)) 

    weights_true_grid = np.repeat(1e-10, len(theta_grid))
    for ww, tt in zip(weights_true, thetas_true):
        weights_true_grid[(np.abs(tt - theta_grid)).argmin()] = ww
    
    tt0 = np.repeat(1e-10, len(theta_grid)) #2.*np.ones(len(theta_grid))
    res = scipy.optimize.minimize(
            Agrad.value_and_grad(lambda tts: Post_sparseprior(tts, yobs, f_density=f_density, silent=False)), 
            tt0, # theta initial 
            jac=True, 
            method='L-BFGS-B', 
            bounds=[(1e-10, 1e1)]*len(tt0))
    ws_inf = res['x']
    post_inf = res['fun'] 
    print ws_inf
    print post_inf 
    print 'true', Post_sparseprior(weights_true_grid, yobs, f_density=f_density, silent=False)
    print 'zero', Post_sparseprior(np.repeat(1e-10, len(tt0)), yobs, f_density=f_density)
    #print ell_sparseprior(weights_true, thetas_true, yobs, f_density=f_density)
    #print ((Psi(weights_true, thetas_true) - yobs)**2/sig_nois**2).sum() 
    #print ((Psi(weights_true_grid, theta_grid) - yobs)**2/sig_nois**2).sum() 

    #loss, ws, thetas = select_k(hist)
    output = Psi(ws_inf, theta_grid) 

    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(xpix, yobs, marker='x', s=10) 
    for i in range(len(thetas_true)):
        sub.plot(xpix, weights_true[i] * psi(thetas_true[i]), c='k', ls=':', lw=0.5)
    sub.plot(xpix, output, c='k') 
    sub.plot(theta_grid, ws_inf)
    #print output 
    #print ws_inf 
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(0., 1.) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    fobs = os.path.join(UT.dat_dir(), 'obs1d.psf%s.nois%s.Nsource%i.seed%i.npy' % 
            (str(sig_psf), str(sig_nois), N_source, seed))
    fig.savefig(
            os.path.join(UT.fig_dir(), 
                'obs1d.psf%s.nois%s.Nsource%i.seed%i.sparsebayes.png' % (str(sig_psf), str(sig_nois), N_source, seed)),
            bbox_inches='tight')
    return None 


def delta_test(): 
    
    x = np.logspace(-10, -2, 1000) 
    fig = plt.figure()
    sub = fig.add_subplot(111)
    print _Norm(np.log(x), loc=np.log(1e-10), scale=1e-3)[::-1]
    print (_Norm(np.log(x), loc=np.log(1e-10), scale=1e-3)/x)[::-1]
    sub.plot(x, _Norm(np.log(x), loc=np.log(1e-10), scale=1e-3), c='k')
    sub.plot(x, _Norm(np.log(x), loc=np.log(1e-10), scale=1e-3)/x, c='C1')
    sub.set_xlim(1e-10, 1.e-2) 
    sub.set_xscale('log')
    sub.set_yscale('log') 
    #sub.set_ylim(0., 10.) 
    fig.savefig(os.path.join(UT.fig_dir(), 'delta_test.png'), bbox_inches='tight') 
    return None


if __name__=="__main__": 
    #ADCG_1d(loss_str='l2')
    #ADCG_1d(loss_str='l2_noise')
    #ADCG_1d(loss_str='l2_noise_sparse', f_density=3./200.)
    SpareBayes_1d(f_density=0.015)
