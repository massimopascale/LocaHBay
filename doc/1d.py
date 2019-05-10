'''

simple 1d localization problem

'''
import os 
import scipy.optimize
import autograd as Agrad
import autograd.numpy as np 
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


xpix = np.linspace(0., 1., 50) # default pixel gridding 
sig_psf = 0.1 # psf width
sig_nois = 0.1 # noise level 

def psi(theta): 
    ''' measurement model, which in our case is just a 1d gaussian of width 
    sigma (PSF) written out to xpix 
    '''
    return np.exp(-((xpix - theta)/sig_psf)**2)

theta_grid = np.linspace(0., 1., 30) 
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

    yobs = Psi(weights, thetas) + sig_nois * np.random.randn(len(xpix)) 
    
    fobs = os.path.join(UT.dat_dir(), 'obs1d.psf%s.nois%s.Nsource%i.seed%i.npy' % 
            (str(sig_psf), str(sig_nois), N_source, seed))
    np.save(fobs, yobs) 
    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(xpix, yobs, marker='x', s=10) 
    for i in range(N_source): 
        plt_true, = sub.plot([thetas[i], thetas[i]], [0., 2.5], c='k', ls='--') 
    sub.legend([plt_true], [r'$x_{\rm true}$'], loc='upper right', fontsize=25) 
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(0., 1.) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    fig.savefig(os.path.join(UT.fig_dir(), os.path.basename(fobs).replace('.npy', '.png')), bbox_inches='tight')
    return None 
    

def ell(ws, thetas, yobs): 
    ''' loss function 
    '''
    return ((Psi(ws, thetas) - yobs)**2).sum() 


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


def coordinate_descent(thetas, yobs, iter=35, min_drop=1e-5):  
    ''' block coordinate descent algorithm. 
    compute weights, prune support, locally improve support
    '''
    def min_ws(): 
        # non-negative least square solver to find the weights that minimize loss 
        return scipy.optimize.nnls(np.stack([psi(tt) for tt in thetas]).T, yobs)[0]
    def min_thetas(): 
        res =  scipy.optimize.minimize(
                Agrad.value_and_grad(lambda tts: ell(ws, tts, yobs)), thetas, 
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


def adcg(yobs, local_update, max_iters): 
    ''' Alternative Descent Conditional Gradient method implementation 
    carefully following Nick Boyd's tutorial. Mainly involves two optimization 
    problems
    '''
    thetas, ws = np.zeros(0), np.zeros(0) 
    output = np.zeros(len(xpix)) 
    history = [] 
    for i in range(max_iters): 
        residual = output - yobs
        loss = (residual**2).sum() 
        print('iter=%i, loss=%f' % (i, loss)) 
        history.append((loss, ws, thetas))

        theta = lmo(residual) 
        ws, thetas = local_update(np.append(thetas, theta), yobs)
        output = Psi(ws, thetas)
    return history 


def select_k(history): 
    drop = np.array([history[i][0]-history[i+1][0] for i in range(len(history)-1)])
    k_hat = np.argmax(drop<0.1)
    return history[k_hat]


def runADCG(): 
    fobs = os.path.join(UT.dat_dir(), 'obs1d.psf%s.nois%s.Nsource%i.seed%i.npy' % 
            (str(sig_psf), str(sig_nois), 5, 1))
    yobs = np.load(fobs) 
    hist = adcg(yobs, coordinate_descent, 10)
    loss, ws, thetas = select_k(hist)
    output = Psi(ws, thetas) 

    # plot data 
    fig = plt.figure(figsize=(10,5))
    sub = fig.add_subplot(111)
    sub.scatter(xpix, yobs, marker='x', s=10) 
    sub.plot(xpix, output, c='k') 
    for i in range(len(thetas)): 
        sub.plot([thetas[i], thetas[i]], [0., 2.5], c='k', ls='--') 
    loss, ws, thetas = hist[-1]
    for i in range(len(thetas)): 
        sub.plot([thetas[i], thetas[i]], [0., 2.5], c='C0', ls='--') 
    sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
    sub.set_xlim(0., 1.) 
    sub.set_ylabel('intensity', fontsize=25) 
    sub.set_ylim(0., 2.5) 
    fig.savefig(os.path.join(UT.fig_dir(), os.path.basename(fobs).replace('.npy', '.adcg.png')), bbox_inches='tight')
    return None 



if __name__=="__main__": 
    #obs1d(N_source=5, seed=1)
    runADCG()
