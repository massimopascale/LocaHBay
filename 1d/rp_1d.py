

import autograd as Agrad
import autograd.numpy as np 
import scipy.optimize
import scipy.stats as st
from scipy.integrate import trapz
from mpmath import hyp2f1
from scipy.integrate import simps
import scipy.signal.find_peaks
# -- plotting --- 
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


########################################################################
'''
rp_1d.py
Code modified from sparse_bayes_1d.ipynb
Author: Massimo Pascale
Last Updated: 10/19/2019

Code uses poisson prior and exponential intensity function to determine
point source locations in psf+noise and recover hyperparameters.
'''
########################################################################


def psi(xpos): 
    ''' measurement model, which in our case is just a 1d gaussian of width 
    sigma (PSF) written out to xpix 
    '''
    return np.exp(-((xpix - xpos)/sig_psf)**2)

def gaussian(x, loc=None, scale=None): 
	'''
	scipy's gaussian pdf didn't work idk
	'''
	y = (x - loc)/scale
	return np.exp(-0.5*y**2)/np.sqrt(2.*np.pi)/scale

def trapezoidal(f, a, b, n):
    h = float(b - a) / n
    s = 0.0
    s += f(a)/2.0
    for i in range(1, n):
        s += f(a + i*h)
    s += f(b)/2.0
    return s * h
    
def Psi(ws, xs): 
    ''' "forward operator" i.e. forward model 
    
    Psi = int psi(theta) dmu(theta) 

    where mu is the signal parameter
    '''
    return np.sum(np.array([w * psi(tt) for (w,tt) in zip(ws, xs)]),0)

def prior_i(w,fdensity,alpha,sig,interval_grid):
	'''
	log of Poisson prior for an indivudial pixel
	'''
	pri=0.;
	if 0. < w <= max(interval_grid):
		norm = (max(interval_grid)**(alpha+1) - min(interval_grid)**(alpha+1))/(alpha+1); #normalization of mass function
		p1 = lambda x: x**alpha /norm #probability of single source
		#now probability of second source
		p2 = 1#trapezoidal(lambda x:p1(w)*p1(x-w), min(interval_grid), max(interval_grid), 100);
		pri += fdensity*p1(w) + p2*fdensity**2
	if w > 0:
		pri += (1.-fdensity - fdensity**2 ) *gaussian(np.log(w),loc=-4., scale=sig)/w
		return pri
    
def lnprior(ws,fdensity,alpha,sig,interval_grid): 
	'''
	calculate log of prior
	'''
	return np.sum([np.log(prior_i(w,fdensity,alpha,sig,interval_grid)) for w in ws])

def lnlike(ws): 
    ''' log likelihood 
    '''
    return -0.5 * np.sum((Psi(ws, theta_grid) - data)**2/sig_noise**2)
 
def lnpost(ws,fdensity,alpha,sig,interval_grid): 
	'''
	posterior prob
	'''
	return lnlike(ws) + lnprior(ws,fdensity,alpha,sig,interval_grid)

########################################################################
#now begin the actual execution
########################################################################
np.random.seed(1)
Ndata = 15
xpix = np.linspace(0., 1., 500) # pixel gridding 
fdensity_true = float(Ndata)/float(len(xpix)); #number density of obj in 1d

sig_psf = 0.02 # psf width
sig_noise = 0.2 # noise level

#create coordinate grid
theta_grid = np.linspace(0., 1., len(xpix)) # gridding of theta (same as pixels)
grid_psi = np.stack([psi(tt) for tt in theta_grid])

#create true values - assign to grid
x_true = np.random.rand(Ndata) # location of sources

#sample intensities from power law function, but assume b/t 1 and 5:
w_interval = (1,5)
w_grid = np.linspace(w_interval[0],w_interval[1],100)
alpha = 2;
w_norm = (w_interval[1]**(alpha+1) - w_interval[0]**(alpha+1))/(alpha+1);
w_func = np.power(w_grid,alpha)/w_norm;
w_true = w_norm*np.random.choice(w_func,Ndata);

#true grid needs to be set up with noise
w_true_grid = np.zeros(len(theta_grid))
for x, w in zip(x_true, w_true): 
    w_true_grid[np.argmin(np.abs(theta_grid - x))] = w
data = Psi(w_true, x_true) + sig_noise * np.random.randn(len(xpix))


#now we begin the optimization
tt0 = np.ones(len(theta_grid)) #begin with high uniform M
f_curr = 0.3;
alpha_curr = 0.2;
sig_delta = 0.75;
step = 0;
#now we begin optimizing step by step, beginning with M, then f, then alpha
while(True):
	#start with m
	res = scipy.optimize.minimize(
			Agrad.value_and_grad(lambda tt: -1.*lnpost(tt,f_curr,alpha_curr,sig_delta,w_grid)),  
			tt0, # theta initial 
			jac=True, 
			method='L-BFGS-B', 
			bounds=[(1e-5, 5)]*len(tt0))
	tt_prime = res['x']
	print('midstep');
	#step f by clipping the data
	m_inds = scipy.signal.find_peaks(tt_prime);
	f_prime = len(m_inds)/len(xpix);
	tt_prime = tt_prime[m_inds];
	#step f and alpha simultaneously
	res = scipy.optimize.minimize(
			Agrad.value_and_grad(lambda x: -1.*lnpost(tt_prime,x,sig_delta,w_grid)),  
			(alpha_curr), # fdensity and alpha initial 
			jac=True, 
			method='L-BFGS-B', 
			bounds=[(-10, 10)])
	f_prime,alpha_prime = res['x']
	#calculate difference for error threshold
	diff_f = abs(f_prime-f_curr);
	diff_alpha = abs(alpha_prime - alpha_curr);
	diff_t = abs(tt_prime - tt0);
	diff_w = np.sqrt(diff_t.dot(diff_t));
	if diff_w+diff_f+diff_alpha < 1e-4:
		break;
	#if not passed, then update values and step
	tt0 = tt_prime;
	alpha_curr = alpha_prime;
	f_curr = f_prime;
	step+=1;
	print(step);

print(alpha_prime);
print(f_prime);

# plot data 
fig = plt.figure(figsize=(10,5))
sub = fig.add_subplot(111)
sub.scatter(xpix, data, marker='x', s=10) 
for x, w in zip(x_true, w_true): 
    sub.vlines(x, 0, w, color='k')
sub.plot(theta_grid, tt_prime, c='C1', ls='--')
sub.set_xlabel(r'$x_{\rm pix}$', fontsize=25) 
sub.set_xlim(0., 1.) 
sub.set_ylabel('intensity', fontsize=25) 
sub.set_ylim(0., 2.5) 
plt.show();
